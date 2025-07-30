import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import LiteNet, QuantizedLiteNet
from data_processing import preprocess_data
from train import get_time, evaluate_model, train_model, test_and_report
import time
import wandb
from sklearn.metrics import classification_report, confusion_matrix
import random
import argparse
import copy
import yaml
from sklearn import metrics
from ptflops import get_model_complexity_info
import gc
import torch.ao.quantization
from tqdm import tqdm
from quantize import quantize_fp16, quantize_int8_static

# --- 1. Utility Functions ---

def seed_everything(seed: int) -> None:
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 2. Argparser and Config ---
parser = argparse.ArgumentParser(description='LiteNet Pruning, Fine-tuning, and Quantization')
parser.add_argument('--data', type=str, help='(Optional) Override the active_dataset from config.yaml.')
parser.add_argument('--quantization', type=str, default='None', choices=['None', 'FP16', 'INT8'], help='Type of quantization to apply after fine-tuning. INT8 is only supported for CPU.')
parser.add_argument('--quantize-only', action='store_true', help='Skip pruning and fine-tuning, and load a pre-existing fine-tuned model for quantization.')
args = parser.parse_args()

# --- Load Base Config from YAML ---
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# --- Override/Set Config with CLI Args ---
dataset_name = args.data if args.data else config['active_dataset']
config['dataset_name'] = dataset_name # Keep track of the active dataset
config['quantization'] = args.quantization

# --- Get Dataset-Specific Settings ---
try:
    dataset_config = config['datasets'][dataset_name]
    config.update(dataset_config) # Merge dataset-specific settings into main config
except KeyError:
    print(f"Error: Dataset '{dataset_name}' not found in config.yaml under the 'datasets' key.")
    exit()

# --- Set derived config values ---
#sequence = config['sequence']
#features = config['features']
#num_features = sequence * features
# Use a more descriptive base name for the output files this script generates
#base_output_name = f"saved_dict/LiteNet_{dataset_name}"

# Corrected path for loading the original model, matching the files in saved_dict
config['model_path'] = f"saved_dict/LiteNet_{dataset_name}_embedding.pth" 


config['model_path_pruned_finetuned'] = f"saved_dict/LiteNet_{dataset_name}_pruned_finetuned_embedding.pth"

# --- 3. WANDB Initialization ---
seed_everything(config['seed'])
wandb.init(project="LiteNet-" + dataset_name + "_prune_finetune", mode="offline", tags=[f"2:4_Linear_{config['quantization']}"], group='PruneQuant')
wandb.config.update(config)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Configuration ---")
for key, value in config.items():
    if key != 'datasets': # Don't print the whole datasets dict
        print(f"  {key}: {value}")
print(f"Using device: {device}")
print("-" * 21)

# --- 4. Sparsity and Pruning Functions ---

def apply_2_4_sparsity_to_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies 2:4 sparsity to a tensor. For every block of 4 elements,
    sets the 2 smallest magnitude elements to zero.
    This function processes the last dimension of the tensor.
    If the last dimension is not divisible by 4, it processes full blocks.
    """
    if tensor.dim() < 2:
        return tensor

    original_shape = tensor.shape
    num_rows = int(np.prod(original_shape[:-1]))
    num_cols = original_shape[-1]
    
    flat_tensor = tensor.view(num_rows, num_cols).contiguous()
    pruned_flat_tensor = torch.zeros_like(flat_tensor, device=tensor.device) # Ensure tensor is on correct device

    block_size = 4
    for r in range(num_rows):
        for c_start in range(0, num_cols, block_size):
            block_end = min(c_start + block_size, num_cols)
            current_block = flat_tensor[r, c_start:block_end]

            if current_block.numel() == block_size:
                magnitudes = torch.abs(current_block)
                _, topk_indices = torch.topk(magnitudes, 2)
                pruned_flat_tensor[r, c_start + topk_indices[0]] = current_block[topk_indices[0]]
                pruned_flat_tensor[r, c_start + topk_indices[1]] = current_block[topk_indices[1]]
            else:
                # For incomplete blocks, simply copy them without 2:4 constraint.
                # This ensures all weights are preserved if they don't form a full 4-block.
                pruned_flat_tensor[r, c_start:block_end] = current_block

    return pruned_flat_tensor.view(original_shape)


def prune_model(model):
    print("\n--- Applying 2:4 sparsity only to Linear layers ---")
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            with torch.no_grad():
                module.weight.data = apply_2_4_sparsity_to_tensor(module.weight.data)
            print(f"  Applied 2:4 sparsity to {name} (Linear layer)")
            
            # Optional: Log sparsity per layer
            total_elements = module.weight.numel()
            nonzero_elements = torch.count_nonzero(module.weight.data).item()
            current_sparsity = 1.0 - (nonzero_elements / total_elements)
            print(f"    Sparsity for {name}: {current_sparsity:.2%}")
            wandb.log({f"sparsity_layer/{name}": current_sparsity})
        
            
    print("--- Model pruning complete ---")
    return model


def count_nonzero_params(model):
    """Count remaining active parameters and total sparsity."""
    total = 0
    nonzero = 0
    for p in model.parameters():
        if p is not None:
            total += p.numel()
            nonzero += torch.sum(p != 0).item()
    
    overall_sparsity = (total - nonzero) / total if total > 0 else 0
    print(f"Non-zero params: {nonzero}/{total} ({nonzero/total:.1%})")
    print(f"Overall Model Sparsity: {(overall_sparsity):.1%}")
    return overall_sparsity, nonzero

# --- 6. Quantization Functions ---

# --- 7. Main Execution Block ---
if __name__ == '__main__':
    # --- Data Loading and Feature Selection (Corrected) ---
    # This logic now exactly matches main.py
    feature_list_file = f"top_features_{config['dataset_name']}.npy"
    print(f"Loading feature list from: {feature_list_file}")
    most_important_list = np.load(feature_list_file)
    
    print(f"Using {len(most_important_list)} pre-selected features.")

    # Load raw data
    try:
        train_data_npy = np.load(f"dataset/{dataset_name}/train.npy")
        test_data_npy = np.load(f"dataset/{dataset_name}/test.npy")
        val_data_npy = np.load(f"dataset/{dataset_name}/val.npy")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure files are in 'dataset/{dataset_name}/'.")
        exit()
        
    train_loader, test_loader, val_loader, pretime, avgpretime = preprocess_data(
        train_data_npy, test_data_npy, val_data_npy, most_important_list,
        config['batch_size'], dataset_name
    )
    wandb.log({"preprocess_time": float(pretime), "average_preprocess_time": float(avgpretime)})
    del train_data_npy, test_data_npy, val_data_npy
    gc.collect()

    if not args.quantize_only:
        # --- Step 1: Pruning and Fine-Tuning ---
        print("\n--- Running in Full Mode: Pruning and Fine-Tuning ---")
        #model = LiteNet(sequence=config['sequence'], features=config['features'], num_class=config['num_class']).to(device)
        model = LiteNet(sequence=config['sequence'], 
                        features=config['features'], 
                        num_class=config['num_class'],
                        vocab_size=256,
                        embedding_dim=24).to(device)
        
        print(f"Loading original pre-trained model from: {config['model_path']}")
        model.load_state_dict(torch.load(config['model_path']))

        pruned_model = prune_model(copy.deepcopy(model))
        pruned_model = pruned_model.to(device)

        # Reusing train_model for fine-tuning
        print(f"\n--- Starting Fine-Tuning for {config['fine_tune_epochs']} epochs with LR: {config['fine_tune_lr']} ---")
        optimizer = optim.AdamW(pruned_model.parameters(), lr=config['fine_tune_lr'], weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3) #best patience=5
        criterion = nn.CrossEntropyLoss()
        
        train_model(
            model=pruned_model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epoch=config['fine_tune_epochs'], 
            model_path=config['model_path_pruned_finetuned']
        )
    else:
        print("\n--- Running in Quantize-Only Mode: Skipping Pruning and Fine-Tuning ---")


    # --- Step 2: Load the best fine-tuned model for post-processing ---
    print(f"\n--- Final Evaluation and Post-Processing ---")
    final_model_fp32 = model = LiteNet(sequence=config['sequence'], 
                        features=config['features'], 
                        num_class=config['num_class'],
                        vocab_size=256,
                        embedding_dim=24).to(device)
    print(f"Loading best fine-tuned model from: {config['model_path_pruned_finetuned']}")
    #print(final_model_fp32)
    final_model_fp32.load_state_dict(torch.load(config['model_path_pruned_finetuned']))
    
    # --- Step 4: Calculate Statistics on FP32 Model ---
    print("\n--- Calculating Statistics on FP32 Model ---")
    overall_sparsity, total_params = count_nonzero_params(final_model_fp32)
    with torch.cuda.device(0):
        macs, _ = get_model_complexity_info(
            final_model_fp32, ( config['sequence'], config['features']),
            as_strings=False, print_per_layer_stat=False, verbose=False
        )
    estimated_sparse_flops = 2 * macs * (1 - overall_sparsity)
    print(f"Total MACs (dense): {macs:.2e} | Estimated Sparse FLOPs: {estimated_sparse_flops:.2e}")
    wandb.log({"total_parameters_final": total_params, "overall_model_sparsity_final": overall_sparsity, "estimated_dense_MACs": macs, "estimated_sparse_FLOPs": estimated_sparse_flops})

    # --- Step 5: Apply Quantization (Optional) ---
    final_model = final_model_fp32
    
    if config['quantization'] == 'FP16':
        # --- FP16 Quantization (Original Logic) ---
        print("\n--- Applying FP16 Quantization to the model ---")
        final_model = quantize_fp16(copy.deepcopy(final_model_fp32))
        print("FP16 quantization complete.")
        

    elif config['quantization'] == 'INT8':
        final_model = quantize_int8_static(final_model_fp32, train_loader, device)

    # --- Step 6: Test the Final Model ---
    start_time = time.perf_counter()
    final_acc = test_and_report(final_model, test_loader, device, config['classes'])

    time_dif, average_time = get_time(start_time, test=1, data=config['dataset_name'])

    print(f"Testing Time usage: {time_dif:.10f} seconds")  
    print(f"Average Testing time: {average_time:.10f} seconds")
    wandb.log({
        "final_test_accuracy": final_acc, 
        "final_test_time": float(time_dif), 
        "final_average_time_per_batch": float(average_time)
    })

    # --- Step 7: Export to ONNX ---
    print("\n--- Exporting final model to ONNX ---")
    onnx_base_path = config['model_path_pruned_finetuned'].replace(".pth", "")
    onnx_path = f"{onnx_base_path}_{config['quantization']}.onnx"
    
    dummy_input = torch.randint(low=0, high=256, size= (1, config['features']), device=device, dtype=torch.long)
    if config['quantization'] == 'FP16':
        dummy_input = dummy_input.half()
    if config['quantization'] == 'INT8':
        dummy_input = dummy_input.to('cpu')

    final_model.eval()
    try:
        torch.onnx.export(
            final_model.to(dummy_input.device),
            dummy_input, onnx_path, verbose=False, opset_version=13,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Final model exported to ONNX at: {onnx_path}")
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")

    wandb.finish()