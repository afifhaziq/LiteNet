import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import LiteNet, QuantizedLiteNet
from data_processing import preprocess_data
from train import get_time, train_model, test_and_report
import time
import wandb
import random
import argparse
import copy
import yaml
from ptflops import get_model_complexity_info
import gc
import torch.ao.quantization
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

    # Flatten the tensor into rows and columns (rows = Inputs, columns = Outputs)
    # So it becomes a matrix of shape (num_rows, num_cols) with rows representing the weight tensor's inputs and columns representing the weight tensor's outputs
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


def apply_2_4_sparsity_to_conv1d_weight(weight: torch.Tensor) -> torch.Tensor:
    """
    Applies 2:4 sparsity to a Conv1d weight tensor along its GEMM inner dimension
    (in_channels * kernel_size) for each output channel.

    Expected Conv1d weight shape: (out_channels, in_channels, kernel_size)
    """
    if weight.dim() != 3:
        return weight

    out_channels, in_channels, kernel_size = weight.shape
    # Flatten to 2D so that the last dimension corresponds to the GEMM K dimension
    weight_2d = weight.view(out_channels, in_channels * kernel_size).contiguous()
    pruned_2d = apply_2_4_sparsity_to_tensor(weight_2d)
    return pruned_2d.view_as(weight)


def prune_model(model):
    print("\n--- Applying 2:4 sparsity to Linear layers and Conv1d inception branches ---")

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            with torch.no_grad():
                module.weight.data = apply_2_4_sparsity_to_tensor(module.weight.data)
            total_elements = module.weight.numel()
            nonzero_elements = torch.count_nonzero(module.weight.data).item()
            current_sparsity = 1.0 - (nonzero_elements / total_elements)
            print(f"  Applied 2:4 sparsity to {name} (Linear) | Sparsity: {current_sparsity:.2%}")
            wandb.log({f"sparsity_layer/{name}": current_sparsity})
        elif isinstance(module, nn.Conv1d) and name.startswith((
            "branch1x1", "branch3x3", "branch5x5", "branch_pool"
        )):
            with torch.no_grad():
                pruned_w = apply_2_4_sparsity_to_conv1d_weight(module.weight.data)
                module.weight.data.copy_(pruned_w)
            total_elements = module.weight.numel()
            nonzero_elements = torch.count_nonzero(module.weight.data).item()
            current_sparsity = 1.0 - (nonzero_elements / total_elements)
            print(f"  Applied 2:4 sparsity to {name} (Conv1d) | Sparsity: {current_sparsity:.2%}")
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

# --- 3. Main Execution Block ---
if __name__ == '__main__':
    # --- Argparser and Config ---
    parser = argparse.ArgumentParser(description='LiteNet Pruning, Fine-tuning, and Quantization')
    parser.add_argument('--dataset_name', type=str, help='(Optional) Override the active_dataset from config.yaml.')
    parser.add_argument('--quantization', type=str, default='None', choices=['None', 'FP16', 'INT8'], help='Type of quantization to apply after fine-tuning. INT8 is only supported for CPU.')
    parser.add_argument('--quantize-only', action='store_true', help='Skip pruning and fine-tuning, and load a pre-existing fine-tuned model for quantization.')
    parser.add_argument('--path', type=str, default=None, help='Path to the input model. Overrides default path generation.')
    args = parser.parse_args()

    # --- Load Base Config from YAML ---
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # --- Override/Set Config with CLI Args ---
    dataset_name = args.dataset_name if args.dataset_name else config['active_dataset']
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
    # Define the output path for the pruned and fine-tuned model
    config['model_path_pruned_finetuned'] = f"saved_dict/LiteNet_{dataset_name}_FullPruned_finetuned_embedding.pth"

    # Determine the correct input model path based on flags
    if args.quantize_only:
        # If quantizing only, the input is the fine-tuned model
        if args.path:
            user_path = args.path
            if '/' in user_path or '\\' in user_path:
                config['input_model_path'] = user_path
            else:
                config['input_model_path'] = f"saved_dict/{user_path}"
        else:
            # Default to the standard fine-tuned model path
            config['input_model_path'] = config['model_path_pruned_finetuned']
    else:
        # If running the full pipeline, the input is the original pre-trained model
        if args.path:
            user_path = args.path
            if '/' in user_path or '\\' in user_path:
                config['input_model_path'] = user_path
            else:
                config['input_model_path'] = f"saved_dict/LiteNet_{dataset_name}_embedding.pth"
        else:
            # Default to the standard pre-trained model path
            config['input_model_path'] = f"saved_dict/LiteNet_{dataset_name}_embedding.pth"

    # --- WANDB Initialization ---
    seed_everything(config['seed'])
    wandb.init(project="LiteNet-" + dataset_name + "_prune_finetune", mode="online", tags=[f"2:4_Linear_{config['quantization']}"])
    wandb.config.update(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Configuration ---")
    for key, value in config.items():
        if key != 'datasets': # Don't print the whole datasets dict
            print(f"  {key}: {value}")
    print(f"Using device: {device}")
    print("-" * 21)
    
    # --- Data Loading and Feature Selection ---
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
        
        print(f"Loading original pre-trained model from: {config['input_model_path']}")
        model.load_state_dict(torch.load(config['input_model_path']))


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

    
    path_to_load = config['model_path_pruned_finetuned'] if not args.quantize_only else config['input_model_path']

    print(f"Loading best fine-tuned model from: {path_to_load}")
    final_model_fp32.load_state_dict(torch.load(path_to_load))
    
    # --- Step 3: Calculate Statistics on FP32 Model ---
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

    # --- Step 4: Apply Quantization (Optional) ---
    final_model = final_model_fp32
    
    if config['quantization'] == 'FP16':
        # --- FP16 Quantization (Original Logic) ---
        print("\n--- Applying FP16 Quantization to the model ---")
        final_model = quantize_fp16(copy.deepcopy(final_model_fp32))
        print("FP16 quantization complete.")
        

    elif config['quantization'] == 'INT8':
        final_model = quantize_int8_static(final_model_fp32, train_loader, device)

    # --- Step 5: Test the Final Model ---
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

   
    # --- Step 6: Export to ONNX ---
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