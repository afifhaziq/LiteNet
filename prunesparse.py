import torch
import torch.nn as nn # Added for clarity with Linear and Conv1d types
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset # For creating dummy loaders if needed
import numpy as np
# Assuming model.py contains LiteNet
from model import LiteNet
# Assuming data_processing.py contains preprocess_data
from data_processing import preprocess_data
# Assuming train.py contains get_time (or define it here if not available)
# Defining get_time here for completeness if train.py is not strictly available for this script
from train import get_time

import time
import wandb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt # Not directly used in the final script but often good for visualization
import random
import argparse
import copy
from sklearn import metrics
from ptflops import get_model_complexity_info
import gc

# --- 1. Utility Functions ---

def seed_everything(seed: int) -> None:
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(134)

# --- 2. Argparser and Config ---

parser = argparse.ArgumentParser(description='Inception Prune and Fine-tune')
parser.add_argument('--data', type=str, default='ISCXVPN2016', help='input dataset source (e.g., ISCXVPN2016 or MALAYAGT)')
parser.add_argument('--fine_tune_epochs', type=int, default=30, help='Number of epochs for fine-tuning after pruning')
parser.add_argument('--fine_tune_lr', type=float, default=0.0001, help='Learning rate for fine-tuning')

args = parser.parse_args()

# First calculate the derived values
sequence = 1
features = 20
data = args.data
num_features = sequence * features # This should be 20 based on your model's input

config = {
    'sequence': 1,
    'features': 20,
    'learning_rate': 0.001, # Original training LR (might not be used for fine-tuning directly)
    'batch_size': 64,
    'num_class': 10,
    'data': data,
    'num_features': 20,
    'model_path': f"saved_dict/LiteNet_{data}_{num_features}Features_best_model.pth", # Path to your *original* pre-trained model
    'model_path_pruned': f"saved_dict/LiteNet_{data}_{num_features}Features_best_model_pruned_finetuned.pth", # Path for the *final* pruned and fine-tuned model
    'output_path': 'global_relevance.pth', # Not directly used in this script's flow
    'fine_tune_epochs': args.fine_tune_epochs,
    'fine_tune_lr': args.fine_tune_lr
}

# --- 3. WANDB Initialization ---

# Note: The 'amount' tag is now less meaningful for 2:4 sparsity, but keeping it
# if you want to track different experimental runs in WandB.
wandb.init(project="Inception-"+ data + "_prune_finetune", mode="online", tags=["2:4_Linear"], group='FineTune')
wandb.config.update({
    "learning_rate_initial": config['learning_rate'],
    "batch_size": config['batch_size'],
    "fine_tune_epochs": config['fine_tune_epochs'],
    "fine_tune_lr": config['fine_tune_lr'],
    "prune_type": "2:4_Linear", # Explicitly state pruning type
})

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- 4. 2:4 Sparsity Function ---

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

# --- 5. Pruning Function (Linear Layers Only) ---

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

# --- 6. Fine-Tuning Function ---

def fine_tune_model(model, train_loader, val_loader, device, num_epochs, learning_rate, model_save_path):
    print(f"\n--- Starting Fine-Tuning for {num_epochs} epochs with LR: {learning_rate} ---")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    wandb.watch(model, log='all')
    best_val_accuracy = -1.0
    
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        start_time_epoch = time.perf_counter()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device).float()
            labels = labels.to(device).long() # Ensure labels are long for CrossEntropyLoss

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        epoch_train_time, _ = get_time(start_time_epoch, test=0)
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} training finished in {epoch_train_time:.2f}s. Avg Train Loss: {avg_train_loss:.4f}")
        
        # --- Validation after each epoch ---
        model.eval() # Set model to evaluation mode
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device).float()
                labels = labels.to(device).long()
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Validation Accuracy: {val_accuracy:.2f}%, Avg Val Loss: {avg_val_loss:.4f}")
        wandb.log({"fine_tune_epoch": epoch + 1, "fine_tune_loss": avg_train_loss,
                   "fine_tune_val_loss": avg_val_loss, "fine_tune_val_accuracy": val_accuracy})

        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"  Saved best fine-tuned model with accuracy: {best_val_accuracy:.2f}% to {model_save_path}")
            
    print("\n--- Fine-Tuning Complete ---")
    print(f"Best validation accuracy achieved during fine-tuning: {best_val_accuracy:.2f}%")

# --- 7. Testing and Metrics Functions ---

def test_model(model, test_loader, device, classes, data, model_path):
    print(f"\n--- Starting final test on {model_path} ---")
    model.load_state_dict(torch.load(model_path)) # Load the fine-tuned model
    model.eval()

    all_preds, all_labels = [], []

    with torch.inference_mode():
        start_time = time.perf_counter()    
        for images, labels in test_loader:
            images = images.to(device).float()
            labels = labels.to(device).long() # Ensure labels are long

            predictions = model(images)
            preds = torch.argmax(predictions, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    time_dif, average_time = get_time(start_time, test=1, data=data)
    print(f"Testing Time usage: {time_dif:.10f} seconds")  
    print(f"Average Testing time: {average_time:.10f} seconds")
    
    acc = metrics.accuracy_score(all_labels, all_preds)
    print(f"Final Test Accuracy: {acc*100:.2f}%")
    
    wandb.log({"final_test_accuracy": acc})
    wandb.log({"final_test_time": float(time_dif)})
    wandb.log({"final_average_time_per_batch": float(average_time)})

    print('--- Classification Report ---')
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))
    print('--- Confusion Matrix ---')
    print(confusion_matrix(all_labels, all_preds))

def count_nonzero_params(model):
    """Count remaining active parameters and total sparsity."""
    total = 0
    nonzero = 0
    for p in model.parameters():
        if p is not None:
            total += p.numel()
            nonzero += torch.sum(p != 0).item()
    
    overall_sparsity = (total - nonzero) / total
    print(f"Non-zero params: {nonzero}/{total} ({nonzero/total:.1%})")
    print(f"Overall Model Sparsity: {(overall_sparsity):.1%}")
    return overall_sparsity, nonzero

# --- 8. Main Execution Block ---

if __name__ == '__main__':
    # Dataset configuration
    if config['data'] == 'ISCXVPN2016':
        classes = ('AIM Chat','Email','Facebook Audio','Facebook Chat','Gmail Chat',
                    'Hangouts Chat','ICQ Chat','Netflix','Spotify','Youtube')
        feature_file = 'top740featuresISCX.npy'
    else:
        classes = ('Bittorent', 'ChromeRDP', 'Discord', 'EAOrigin', 'MicrosoftTeams',
                    'Slack', 'Steam', 'Teamviewer', 'Webex', 'Zoom')
        feature_file = 'top740featuresMALAYAGT.npy'

    # Load features
    most_important_list = np.load(feature_file)
    most_important_list = [x - 1 for x in most_important_list]
    most_important_list = most_important_list[:config['num_features']]

    # Load raw data
    try:
        train_data_npy = np.load(f"{config['data']}//train.npy", allow_pickle=True)
        test_data_npy = np.load(f"{config['data']}//test.npy", allow_pickle=True)
        val_data_npy = np.load(f"{config['data']}//val.npy", allow_pickle=True)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure data files are in '{config['data']}/' directory.")
        exit() # Exit if data not found
        
    # Preprocess data to get DataLoaders
    # This call now needs to return train_loader and val_loader too.
    # Adjust preprocess_data in data_processing.py if it doesn't already return these.
    # Example signature: preprocess_data(train_data, test_data, val_data, ..., batch_size, ...)
    train_loader, test_loader, val_loader, pretime, avgpretime = preprocess_data(
        train_data_npy, test_data_npy, val_data_npy, most_important_list,
        config['batch_size'], config['data']
    )
    
    wandb.log({"preprocess_time": float(pretime)})
    wandb.log({"average_preprocess_time": float(avgpretime)})

    del train_data_npy, test_data_npy, val_data_npy, most_important_list
    gc.collect() # Free up memory

    # Initialize model
    model = LiteNet(sequence=config['sequence'], 
                  features=config['features'], 
                  num_class=config['num_class']).to(device)

    # Load the original pre-trained model weights
    print(f"Loading original pre-trained model from: {config['model_path']}")
    model.load_state_dict(torch.load(config['model_path']))

    # --- Pruning ---
    pruned_model = prune_model(copy.deepcopy(model)) # Work on a copy of the model for pruning
    pruned_model = pruned_model.to(device) # Ensure pruned model is on GPU

    # --- Fine-tuning ---
    # The fine_tune_model function will save the best fine-tuned model to config['model_path_pruned']
    fine_tune_model(pruned_model, train_loader, val_loader, device,
                    config['fine_tune_epochs'], config['fine_tune_lr'], config['model_path_pruned'])

    # --- Final Evaluation of the Fine-Tuned Pruned Model ---
    print(f"\n--- Final Evaluation after Pruning and Fine-Tuning ---")
    # Reload the best fine-tuned model (in case the last epoch wasn't the best)
    final_model = LiteNet(sequence=config['sequence'], 
                      features=config['features'], 
                      num_class=config['num_class']).to(device)
    final_model.load_state_dict(torch.load(config['model_path_pruned']))
    test_model(final_model, test_loader, device, classes, config['data'], config['model_path_pruned'])

    # --- Calculate Final Sparsity and FLOPs ---
    print("\n--- Final Model Statistics ---")
    overall_sparsity, total_params = count_nonzero_params(final_model)
    
    with torch.cuda.device(0): # Use GPU 0 for FLOPs calculation
        macs, _ = get_model_complexity_info(
            final_model, # Use the final, loaded model
            (config['batch_size'], config['sequence'], config['features']),
            as_strings=False, 
            print_per_layer_stat=False,
            verbose=False
        )

    # Note: This is an estimated FLOPs. TensorRT's actual sparse acceleration
    # on Orin may lead to higher effective speedup.
    estimated_sparse_flops = 2 * macs * (1 - overall_sparsity) 
    print(f"Total MACs (estimated dense equivalent): {macs:.2e}")
    print(f"Estimated Sparse FLOPs (based on overall sparsity): {estimated_sparse_flops:.2e}")

    wandb.log({"total_parameters_final": total_params})
    wandb.log({"overall_model_sparsity_final": overall_sparsity})
    wandb.log({"estimated_dense_MACs": macs})
    wandb.log({"estimated_sparse_FLOPs": estimated_sparse_flops})

    # --- ONNX Export ---
    print("\n--- Exporting the fine-tuned and pruned model to ONNX ---")
    onnx_path = config['model_path_pruned'].replace(".pth", ".onnx") # Use .onnx extension
    
    # Create a dummy input matching your model's expected input shape
    # Example: (batch_size, sequence, features)
    dummy_input = torch.randn(config['batch_size'], config['sequence'], config['features'], device=device).float()

    # Ensure model is in evaluation mode for export
    final_model.eval()

    try:
        torch.onnx.export(
            final_model,
            dummy_input,
            onnx_path,
            verbose=False,
            opset_version=13, # Recommended opset for good TensorRT compatibility
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # For dynamic batching
        )
        print(f"Fine-tuned and pruned model exported to ONNX at: {onnx_path}")
        wandb.log({"onnx_export_status": "success", "onnx_path": onnx_path})

    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")
        wandb.log({"onnx_export_status": "failed", "onnx_error": str(e)})

    wandb.finish()