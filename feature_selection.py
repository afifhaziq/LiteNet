import numpy as np
import torch
import shap
import yaml
import argparse
import os
import gc
from data_processing import preprocess_data
from model import LiteNet
from train import train_model
import wandb
import re

def train_full_model(config):
    """
    Trains a model on all available features.
    """
    print("--- Mode: Training Full Model ---")
    
    # --- Load Configuration ---
    dataset_name = config['dataset_name']
    batch_size = config['batch_size']
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    num_features = config['num_features']
    sequence = 1 # Use sequence=1 for full feature training
    project_name = "Inception-" + re.sub(r'[\\/\#\?%:]', '_', str(dataset_name))
    wandb.init(project=project_name, tags=[str(num_features)], mode="disabled")
    # --- Initialize Model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LiteNet(sequence=37, features=20, num_class=config.get('num_class', 10)).to(device)
    model_path = f"saved_dict/LiteNet_{dataset_name}_large.pth"

    # --- Load Data and Train ---
    print("Loading dataset for training...")
    train_data = np.load(f"dataset/{dataset_name}/train.npy", allow_pickle=True)
    test_data = np.load(f"dataset/{dataset_name}/test.npy", allow_pickle=True)
    val_data = np.load(f"dataset/{dataset_name}/val.npy", allow_pickle=True)
    

    
    # num_model_features = 36 * 20
    # train_data = np.concatenate((train_data[:, :num_model_features], train_data[:, -1:]), axis=1)
    # test_data = np.concatenate((test_data[:, :num_model_features], test_data[:, -1:]), axis=1)
    # val_data = np.concatenate((val_data[:, :num_model_features], val_data[:, -1:]), axis=1)

    train_loader, _, val_loader, _, _ = preprocess_data(
        train_data, test_data, val_data, [], batch_size, dataset_name
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_model(model, train_loader, val_loader, device, criterion, optimizer, epochs, dataset_name, True)
    print("Training complete. Model saved.")
    
    print("Clearing training data from memory...")
    del train_data, test_data, val_data, train_loader, val_loader
    gc.collect()
    
    return model, model_path

def calculate_shap_importance(model, model_path, config):
    """
    Calculates and saves feature importance using SHAP on a given model.
    """
    print("--- Mode: Calculating SHAP Importance ---")
    
    # --- Load Configuration ---
    dataset_name = config['dataset_name']
    num_features = config['num_features']
    sequence = 1
    
    # --- Load Model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # --- Load Data for SHAP ---
    print("Loading training data for SHAP analysis...")
    train_data_npy = np.load(f"dataset/{dataset_name}/train.npy", allow_pickle=True)
    train_data_npy = train_data_npy[:1000]

    train_data_npy[:, [12,13,14,15,16,17,18,19]] = 0 # IP Masking
    
    # Trim features to match model dimensions (36 * 20 = 720 features)
    num_model_features = 36 * 20
    x_train_full = train_data_npy[:, :-1]
    x_train_trimmed = x_train_full[:, :num_model_features]
    x_train = torch.from_numpy(x_train_trimmed.astype(np.float32))

    # --- Run SHAP Analysis ---
    background = x_train.to(device)
    explainer = shap.GradientExplainer(model, background)

    print("Calculating SHAP values... (This may take a while)")
    shap_values = explainer.shap_values(background)

    # Calculate the mean of the absolute values of the SHAP values across all classes and background samples output size will be 740 (features size)
    shap_values = np.abs(shap_values).mean(axis=2).mean(axis=0)

    # Sort the indices to get the top 20 most important features in ascending order of index value (0-739)
    most_important_indices = np.sort(np.argsort(shap_values)[::-1][:20])
    
    output_filename = f"top_features_{dataset_name}.npy"
    np.save(output_filename, most_important_indices)
    
    print(f"Feature importance list saved to: {output_filename}")
    print("Top 20 most important feature indices:")
    print(most_important_indices)

def main():
    """Main function to drive the script."""
    parser = argparse.ArgumentParser(description="Feature Selection using SHAP")
    parser.add_argument('--mode', type=str, required=True, choices=['tr', 'fs'], help="Operation mode (tr: train and select, fs: feature select only)")
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset folder name (e.g., ISCXVPN2016)')
    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['dataset_name'] = args.dataset_name

    

    if args.mode == 'tr':
        model, model_path = train_full_model(config)
        calculate_shap_importance(model, model_path, config)
    
    elif args.mode == 'fs':

        model_path = f"saved_dict/LiteNet_{config['dataset_name']}_large.pth" 
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}. Please run with '--mode tr' first.")
            return
        
        print(f"Using existing model from: {model_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = LiteNet(sequence=36, features=20, num_class=config.get('num_class', 10)).to(device)
        calculate_shap_importance(model, model_path, config)

if __name__ == "__main__":
    main() 