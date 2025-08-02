import numpy as np
import torch
import shap
import yaml
import argparse
import os
import gc
import re
import random
from data_processing import preprocess_data
from model import LiteNetLarge
from train import train_model
import wandb
from torchinfo import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau

def seed_everything(seed: int) -> None:
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset_info(config, dataset_name):
    """Reads dataset-specific information from the config."""
    try:
        dataset_config = config['datasets'][dataset_name]
        classes = tuple(dataset_config['classes'])
        num_class = dataset_config['num_class']
        
        if len(classes) != num_class:
            print(f"Warning: Number of classes in config ({len(classes)}) does not match num_class ({num_class}) for {dataset_name}.")
            
        return classes, num_class
    except KeyError:
        print(f"Error: Configuration for dataset '{dataset_name}' not found in config.yaml.")
        exit()

def calculate_shap_importance(model, model_path, config, device):
    """
    Calculates and saves feature importance using SHAP on a given model.
    """
    print("--- Mode: Calculating SHAP Importance ---")
    
    # --- Load Configuration ---
    dataset_name = config['dataset_name']
    
    # --- Load Model ---
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Load Data for SHAP ---
    print("Loading training data for SHAP analysis...")
    # Use 1000 samples
    # More data is better, but it takes a long time to calculate.
    train_data_npy = np.load(f"dataset/{dataset_name}/train.npy")[:1000]

    # IP Masking (if necessary for the dataset)
    # train_data_npy[:, [12,13,14,15,16,17,18,19]] = 0 

    # The LiteNetLarge model expects a certain number of features.
    num_model_features = config['sequence'] * config['features']
    x_train_full = train_data_npy[:, :-1]
    
    # Ensure data matches model's expected input features
    if x_train_full.shape[1] < num_model_features:
        raise ValueError(f"Input data has {x_train_full.shape[1]} features, but model expects {num_model_features}")
    
    x_train_trimmed = x_train_full[:, :num_model_features]
    x_train = torch.from_numpy(x_train_trimmed.astype(np.float32))

    # --- Run SHAP Analysis ---
    background = x_train.to(device)
    explainer = shap.GradientExplainer(model, background)

    print("Calculating SHAP values... (This may take a while)")
    shap_values = explainer.shap_values(background)

    # shap_values is a list of arrays (one for each class)
    # We take the absolute values and average over all classes and samples
    mean_abs_shap = np.abs(shap_values).mean(axis=0).mean(axis=0)

    # Get the indices of the top 20 features
    most_important_indices = np.argsort(mean_abs_shap)[::-1][:20]
    
    
    output_filename = f"top_features_{dataset_name}.npy"
    np.save(output_filename, most_important_indices)
    
    print(f"Feature importance list saved to: {output_filename}")
    print("Top 20 most important feature indices (sorted):")
    print(np.sort(most_important_indices))

def feature_selection_pipeline(config):
    """Orchestrates the model training and feature selection pipeline."""
    seed_everything(134)

    # --- Configuration ---
    dataset_name = config['dataset_name']
    sequence = config['sequence']
    features = config['features']
    num_total_features = sequence * features
    
    project_name = "LiteNet-FeatureSelection-" + re.sub(r'[\\/\#\?%:]', '_', str(dataset_name))
    wandb.init(project=project_name, tags=[str(num_total_features)], config=config, mode="disabled")

    # --- Load Data ---
    data_path = f"dataset/{dataset_name}"
    train_data = np.load(f"{data_path}/train.npy")
    test_data = np.load(f"{data_path}/test.npy")
    val_data = np.load(f"{data_path}/val.npy")
    print('Data loaded')

    print('Preprocessing data for full-feature model...')
    # Pass empty list for features to use all available features
    train_loader, _, val_loader, _, _ = preprocess_data(
        train_data, test_data, val_data, [], config['batch_size'], dataset_name
    )

    # --- Model Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LiteNetLarge(
        sequence=sequence, 
        features=features, 
        num_class=config['num_class'],
        vocab_size=256, 
        embedding_dim=24).to(device)
    
    model_path = config['model_path']
    print(f"Using model path: {model_path}")

    summary(model, input_size=(config['batch_size'], sequence*features), device=device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

    # --- Execution ---
    if config['mode'] == 'tr':
        print("--- Running in Training & Feature Selection Mode ---")
        train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, config['epochs'], model_path)
        print("Training complete. Starting SHAP analysis.")
        calculate_shap_importance(model, model_path, config, device)
    
    elif config['mode'] == 'fs':
        print("--- Running in Feature Selection Only Mode ---")
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}. Please run with '--mode tr' first to train the model.")
            return
        print(f"Loading existing model from: {model_path}")
        calculate_shap_importance(model, model_path, config, device)
        
    print("Clearing data from memory...")
    del train_data, test_data, val_data, train_loader, val_loader
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Selection using SHAP")
    parser.add_argument('--data', type=str, required=True, help='Name of the dataset folder in ./dataset/')
    parser.add_argument('--mode', type=str, required=True, choices=['tr', 'fs'], help="Operation mode (tr: train and select, fs: feature select only)")
    parser.add_argument('--path', type=str, default=None, help='Path to model. Overrides default path generation.')
    args = parser.parse_args()

    # --- Load Base Config ---
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # --- Override Config with CLI Args ---
    config['dataset_name'] = args.data
    config['mode'] = args.mode

    # --- Determine model path ---
    if args.path:
        # If it's a full path, use it. If it's just a filename, assume it's in saved_dict/
        if '/' in args.path or '\\' in args.path:
            config['model_path'] = args.path
        else:
            config['model_path'] = f"saved_dict/{args.path}"
    else:
        # Default path for the large model used for feature selection
        config['model_path'] = f"saved_dict/LiteNet_{config['dataset_name']}_large.pth"

    # --- Dynamically Set Config Values ---
    classes, num_class = get_dataset_info(config, config['dataset_name'])
    config['num_class'] = num_class
    config['classes'] = classes
    
    # For feature selection, we use the 'large' model parameters from the top level of the config
    if 'large_model' in config:
        large_model_config = config['large_model']
        config['sequence'] = large_model_config['sequence']
        config['features'] = large_model_config['features']
    else:
        print(f"Warning: 'large_model' configuration not found in config.yaml. Using default values.")
        # Default values
        config['sequence'] = 37 
        config['features'] = 20  

    feature_selection_pipeline(config) 