import numpy as np
import torch
import yaml
import argparse
import random
import re
import time
import torch.ao.quantization
from torchinfo import summary
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import wandb
from data_processing import preprocess_data
from model import LiteNet, QuantizedLiteNet
from train import train_model, get_time, evaluate_model
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
        feature_file = dataset_config['feature_file']
        
        if len(classes) != num_class:
            print(f"Warning: Number of classes in config ({len(classes)}) does not match num_class ({num_class}) for {dataset_name}.")
            
        return classes, num_class, feature_file
    except KeyError:
        print(f"Error: Configuration for dataset '{dataset_name}' not found in config.yaml.")
        exit()

def training_model_pipeline(config):
    """Orchestrates the model training and evaluation pipeline."""
    seed_everything(134)

    # --- Configuration ---
    dataset_name = config['dataset_name']
    sequence = config['sequence']
    features = config['features'] # Use selected_features for retraining
    num_selected_features = sequence * features
    
    project_name = "LiteNet-" + re.sub(r'[\\/\#\?%:]', '_', str(dataset_name))
    wandb.init(project=project_name, tags=[str(num_selected_features)], config=config, mode="disabled")

    # --- Load Data ---
    data_path = f"dataset/{dataset_name}"
    train = np.load(f"{data_path}/train.npy")
    test = np.load(f"{data_path}/test.npy")
    val = np.load(f"{data_path}/val.npy")
    print('Data loaded')

    # --- Feature Selection ---
    print(f"Loading feature list from: {config['feature_file']}")
    most_important_list = np.load(config['feature_file'])
    
    print(f"Selected {len(most_important_list)} features.")
    print('Preprocessing data...')
    train_loader, test_loader, val_loader, pretime, avgpretime = preprocess_data(
        train, test, val, most_important_list, config['batch_size'], dataset_name
    )
    wandb.log({"preprocess_time": float(pretime), "average_preprocess_time": float(avgpretime)})

    # --- Model Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LiteNet(
        sequence=sequence, 
        features=features, 
        num_class=config['num_class'],
        vocab_size=256,
        embedding_dim=24).to(device)
    
    # Model path is now taken directly from the config
    model_path = config['model_path']
    print(f"Using model path: {model_path}")

    summary(model, input_size=(config['batch_size'], features), device=device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

    # --- Execution ---
    if config['test_mode'] == True:
        print(f"--- Running in Test-Only Mode ---")
        test_model(model, model_path, test_loader, device, config['classes'], config['dataset_name'])
    else:
        print(f"--- Running in Training Mode ---")
        train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, config['epochs'], model_path)
        print("\n--- Testing after training ---")
        test_model(model, model_path, test_loader, device, config['classes'], dataset_name)

def test_model(model, model_path, test_loader, device, classes, dataset_name):
    """Loads a model and evaluates it on the test set."""
    # --- Check for INT8 model and adjust loading ---
    if 'INT8' in model_path:
        print("\n--- INT8 Model Detected: Adjusting for CPU inference ---")
        device = 'cpu'
        
        # The model must be prepared exactly as it was during quantization
        model.to(device)
        model.eval()

        # 1. Fuse modules
        print("Fusing modules to match quantized model structure...")
        layers_to_fuse = [
            ['branch1x1.0', 'branch1x1.1'],
            ['branch3x3.1', 'branch3x3.2'],
            ['branch5x5.1', 'branch5x5.2'],
            ['branch_pool.1', 'branch_pool.2'],
            ['fc2', 'activation6']
        ]
        try:
            torch.ao.quantization.fuse_modules(model, layers_to_fuse, inplace=True)
            print("Fusion complete.")
        except Exception as e:
            print(f"Could not fuse modules during loading: {e}")

        # 2. Wrap the model
        model = QuantizedLiteNet(model)
        model.to(device)
        model.eval()
        
        # 3. Prepare for quantization to finalize the architecture
        model.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
        torch.ao.quantization.prepare(model, inplace=True)
        torch.ao.quantization.convert(model, inplace=True)
        print("Model architecture prepared for INT8 loading.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device) # Ensure model is on the correct device
    
    start_time = time.perf_counter()
    _, all_preds, all_labels = evaluate_model(model, test_loader, device)
    
    time_dif, average_time = get_time(start_time, test=1, data=dataset_name)
    print(f"Testing Time usage: {time_dif:.10f} seconds")
    print(f"Average Testing time: {average_time:.10f} seconds")
    
    acc = metrics.accuracy_score(all_labels, all_preds)
    wandb.log({"accuracy": acc})
    wandb.log({"test_time": float(time_dif)})
    wandb.log({"average_time": float(average_time)})
    
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))
    print(confusion_matrix(all_labels, all_preds))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LiteNet Training and Testing')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset folder in ./dataset/')
    parser.add_argument('--test', type=bool, default=False, help='Enable test-only mode.')
    parser.add_argument('--path', type=str, default=None, help='Path to model. Overrides default path generation. Can be a filename in saved_dict/ or a full path.')
    args = parser.parse_args()

    # --- Load Base Config ---
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # --- Override Config with CLI Args ---
    if args.dataset_name:
        config['dataset_name'] = args.dataset_name
    else:
        config['dataset_name'] = config.get('active_dataset')
    
    if not config.get('dataset_name'):
        print("Error: No dataset specified. Use --dataset_name or set active_dataset in config.yaml.")
        exit()
    
    # --- Determine model path: prioritize --path flag, then fall back to default ---
    if args.path:
        user_path = args.path
        # If it's a full path, use it. If it's just a filename, assume it's in saved_dict/
        if '/' in user_path or '\\' in user_path:
            config['model_path'] = user_path
        else:
            config['model_path'] = f"saved_dict/{user_path}"
    else:
        # Default path if --path is not provided
        config['model_path'] = f"saved_dict/LiteNet_{config['dataset_name']}_embedding.pth"

    # --- Add CLI args to config ---
    config['test_mode'] = args.test

    # --- Dynamically Set Config Values ---
    classes, num_class, feature_file = get_dataset_info(config, config['dataset_name'])
    config['num_class'] = num_class
    config['classes'] = classes
    config['feature_file'] = feature_file
    
    

    training_model_pipeline(config)

