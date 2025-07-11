import numpy as np
import torch
import yaml
import argparse
import random
import re
import time
from torchinfo import summary
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import wandb
from data_processing import preprocess_data
from model import LiteNet
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

def get_class_info(dataset_name):
    """Reads class information from the dataset directory."""
    data_path = f"dataset/{dataset_name}"
    try:
        with open(f"{data_path}/classes.txt", 'r') as f:
            classes = tuple(line.strip() for line in f if line.strip())
        num_class = len(classes)
        return classes, num_class
    except FileNotFoundError:
        print(f"Warning: classes.txt not found for dataset {dataset_name}. Defaulting to 10 classes.")
        return classes, num_class

def training_model_pipeline(config):
    """Orchestrates the model training and evaluation pipeline."""
    seed_everything(134)

    # --- Configuration ---
    dataset_name = config['dataset_name']
    sequence = config['sequence']
    features = config['selected_features'] # Use selected_features for retraining
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
    feature_list_file = f"top_features_{dataset_name}_Original.npy"
    print(f"Loading feature list from: {feature_list_file}")
    most_important_list = np.load(feature_list_file)
    
    print(f"Selected {len(most_important_list)} features.")
    print('Preprocessing data...')
    train_loader, test_loader, val_loader, pretime, avgpretime = preprocess_data(
        train, test, val, most_important_list, config['batch_size'], dataset_name
    )
    wandb.log({"preprocess_time": float(pretime), "average_preprocess_time": float(avgpretime)})

    # --- Model Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LiteNet(sequence=sequence, features=features, num_class=config['num_class']).to(device)
    model_path = f"saved_dict/LiteNet_{dataset_name}.pth"

    summary(model, input_size=(config['batch_size'], sequence, features), device=device)

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
    model.load_state_dict(torch.load(model_path))
    
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
    args = parser.parse_args()

    # --- Load Base Config ---
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # --- Override Config with CLI Args ---
    if args.dataset_name:
        config['dataset_name'] = args.dataset_name
    
    # --- Dynamically Set Config Values ---
    classes, num_class = get_class_info(config['dataset_name'])
    config['num_class'] = num_class
    config['classes'] = classes
    
    # Set test_mode in config for the pipeline
    config['test_mode'] = args.test
    
    training_model_pipeline(config)

