import argparse
import yaml
import os
import sys
from main import run_inception_pipeline
from train import train_model
import torch
import wandb
from model import LiteNet
from data_processing import preprocess_data
import numpy as np
# from prunesparse import prune_and_export_onnx  # To be implemented
# from tensorrtinference import run_tensorrt_inference  # To be implemented


def train_full(config):
    print("[Stage: Train Full] Training with all features and computing feature importance...")
    # Set up config values
    dataset_name = config.get('dataset_name')
    sequence = config.get('sequence', 1)
    features = config.get('features', 0) or config.get('num_features', 20)
    learning_rate = config.get('learning_rate', 0.001)
    batch_size = config.get('batch_size', 64)
    num_class = config.get('num_class', 10)
    num_epoch = config.get('epochs', 20)

    data_path = f"dataset/{dataset_name}"
    project_name = "Inception-" + dataset_name
    wandb.init(project=project_name, tags=[str(sequence * features)])
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": num_epoch,
        "batch_size": batch_size,
        "sequence": sequence,
        "features": features
    }

    # Load data
    train = np.load(data_path + "/train.npy", allow_pickle=True)
    test = np.load(data_path + "/test.npy", allow_pickle=True)
    val = np.load(data_path + "/val.npy", allow_pickle=True)
    print('Data loaded')

    # Class and feature selection
    # Load class names from classes.txt in the dataset folder
    classes_file = f"{data_path}/classes.txt"
    with open(classes_file, 'r') as f:
        classes = tuple(line.strip() for line in f if line.strip())


    # Use top_features_file 
    most_important_list = np.load("top740features"+ dataset_name + ".npy")

    most_important_list = [x - 1 for x in most_important_list]
    # Use all features
    num_features = len(most_important_list)
    most_important_list = most_important_list[:num_features]

    # Preprocess data
    train_loader, test_loader, val_loader, pretime, avgpretime = preprocess_data(
        train, test, val, most_important_list, batch_size, dataset_name)
    wandb.log({"preprocess_time":  float(pretime)})
    wandb.log({"average_preprocess_time":  float(avgpretime)})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LiteNet(sequence=sequence, features=num_features, num_class=num_class).to(device)

    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    train_model(model, train_loader, val_loader, device, criterion, optimizer, num_epoch, str(num_features), dataset_name, learning_rate)
    print("[Stage: Train Full] Training complete. Model saved.")
    # TODO: Feature importance extraction and saving


def train_reduced(config):
    print("[Stage: Train Reduced] Training with top-N features and testing...")
    # TODO: Retrain with top-N features from feature importance
    # TODO: Test the model (PyTorch test)
    print("[Stub] Retraining and testing not yet implemented.")


def deploy(config):
    print("[Stage: Deploy] Pruning, exporting to ONNX, quantizing, converting to TRT, and testing with TensorRT...")
    # TODO: Prune the model
    # TODO: Export to ONNX
    # TODO: Quantize and convert to TRT
    # TODO: Test with tensorrtinference.py
    print("[Stub] Deploy (prune, export, quantize, TRT test) not yet implemented.")


def main():
    parser = argparse.ArgumentParser(description="Unified ML Pipeline")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    train_full_parser = subparsers.add_parser("train_full")
    train_full_parser.add_argument("--config", required=True, help="Path to config.yaml")

    train_reduced_parser = subparsers.add_parser("train_reduced")
    train_reduced_parser.add_argument("--config", required=True, help="Path to config.yaml")

    deploy_parser = subparsers.add_parser("deploy")
    deploy_parser.add_argument("--config", required=True, help="Path to config.yaml")

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--config", required=True, help="Path to config.yaml")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(config.get('artifacts_dir', 'artifacts'), exist_ok=True)

    if args.stage == "train_full":
        train_full(config)
    elif args.stage == "train_reduced":
        train_reduced(config)
    elif args.stage == "deploy":
        deploy(config)
    elif args.stage == "all":
        train_full(config)
        train_reduced(config)
        deploy(config)
    else:
        print(f"Unknown stage: {args.stage}")
        sys.exit(1)

if __name__ == "__main__":
    main() 