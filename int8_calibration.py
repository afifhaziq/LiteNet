import numpy as np
import os
from data_processing import preprocess_data # Reuse your existing function

# --- Load your data and feature list ---
# (Same data loading logic as in prunesparse.py or train.py)
dataset_name = "ISCXVPN2016"
batch_size = 64
num_features = 20
feature_file = f'top740features{dataset_name}.npy'
most_important_list = np.load(feature_file)
most_important_list = [x - 1 for x in most_important_list][:num_features]
train_data_npy = np.load(f"dataset/{dataset_name}/train.npy", allow_pickle=True)
# We only need dummy data for test/val here as we only need the train_loader
dummy_data = np.array([]) 

# --- Preprocess and get the train_loader ---
train_loader, _, _, _, _ = preprocess_data(
    train_data_npy, dummy_data, dummy_data, most_important_list,
    batch_size, dataset_name
)

# --- Create directory and save batches ---
CALIB_DATA_DIR = "calibration_data"
os.makedirs(CALIB_DATA_DIR, exist_ok=True)

print(f"Saving calibration files to {CALIB_DATA_DIR}...")
# A few hundred batches is usually enough
num_calib_batches = 200 

for i, (batch_data, _) in enumerate(train_loader):
    if i >= num_calib_batches:
        break
    # Save each batch as a flat binary file
    batch_data.numpy().tofile(os.path.join(CALIB_DATA_DIR, f"batch_{i}.bin"))

print(f"Successfully saved {i+1} calibration batches.")