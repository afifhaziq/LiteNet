import numpy as np
import time
from train import get_time
from torch.utils.data import DataLoader, TensorDataset
import torch


def preprocess_data(train, test, val, most_important_list, batch_size, data):
    """Prepares data by selecting features and creating DataLoaders."""
    start_time = time.perf_counter()
    
    # IP Masking
    train[:, [12,13,14,15,16,17,18,19]] = 0
    test[:, [12,13,14,15,16,17,18,19]] = 0
    val[:, [12,13,14,15,16,17,18,19]] = 0

    # Split into features (X) and labels (y)
    x_train, y_train = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]
    x_val, y_val = val[:, :-1], val[:, -1]

    # Perform feature selection if a list is provided
    if most_important_list.size > 0:
        print(f'Selecting {most_important_list.shape[0]} important features based on list...')
        x_train = x_train[:, most_important_list]
        x_test = x_test[:, most_important_list]
        x_val = x_val[:, most_important_list]
    else:
        print('Using all features for training...')

    train_loader, test_loader, val_loader = prepare_dataloader(
        x_train, y_train, x_test, y_test, x_val, y_val, batch_size
    )

    pretime, avgpretime = get_time(start_time, test=0, data=data)
    print('Preprocess Time: ', pretime, 'Average Preprocess Time: ', avgpretime)
    return train_loader, test_loader, val_loader, pretime, avgpretime

def prepare_dataloader(x_train, y_train, x_test, y_test, x_val, y_val, batch_size):
    """Converts data to PyTorch tensors and prepares data loaders."""

    # Dtype conversion AND Normalization
    # 1. Convert the uint8 data to float32
    # 2. Normalize the values from [0, 255] to [0, 1]
    x_train = torch.from_numpy(x_train.astype(np.float32))
    x_test = torch.from_numpy(x_test.astype(np.float32))
    x_val = torch.from_numpy(x_val.astype(np.float32))

    # Labels remain as int64
    y_train = torch.from_numpy(y_train.astype(np.int64))
    y_test = torch.from_numpy(y_test.astype(np.int64))
    y_val = torch.from_numpy(y_val.astype(np.int64))

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    
    return train_loader, test_loader, val_loader



