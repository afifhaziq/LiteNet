import numpy as np
import time
from train import get_time
from torch.utils.data import DataLoader, TensorDataset
import torch


def preprocess_data(train, test, val, most_important_list, batch_size, data):
    """Select features"""
    start_time = time.perf_counter()
    # IP Masking
    #print(train.shape)
    train[:, [12,13,14,15,16,17,18,19]] = 0
    test[:, [12,13,14,15,16,17,18,19]] = 0
    val[:, [12,13,14,15,16,17,18,19]] = 0

    # train = np.delete(train, [12,13,14,15,16,17,18,19], 1)
    # test = np.delete(test, [12,13,14,15,16,17,18,19], 1)
    # val = np.delete(val, [12,13,14,15,16,17,18,19], 1)

    # print(train.shape)
    # print(test.shape)
    # print(val.shape)
    if len(most_important_list) > 1:
        train_reduced = train[:, most_important_list]
        test_reduced = test[:, most_important_list]
        val_reduced = val[:, most_important_list]
        print('Important features based on SHAP')
    
    else:
        limit = most_important_list[0] + 1
        train_reduced = train[:, :limit]
        test_reduced = test[:, :limit]
        val_reduced = val[:, :limit]
        print(f'top {limit} features')
        
        
    train = np.column_stack((train_reduced, train[:, -1]))
    test = np.column_stack((test_reduced, test[:, -1]))
    val = np.column_stack((val_reduced, val[:, -1]))

    #print('train.shape before loader:', train.shape)
    # print(test.shape)
    # print(val.shape)
    
    train_loader, test_loader, val_loader = prepare_dataloader(train, test, val, batch_size)
    #print('train.shape after loader:', train.shape)
    pretime, avgpretime = get_time(start_time, test=0, data=data)
    print('Preprocess Time: ', pretime, 'Average Preprocess Time: ', avgpretime)
    return train_loader, test_loader, val_loader, pretime, avgpretime

def prepare_dataloader(train, test, val, batch_size):
    """Converts data to PyTorch tensors and prepares data loaders."""



    # Split into features (X) and labels (y)
    x_train, y_train = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]
    x_val, y_val = val[:, :-1], val[:, -1]

    #print(val[:, -1])

    # Dtype conversion
    x_train = torch.from_numpy(x_train.astype(np.float32))
    x_test = torch.from_numpy(x_test.astype(np.float32))
    x_val = torch.from_numpy(x_val.astype(np.float32))

    y_train = torch.from_numpy(y_train.astype(np.int64))
    y_test = torch.from_numpy(y_test.astype(np.int64))
    y_val = torch.from_numpy(y_val.astype(np.int64))
    
    #print(y_train.min(), y_train.max())



    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    val_dataset = TensorDataset(x_val, y_val)

    #print(x_train.size)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    
    return train_loader, test_loader, val_loader



