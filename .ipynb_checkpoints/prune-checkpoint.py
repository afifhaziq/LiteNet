import torch
import torch.nn.utils.prune as prune
from model import NtCNN
import numpy as np
from data_processing import preprocess_data
import time
from train import get_time
import wandb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from torchinfo import summary
import random
import argparse
import copy
from sklearn import metrics
from ptflops import get_model_complexity_info
import gc

def seed_everything(seed: int) -> None:
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(134)

parser = argparse.ArgumentParser(description='Inception Prune')
parser.add_argument('--data', type=str, required='ISCXVPN2016', help='input dataset source')
parser.add_argument('--amount', type=float, default=0.0, help='prune amount')
#parser.add_argument('--numfeatures', type=int, default=0, help='Number of features')

args = parser.parse_args()

# First calculate the derived values
sequence = 1
features = 20
data = args.data  # 'ISCXVPN2016' or 'MALAYAGT'
amount = args.amount
num_features = sequence * features

config = {
    'sequence': 1,
    'features': 20,
    'learning_rate': 0.001,
    'batch_size': 64,
    'num_class': 10,
    'data': data,
    'num_features': 20,
    'model_path': f"saved_dict/NtCNN_{data}_{num_features}Features_best_model.pth",
    'model_path_pruned': f"saved_dict/NtCNN_{data}_{num_features}Features_best_model_pruned.pth",
    'output_path': 'global_relevance.pth'
}

"""Trains the model and evaluates it on validation data."""
wandb.init(project="Inception-"+ data + "_prune", mode="online", tags= [str(amount)], group= 'Linear1')
wandb.config = {
"learning_rate": config['learning_rate'],
"epochs": 30,
"batch_size": config['batch_size'],
"prune_type": amount,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

def prune_model(model):
    importance_data = {}
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            # Conv1d pruning (structured)
            weights = module.weight.detach()
            importance = torch.norm(weights, p=1, dim=(0, 2))
            amount = 0.0
            # # Determine pruning amount
            '''if 'branch1x1.0' in name:
                amount = 0.7 #best
            elif 'branch3x3.1' in name:
                amount = 0.3 #best
            elif 'branch5x5.1' in name:
                amount = 0.2 #best
            elif 'branch_pool.1' in name:
                amount = 0.2 #best
            else:
                amount = 0.0'''
            # Store data
            importance_data[name] = {
                'scores': importance.cpu().numpy(),
                'weights': weights.cpu().numpy(),  # Store weights for visualization
                'amount': amount,
                'type': 'Conv1d'
            }
            
            prune.ln_structured(module, 'weight', amount=amount, n=1, dim=0)
            prune.remove(module, 'weight')
            
        elif isinstance(module, torch.nn.Linear):
            # Linear pruning (unstructured)
            weights = module.weight.detach()
            importance = weights.abs()
            
            # Set layer-specific amounts
            if 'fc1' in name:
                amount = 0.81  # .81 for above 90% acc
            #if 'fc2' in name:
                #amount = 0.66 # .66 best
            else:
                amount = 0.0
            #if 'fc3' in name:  # fc3
                #amount = 0.13

            # Store data
            importance_data[name] = {
                'scores': importance.cpu().numpy().flatten(),
                'weights': weights.cpu().numpy(),  
                'amount': amount,
                'type': 'Linear'
            }
            
            # Apply actual pruning
            prune.l1_unstructured(module, 'weight', amount=amount)
            prune.remove(module, 'weight')
    
    return model

'''def global_prune(model, amount):
    """Apply global unstructured pruning to the model."""
    parameters_to_prune = [
        (module, 'weight') 
        for module in model.modules() 
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear))
    ]
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount)
    

    for module, _ in parameters_to_prune:
        if hasattr(module, 'weight_orig'):
            prune.remove(module, 'weight')
            # Direct buffer removal (safer than delattr)
            module._buffers.pop('weight_orig', None)
            module._buffers.pop('weight_mask', None)
    
    return model'''

def test_model(model, test_loader, device, classes, data, pruned=False):
    # Load best model

    if pruned:
        state_dict = torch.load(config['model_path_pruned'])
        # Remove pruning buffers before loading
        '''for key in list(state_dict.keys()):
            if '_orig' in key or '_mask' in key:
                del state_dict[key]'''
        model.load_state_dict(state_dict, strict=False)
    else:   
        model.load_state_dict(torch.load(config['model_path']))
    model.eval()

    # Evaluate model on test set
    all_preds, all_labels = [], []

    model.eval()  # Set model to evaluation mode


    # Disable gradient computation for testing
    with torch.inference_mode():
        start_time = time.perf_counter()    
        for images, labels in test_loader:
            images = images.to(device).float()  # Move images to the appropriate device
            labels = labels.to(device).float()  # Move labels to the appropriate device

            predictions = model(images)  # Get predictions from the model

            # Convert model output (predictions) to class indices
            preds = torch.argmax(predictions, dim=1)
            
            # Convert one-hot encoded labels to class indices
            #labels = torch.argmax(labels, dim=1) # add this line for one hot encoded labels
            
            # Store predictions and true labels
            all_preds.extend(preds.cpu().numpy())  # Move to CPU and convert to numpy
            all_labels.extend(labels.cpu().numpy())  # Move to CPU and convert to numpy
        

    time_dif, average_time = get_time(start_time, test=1, data=data)
    print(f"Testing Time usage: {time_dif:.10f} seconds")  
    print(f"Average Testing time: {average_time:.10f} seconds")
    acc = metrics.accuracy_score(all_labels, all_preds)
    wandb.log({"accuracy":  (acc)})
    wandb.log({"test_time":  float(time_dif)})
    wandb.log({"average_time":  float(average_time)})

    #print('Inference Time:' , get_time(start_time, test=1, data=args.data))
    # Generate and print the confusion matrix and classification report
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))
    print(confusion_matrix(all_labels, all_preds))

def count_nonzero_params(model):
    """Count remaining active parameters"""
    total = 0
    nonzero = 0
    for p in model.parameters():
        if p is not None:
            total += p.numel()
            nonzero += torch.sum(p != 0).item()
    print(f"Non-zero params: {nonzero}/{total} ({nonzero/total:.1%})")
    print(f"Sparsity: {total-nonzero}/{total} ({(total-nonzero)/total:.1%})")
    return (total-nonzero)/total, nonzero

        
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
# Load data
try:
    train = np.load(f"{config['data']}//train.npy", allow_pickle=True)
    test = np.load(f"{config['data']}//test.npy", allow_pickle=True)
    val = np.load(f"{config['data']}//val.npy", allow_pickle=True)
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    

# Preprocess data
_, test_loader, _, pretime, avgpretime = preprocess_data(train, test, val, most_important_list,
                                        config['batch_size'], config['data'])

wandb.log({"preprocess_time":  float(pretime)})
wandb.log({"average_preprocess_time":  float(avgpretime)})

del train, test, val, most_important_list
gc.collect()

# Initialize model
model = NtCNN(sequence=config['sequence'], 
                features=config['features'], 
                num_class=config['num_class']).to(device)


#count_nonzero_params(model)
#test_model(model, test_loader, device, classes, config['data'] )

model.load_state_dict(torch.load(config['model_path']))
#pruned_model = copy.deepcopy(model)
print("Pruning model...")

# Local pruning
pruned_model = prune_model(model)

# Global pruning
#pruned_model = global_prune(model, amount)

print("Model pruned.")

#save model
torch.save(
    {k: v for k, v in pruned_model.state_dict().items() 
     if not ('_orig' in k or '_mask' in k)},
    config['model_path_pruned']
)


test_model(pruned_model, test_loader, device, classes, config['data'], pruned=True )

sparsity, params = count_nonzero_params(pruned_model)


with torch.cuda.device(0):
    macs, _ = get_model_complexity_info(
        pruned_model,
        (config['batch_size'], config['sequence'], config['features']),
        as_strings=False, 
        print_per_layer_stat=False,
        verbose=False
    )


print(f"MACs: {macs:.2e}")
sparse_flops = 2 * macs * (1 - sparsity)
print(f"Sparse FLOPs: {sparse_flops:.2e}")

wandb.log({"amount":  float(amount)})
wandb.log({"Sparse_FLOPs":  sparse_flops})
wandb.log({"parameters":  params})

