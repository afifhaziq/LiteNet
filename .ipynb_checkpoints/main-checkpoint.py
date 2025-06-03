import numpy as np
import torch
from data_processing import preprocess_data
from model import NtCNN
from train import train_model, get_time
from sklearn.metrics import classification_report, confusion_matrix
import time
import random
from torchinfo import summary
import wandb
import argparse
from sklearn import metrics

# Set seed
def seed_everything(seed: int) -> None:
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(134)

parser = argparse.ArgumentParser(description='Inception')
parser.add_argument('--data', type=str, required=True, help='input dataset source')
parser.add_argument('--test', type=bool, default=False, help='Train or test')
parser.add_argument('-s','--sequence', type=int, default=0, help='sequence')
parser.add_argument('-f','--features', type=int, default=0, help='features')
#parser.add_argument('--numfeatures', type=int, default=0, help='Number of features')

args = parser.parse_args()


# Hyperparameters
sequence = args.sequence
features = args.features
learning_rate = 0.001
batch_size = 64
num_class = 10
num_epoch = 30
numfeatures = sequence * features

"""Trains the model and evaluates it on validation data."""
wandb.init(project="Inception-"+ args.data, tags= [str(numfeatures)], )
wandb.config = {
"learning_rate": learning_rate,
"epochs": num_epoch,
"batch_size": 64,
"sequence":  sequence,
"features":  features
}




# Load data
train = np.load(args.data + "/train.npy", allow_pickle=True)
test = np.load(args.data + "/test.npy", allow_pickle=True)
val = np.load(args.data + "/val.npy", allow_pickle=True)
print('Data loaded')


if args.data == 'ISCXVPN2016':
    classes = ('AIM Chat','Email','Facebook Audio','Facebook Chat','Gmail Chat','Hangouts Chat','ICQ Chat','Netflix','Spotify','Youtube')
    most_important_list = np.load('top740featuresISCX.npy')
else:
    classes = ('Bittorent', 'ChromeRDP', 'Discord', 'EAOrigin', 'MicrosoftTeams', 'Slack', 'Steam', 'Teamviewer', 'Webex', 'Zoom')
    most_important_list = np.load('top740featuresMALAYAGT.npy')

most_important_list = most_important_list[:numfeatures]


# Feature selection
# most_important_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
#      24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
#      45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 62, 65, 74, 75, 85] # Adjusted feature indices

#most_important_list = [740]

most_important_list = [x - 1 for x in most_important_list]

if str(len(most_important_list)) == '1':
    num_features = str(most_important_list[0])
else:
    num_features = str(len(most_important_list))

print(len(most_important_list))
#train, test, val = preprocess_data(train, test, val, most_important_list)
print('Preprocessing data...')
train_loader, test_loader, val_loader, pretime, avgpretime = preprocess_data(train, test, val, most_important_list, batch_size, args.data)
wandb.log({"preprocess_time":  float(pretime)})
wandb.log({"average_preprocess_time":  float(avgpretime)})

#train_loader, test_loader, val_loader = prepare_dataloader(train, test, val, batch_size)

# train_samples = len(train_loader) * train_loader.batch_size
# print(f"Estimated Total train_loader Samples: {train_samples}")

# test_samples = len(test_loader) * test_loader.batch_size
# print(f"Estimated Total test_loader Samples: {test_samples}")

# Initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = NtCNN(sequence=sequence, features=features, num_class=num_class).to(device)


summary(model, 
        input_size=[batch_size, sequence, features], 
        device=device, 
        col_names=["input_size","output_size", "num_params"])

# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



def test_model(model, test_loader, device, classes, num_features, args):
    # Load best model
    model.load_state_dict(torch.load('saved_dict/' + type(model).__name__  +'_'+ args.data +'_'+ num_features + "Features_best_model.pth"))
    

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
        #end_time = time.perf_counter()

    time_dif, average_time = get_time(start_time, test=1, data=args.data)
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

if args.test == True:
    test_model(model, test_loader, device, classes, num_features, args)
    
else:
    # Train model
    train_model(model, train_loader, val_loader, device, criterion, optimizer, num_epoch, num_features, args.data, learning_rate)
    test_model(model, test_loader, device, classes, num_features, args)

