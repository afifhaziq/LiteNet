import torch
from tqdm.auto import tqdm
import time
from datetime import timedelta
import wandb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch import amp


def train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, num_epoch, model_path):
    
    best_val_loss = float('inf')
    wandb.watch(model, log='all')
    
    model.train()
    start_time = time.perf_counter()
    for epoch in range(num_epoch):
        
        train_loss = 0
        for samples, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}"):
            samples, labels = samples.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(samples)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        val_loss, _, _ = evaluate_model(model, val_loader, device, criterion)
        
        scheduler.step(val_loss)

        wandb.log({
            "Train Loss": train_loss, 
            "Val Loss": val_loss,
            "Learning Rate": optimizer.param_groups[0]['lr']
        })
        print(f"Epoch {epoch+1}/{num_epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    end_time = time.perf_counter()
    time_dif = end_time - start_time
    time_dif = time_dif/60
    average_time = time_dif/num_epoch
    print(f"Training Time : {time_dif:.2f} Minutes")  # Show 6 decimal places
    print(f"Average Training time (epoch): {average_time:.2f} Minutes")
    wandb.log({"training_time":  float(time_dif)})
    wandb.log({"average_training_time":  float(average_time)})
    return model

def evaluate_model(model, dataloader, device, criterion=None):
    """
    Evaluates the model on a given dataloader.
    Can compute loss and/or return predictions and labels.
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for samples, labels in dataloader:
            samples = samples.to(device).float()
            labels = labels.to(device)

            predictions = model(samples)

            if criterion:
                loss = criterion(predictions, labels.long())
                total_loss += loss.item()

            preds = torch.argmax(predictions, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader) if criterion and len(dataloader) > 0 else 0
    return avg_loss, all_preds, all_labels

def test_and_report(model, test_loader, device, class_names):
    """
    Evaluates a model on the test set and prints a classification report
    and confusion matrix.
    """
    print("\n--- Starting Final Test ---")
    model.eval()

    all_preds, all_labels = [], []
    with torch.inference_mode():
        for samples, labels in test_loader:
            samples = samples.to(device)
            labels = labels.to(device).long()
            
            # --- TEMPORARY DEBUGGING CHANGE ---
            # If the model is FP16, force the input to be FP16
            if next(model.parameters()).is_cuda and next(model.parameters()).dtype == torch.float16:
                samples = samples.half()
            # ------------------------------------

            # No autocast for this test
            predictions = model(samples)
            
            preds = torch.argmax(predictions, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Final Test Accuracy: {acc*100:.2f}%")
    
    print('--- Classification Report ---')
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    print('--- Confusion Matrix ---')
    print(confusion_matrix(all_labels, all_preds))
    
    return acc

def get_time(start_time, test = 5, data=None):
    end_time = time.perf_counter()
    time_dif = end_time - start_time

    # preprocess time
    if test == 0:
        
        if data == 'MALAYAGT':
            average_time = time_dif / 1046795
        
        else:
            average_time = time_dif / 1090302

        return time_dif, average_time
    
    # Testing Time
    elif test == 1:

        if data == 'MALAYAGT':
            average_time = time_dif / 104679
        
        else:
            average_time = time_dif / 109031

        return time_dif, average_time
    
    # Training time
    elif test == 2:
        
        if data == 'MALAYAGT':
            average_time = time_dif / 837436
        
        else:
            average_time = time_dif / 872241

        return time_dif, average_time
    
    return timedelta(seconds=int(round(time_dif)))