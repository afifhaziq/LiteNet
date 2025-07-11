import torch
from tqdm.auto import tqdm
import time
from datetime import timedelta
import wandb


def train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, num_epoch, model_path):
    

    best_val_loss = float('inf')
    wandb.watch(model, log='all')
    model.train()
    start_time = time.perf_counter()
    for epoch in range(num_epoch):
        
        train_loss = 0
        #start_time = time.time()

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
        
        # Step the scheduler with the validation loss
        scheduler.step(val_loss)

        # Log metrics to wandb
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