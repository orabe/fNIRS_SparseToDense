from torch.utils.data import DataLoader
import pickle
from datasets_v02 import fNIRSChannelSpaceSegmentLoad, fNIRSPreloadDataset
import torch
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from model import CNN2DImage, CNN2DChannelV2, CNN2D_BaselineV2
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from utils import create_train_test_files, create_train_test_segments, create_train_test_segments_grad
warnings.filterwarnings("ignore")
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)



print("hello hydra")
print(f"job ID: {os.getenv('SLURM_JOB_ID')}")
print(f"array job ID: {os.getenv('SLURM_ARRAY_JOB_ID')}")
print(f"array task ID: {os.getenv('SLURM_ARRAY_TASK_ID')}")
print(f"CUDA available: {torch.cuda.is_available()}")


# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)

        # Forward pass
        outputs = model(data)   
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    f1_avg = []
    acc_avg = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            f1_avg.append(f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='micro'))
            acc_avg.append((predicted == labels).sum().item() / labels.size(0))

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='micro')  # or 'macro' if you prefer
    return total_loss / len(test_loader), accuracy, f1, np.mean(f1_avg), np.mean(acc_avg)

# Main function
if __name__ == "__main__":
    # Hyperparameters
    num_epochs = 400
    learning_rate = 1e-4
    batch_size = 16
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    # base_dir = "/home"
    # dataset_path = os.path.join(base_dir, "data/BallSqueezingHD_modified")
    # preprocessed_path = os.path.join(base_dir, "data/yuanyuan_v2_processed_partial/")
    # DATASET_NAME = "BallSqueezingHD_modified"
    # DATASET_NAME = "parcel_BallSqueezingHD_modified"
    # preprocessed_path = os.path.join("datasets/processed", DATASET_NAME)
    
    # DATASET_NAME = "fullParcel_FreshMotor"
    DATASET_NAME = "fullParcel_BallSqueezingHD_modified"
    preprocessed_path = os.path.join("datasets/full_processed", 'BallSqueezingHD_modified')
        
    os.makedirs(f"results/{DATASET_NAME}/checkpoints/", exist_ok=True)
        
    # subject_ids = ['sub-170', 'sub-173', 'sub-176', 'sub-179',
    #             'sub-182', 'sub-577', 'sub-581', 'sub-586',  
    #             'sub-613', 'sub-619', 'sub-633', 'sub-177',  
    #             'sub-181', 'sub-183', 'sub-185', 'sub-568', 
    #             'sub-580', 'sub-583', 'sub-587', 'sub-592',  
    #             'sub-618', 'sub-621', 'sub-638', 'sub-640']
    
    if DATASET_NAME == "fullParcel_BallSqueezingHD_modified":
        subject_ids = ['sub-170', 'sub-173', 'sub-171', 'sub-174',
                       'sub-176', 'sub-179', 'sub-182', 'sub-177',
                       'sub-181', 'sub-183', 'sub-184', 'sub-185']
    elif DATASET_NAME == "fullParcel_FreshMotor":
        subject_ids = ['sub-01', 'sub-02', 'sub-03', 'sub-04',
                       'sub-05', 'sub-06', 'sub-07', 'sub-08',
                       'sub-09', 'sub-10']
        
    k = len(subject_ids)  # = 10 FreshMotor, 12 BSQ-HD (LOSO)

    # Parameters
    random_state = 42  # For reproducibility

    # Shuffle the subject list
    rng = np.random.default_rng(seed=random_state)
    shuffled_subjects = rng.permutation(subject_ids)

    # Split into k roughly equal folds
    folds = np.array_split(shuffled_subjects, k)

    # Optional: convert each fold to a list
    folds = [list(fold) for fold in folds]

 
    # exclude_subjects = ['sub-547', 'sub-639', 'sub-588', 'sub-171', 'sub-174', 'sub-184']
    exclude_subjects = ['sub-547', 'sub-639', 'sub-588']
    chromo = "yuanyuan_v2"
    for fold in folds:
        subs = "_".join(fold)

        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        train_csv_path, test_csv_path = create_train_test_segments(
            None,
            preprocessed_path,
            test_subjects_list=fold,
            exclude_subjects=exclude_subjects
        )
        train_csv = pd.read_csv(train_csv_path)
        test_csv = pd.read_csv(test_csv_path)

        train_dataset = fNIRSPreloadDataset(
            train_csv_path, chromo='HbO')
        test_dataset = fNIRSPreloadDataset(
            test_csv_path, mode="test", chromo='HbO')
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # Initialize model, loss, and optimizer
        model = CNN2DImage().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        test_f1_avgs = []
        test_f1s = []
        train_f1_avgs = []
        train_f1s = []

        # Training loop
        for epoch in range(num_epochs):

            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            _, train_accuracy, train_f1, train_f1_avg, train_acc_avg = evaluate_model(model, train_loader, criterion, device)
            test_loss, test_accuracy, test_f1, test_f1_avg, test_acc_avg = evaluate_model(model, test_loader, criterion, device)


            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            test_f1_avgs.append(test_f1_avg)
            test_f1s.append(test_f1)
            train_f1_avgs.append(train_f1_avg)
            train_f1s.append(train_f1)

            logging.info(f"Sub: {subs}, Epoch [{epoch+1}], Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")

            # print(f"Sub: {subs}, Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            #     f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}, "
            #     f"Test F1 Avg: {test_f1_avg:.4f}")
       
        res = {"train_loss": train_losses, "train_accuracy": train_accuracies,
                "test_loss": test_losses, "test_accuracy": test_accuracies, "test_f1": test_f1s,
                "test_f1_avg": test_f1_avgs, "test_acc_avg": test_acc_avg,
                "train_f1": train_f1s, "train_f1_avg": train_f1_avgs, "train_acc_avg": train_acc_avg}
        
        with open(f"results/{DATASET_NAME}/res_{subs}_{chromo}.pkl", "wb") as f:
            pickle.dump(res, f)

        torch.save(model.state_dict(), f"results/{DATASET_NAME}/checkpoints/model_{subs}_{chromo}.pth")
        
        print("Model saved successfully!")
    
    print("\n-----Training complete! -----\n")
