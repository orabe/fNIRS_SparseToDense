from torch.utils.data import DataLoader, ConcatDataset
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
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, precision_score, recall_score, confusion_matrix
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


# Function to calculate class weights from dataset
def calculate_class_weights(dataset, device):
    """Calculate class weights for imbalanced datasets"""
    all_labels = []
    if isinstance(dataset, ConcatDataset):
        for sub_dataset in dataset.datasets:
            for _, label in sub_dataset:
                all_labels.append(label)
    else:
        for _, label in dataset:
            all_labels.append(label)
    
    all_labels = np.array(all_labels)
    unique_classes = np.unique(all_labels)
    n_samples = len(all_labels)
    n_classes = len(unique_classes)
    
    # Calculate weights: n_samples / (n_classes * n_samples_per_class)
    class_weights = n_samples / (n_classes * np.bincount(all_labels))
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    logging.info(f"Class distribution: {dict(zip(unique_classes, np.bincount(all_labels)))}")
    logging.info(f"Class weights: {dict(zip(unique_classes, class_weights))}")
    
    return class_weights_tensor

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
    all_probs = []  # For ROC curve
    acc_avg = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get probabilities for ROC curve
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            acc_avg.append((predicted == labels).sum().item() / labels.size(0))

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    accuracy = correct / total
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Calculate AU-ROC for binary or multi-class
    n_classes = all_probs.shape[1]
    if n_classes == 2:
        # Binary classification - use probability of positive class
        auroc = roc_auc_score(all_labels, all_probs[:, 1])
    
    metrics = {
        'loss': total_loss / len(test_loader),
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'acc_avg': np.mean(acc_avg),
        'precision': precision,
        'recall': recall,
        'auroc': auroc,
        'all_labels': all_labels,
        'all_preds': all_preds,
        'all_probs': all_probs
    }
    
    return metrics

def plot_roc_curve(all_labels, all_probs, save_path):
    """Plot ROC curve for binary classification"""
    n_classes = all_probs.shape[1]
    
    plt.figure(figsize=(8, 6))
    
    if n_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        auroc = roc_auc_score(all_labels, all_probs[:, 1])
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auroc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"ROC curve saved to {save_path}")

def plot_precision_recall_curve(all_labels, all_probs, save_path):
    """Plot Precision-Recall curve"""
    n_classes = all_probs.shape[1]
    
    plt.figure(figsize=(8, 6))
    
    if n_classes == 2:
        # Binary classification
        precision, recall, _ = precision_recall_curve(all_labels, all_probs[:, 1])
        avg_precision = average_precision_score(all_labels, all_probs[:, 1])
        plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})', linewidth=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Precision-Recall curve saved to {save_path}")

def plot_confusion_matrix(all_labels, all_preds, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14)
    plt.colorbar()
    
    classes = np.unique(all_labels)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Confusion matrix saved to {save_path}")

def extract_subject_id(path):
    parts = os.path.normpath(path).split(os.sep)
    for part in parts:
        if part.startswith("sub-"):
            return part

def filter_df_by_subjects(df, subjects):
    if not subjects:
        return df
    subject_set = set(subjects)
    mask = df["snirf_file"].apply(lambda p: extract_subject_id(p) in subject_set)
    return df[mask].reset_index(drop=True)

def sample_sparse_by_subject(df, fraction, seed=42):
    if fraction is None or fraction >= 1.0:
        return df.reset_index(drop=True)
    if fraction <= 0 or df.empty:
        return df.iloc[0:0].copy()
    df = df.copy()
    df["subject_id"] = df["snirf_file"].apply(extract_subject_id)
    rng = np.random.default_rng(seed=seed)
    sampled = []
    for subject_id, group in df.groupby("subject_id"):
        n = int(round(len(group) * fraction))
        if n < 1:
            n = 1
        n = min(n, len(group))
        sampled.append(group.sample(n=n, random_state=int(rng.integers(0, 1_000_000_000))))
    sampled_df = pd.concat(sampled, ignore_index=True)
    return sampled_df.drop(columns=["subject_id"])

def write_csv(df, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    df.to_csv(out_path, index=False)
    return out_path

def run_mixed_training():
    num_epochs = 400
    learning_rate = 1e-4
    batch_size = 16
    random_state = 42
    chromo = "HbO"
    sparse_sample_ratio = 1.0       # 0.1, 0.3, 0.5, 0.7, 1.0
    USE_CLASS_WEIGHTS = False  # Set to False to disable class weights

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset configuration
    # ballsqueezing_subjects = [
    #     'sub-170', 'sub-173', 'sub-171', 'sub-174',
    #     'sub-176', 'sub-179', 'sub-182', 'sub-177',
    #     'sub-181', 'sub-183', 'sub-184', 'sub-185'
    # ]
    # freshmotor_subjects = [
    #     'sub-01', 'sub-02', 'sub-03', 'sub-04',
    #     'sub-05', 'sub-06', 'sub-07', 'sub-08',
    #     'sub-09', 'sub-10'
    # ]
    # bs_laura_subjects = [
    #     'sub-538', 'sub-580', 'sub-586', 'sub-587',
    #     'sub-592', 'sub-613', 'sub-618', 'sub-619',
    #     'sub-621', 'sub-633', 'sub-638', 'sub-639',
    #     'sub-640'
    # ]
    
    # vfc_hd_subjects = [
    #     'sub-01', 'sub-06', 'sub-08', 'sub-09', 
    #     'sub-11', 'sub-12',  'sub-14', # 'sub-13',
    #     'sub-15', 'sub-17', 'sub-20', 'sub-22', 
    #     'sub-23', 'sub-24', 'sub-25', 'sub-26',
    #     'sub-27'
    # ]
    
    # Anderson_sparse_subjects = [
    #     'sub-1', 'sub-2', 'sub-3', 'sub-4', 
    #     'sub-5', 'sub-6', 'sub-7', 'sub-8',
    #     'sub-9', 'sub-10', 'sub-11', 'sub-12',
    #     'sub-13', 'sub-14', 'sub-15', 'sub-16',
    #     'sub-17'
    # ]
    
    # laura_exclude_subjects = ['sub-547', 'sub-549', 'sub-639', 'sub-588']
    laura_multi_sparse_motor_chs_subjects = [
        'sub-568', 'sub-577',
        'sub-580', 'sub-581', 'sub-583', 'sub-586',
        'sub-587', 'sub-592', 'sub-613',
        'sub-618', 'sub-619', 'sub-621', 'sub-633',
        'sub-638', 'sub-640',
    ]

    # train_datasets_config = {
    #     # "BallSqueezing_dense": {
    #     #     "root": "datasets/processed/parcel_BallSqueezingHD_modified",
    #     #     "subjects": ballsqueezing_subjects,
    #     #     "exclude_subjects": [],
    #     #     "sample_ratio": 1.0,
    #     # },
    #     "BS_Laura_sparse": {
    #         "root": "datasets/motor_processed/BS_Laura",
    #         "subjects": bs_laura_subjects,
    #         "sample_ratio": 1.0,
    #         "exclude_subjects": [],
    #     },        
    #     # "BS_Laura_sparse": {
    #     #     "root": "datasets/motor_processed/BS_Laura",
    #     #     "subjects": bs_laura_subjects,
    #     #     "sample_ratio": sparse_sample_ratio,
    #     #     "exclude_subjects": [],
    #     # },
    # }

    # eval_datasets_config = {
    #     # "BallSqueezing_dense": {
    #     #     "root": "datasets/processed/parcel_BallSqueezingHD_modified",
    #     #     "subjects": ballsqueezing_subjects,
    #     #     "exclude_subjects": [],
    #     #     "sample_ratio": 1.0,
    #     # },
    #     "BS_Laura_sparse": {
    #         "root": "datasets/motor_processed/BS_Laura",
    #         "subjects": bs_laura_subjects,
    #         "sample_ratio": 1.0,
    #         "exclude_subjects": [],
    #     },
    # }

    train_datasets_config = {
        # "BallSqueezing_dense": {
        #     "root": "datasets/processed/parcel_BallSqueezingHD_modified",
        #     "subjects": ballsqueezing_subjects,
        #     "exclude_subjects": [],
        #     "sample_ratio": 1.0,
        # },
        # "vfc_hd_dense": {
        #     "root": "datasets/processed/vfc_hd",
        #     "subjects": vfc_hd_subjects,
        #     "sample_ratio": 1.0,
        #     "exclude_subjects": [],
        # },     
        # "Anderson_sparse": {
        #     "root": "datasets/processed/Anderson_sparse",
        #     "subjects": Anderson_sparse_subjects,
        #     "sample_ratio": sparse_sample_ratio,
        #     "exclude_subjects": [],  
        # },
        
        "full": {
            "root": "datasets/processed/BS_Laura/full",
            "subjects": laura_multi_sparse_motor_chs_subjects,
            "sample_ratio": 1.0,
            "exclude_subjects": [],
        },
        
        "motor_100chs": {
            "root": "datasets/processed/BS_Laura/motor_100chs",
            "subjects": laura_multi_sparse_motor_chs_subjects,
            "sample_ratio": sparse_sample_ratio,
            "exclude_subjects": [],
        },
        
        "motor_91chs": {
            "root": "datasets/processed/BS_Laura/motor_91chs",
            "subjects": laura_multi_sparse_motor_chs_subjects,
            "sample_ratio": sparse_sample_ratio,
            "exclude_subjects": [],
        }, 
         
        "motor_80chs": {
            "root": "datasets/processed/BS_Laura/motor_80chs",
            "subjects": laura_multi_sparse_motor_chs_subjects,
            "sample_ratio": sparse_sample_ratio,
            "exclude_subjects": [],
        },
        
        "motor_70chs": {
            "root": "datasets/processed/BS_Laura/motor_70chs",
            "subjects": laura_multi_sparse_motor_chs_subjects,
            "sample_ratio": sparse_sample_ratio,
            "exclude_subjects": [],
        },

        "motor_59chs": {
            "root": "datasets/processed/BS_Laura/motor_59chs",
            "subjects": laura_multi_sparse_motor_chs_subjects,
            "sample_ratio": sparse_sample_ratio,
            "exclude_subjects": [],
        },   
        
        "motor_50chs": {
            "root": "datasets/processed/BS_Laura/motor_50chs",
            "subjects": laura_multi_sparse_motor_chs_subjects,
            "sample_ratio": sparse_sample_ratio,
            "exclude_subjects": [],
        },        
        
    }

    eval_datasets_config = {
        # "BallSqueezing_dense": {
        #     "root": "datasets/processed/parcel_BallSqueezingHD_modified",
        #     "subjects": ballsqueezing_subjects,
        #     "exclude_subjects": [],
        #     "sample_ratio": 1.0,
        # },
        # "vfc_hd_dense": {
        #     "root": "datasets/processed/vfc_hd",
        #     "subjects": vfc_hd_subjects,
        #     "sample_ratio": 1.0,
        #     "exclude_subjects": [],
        # },
        # "Anderson_sparse": {
        #     "root": "datasets/processed/Anderson_sparse",
        #     "subjects": Anderson_sparse_subjects,
        #     "sample_ratio": 1.0,
        #     "exclude_subjects": [],  
        # },
        "laura_full": {
            "root": "datasets/processed/BS_Laura/full",
            "subjects": laura_multi_sparse_motor_chs_subjects,
            "sample_ratio": 1.0,
            "exclude_subjects": [],
        },
    }
    fold_dataset_name = list(eval_datasets_config.keys())[0]
    subject_ids = eval_datasets_config[fold_dataset_name]["subjects"]
    k = len(subject_ids) # Number of folds

    train_names = list(train_datasets_config.keys())
    eval_names = list(eval_datasets_config.keys())
    run_name = f"train_{'+'.join(train_names)}__eval_{'+'.join(eval_names)}"
    # results_dir = f"results/{run_name}_onlyBS_Laura"
    results_dir = f"results/{run_name}_{sparse_sample_ratio}"
    csv_dir = os.path.join(results_dir, "csv")
    os.makedirs(os.path.join(results_dir, "checkpoints"), exist_ok=True)

    # Shuffle the subject list
    rng = np.random.default_rng(seed=random_state)
    shuffled_subjects = rng.permutation(subject_ids)

    # Split into k roughly equal folds
    folds = np.array_split(shuffled_subjects, k)
    folds = [list(fold) for fold in folds]

    for fold_idx, fold in enumerate(folds):
        subs = "_".join(fold)

        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        train_datasets = []
        for name, cfg in train_datasets_config.items():
            ratio = cfg.get("sample_ratio", 1.0)
            if ratio <= 0:
                continue
            meta_tag = f"meta_files_{ratio}"
            test_subjects_list = fold if name == fold_dataset_name else []
            if test_subjects_list:
                logging.info(
                    f"Excluded test subjects for training ({name}): {', '.join(test_subjects_list)}"
                )
            else:
                logging.info(f"Excluded test subjects for training ({name}): none")
            train_csv_path, _ = create_train_test_segments(
                None,
                cfg["root"],
                test_subjects_list=test_subjects_list,
                exclude_subjects=cfg.get("exclude_subjects", []),
                meta_tag=meta_tag,
            )
            train_df = pd.read_csv(train_csv_path)
            train_df = filter_df_by_subjects(train_df, cfg.get("subjects"))
            train_df = sample_sparse_by_subject(
                train_df,
                ratio,
                seed=random_state + fold_idx,
            )
            if train_df.empty:
                raise ValueError(f"No training samples found for dataset: {name}")
            train_csv = write_csv(train_df, csv_dir, f"train_{name}_{subs}.csv")
            train_datasets.append(fNIRSPreloadDataset(train_csv, chromo='HbO'))

        if len(train_datasets) == 1:
            train_dataset = train_datasets[0]
        else:
            train_dataset = ConcatDataset(train_datasets)

        eval_datasets = []
        for name, cfg in eval_datasets_config.items():
            ratio = cfg.get("sample_ratio", 1.0)
            meta_tag = f"meta_files_{ratio}"
            if name == fold_dataset_name:
                test_subjects_list = fold
            else:
                test_subjects_list = cfg.get("subjects", [])
            _, test_csv_path = create_train_test_segments(
                None,
                cfg["root"],
                test_subjects_list=test_subjects_list,
                exclude_subjects=cfg.get("exclude_subjects", []),
                meta_tag=meta_tag,
            )
            test_df = pd.read_csv(test_csv_path)
            test_df = filter_df_by_subjects(test_df, cfg.get("subjects"))
            if test_df.empty:
                raise ValueError(f"No evaluation samples found for dataset: {name}")
            test_csv = write_csv(test_df, csv_dir, f"test_{name}_{subs}.csv")
            eval_datasets.append(fNIRSPreloadDataset(test_csv, mode="test", chromo='HbO'))

        if len(eval_datasets) == 1:
            test_dataset = eval_datasets[0]
        else:
            test_dataset = ConcatDataset(eval_datasets)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # Calculate class weights for imbalanced data (optional)
        if USE_CLASS_WEIGHTS:
            class_weights = calculate_class_weights(train_dataset, device)
        else:
            class_weights = None

        # Initialize model, loss, and optimizer
        model = CNN2DImage().to(device)
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        test_f1_micros = []
        test_f1_macros = []
        train_f1_micros = []
        train_f1_macros = []
        test_aurocs = []
        train_aurocs = []
        test_precisions = []
        test_recalls = []
        train_precisions = []
        train_recalls = []

        # Training loop
        for epoch in range(num_epochs):

            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            train_metrics = evaluate_model(model, train_loader, criterion, device)
            test_metrics = evaluate_model(model, test_loader, criterion, device)

            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_metrics['accuracy'])
            test_losses.append(test_metrics['loss'])
            test_accuracies.append(test_metrics['accuracy'])
            
            test_f1_micros.append(test_metrics['f1_micro'])
            test_f1_macros.append(test_metrics['f1_macro'])
            train_f1_micros.append(train_metrics['f1_micro'])
            train_f1_macros.append(train_metrics['f1_macro'])
            
            test_aurocs.append(test_metrics['auroc'])
            train_aurocs.append(train_metrics['auroc'])
            test_precisions.append(test_metrics['precision'])
            test_recalls.append(test_metrics['recall'])
            train_precisions.append(train_metrics['precision'])
            train_recalls.append(train_metrics['recall'])

            logging.info(f"Sub: {subs}, Epoch [{epoch+1}], "
                        f"Train F1: {train_metrics['f1_micro']:.4f}, Test F1: {test_metrics['f1_micro']:.4f}, "
                        f"Train AUROC: {train_metrics['auroc']:.4f}, Test AUROC: {test_metrics['auroc']:.4f}")
       
        # Generate final evaluation plots
        final_train_metrics = evaluate_model(model, train_loader, criterion, device)
        final_test_metrics = evaluate_model(model, test_loader, criterion, device)
        
        plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot ROC curve
        plot_roc_curve(final_test_metrics['all_labels'],
                       final_test_metrics['all_probs'],
                       os.path.join(plots_dir, f"roc_curve_{subs}_{chromo}.png"))
        
        # Plot Precision-Recall curve
        plot_precision_recall_curve(final_test_metrics['all_labels'],
                                    final_test_metrics['all_probs'],
                                    os.path.join(plots_dir, f"pr_curve_{subs}_{chromo}.png"))
        
        # Plot Confusion Matrix
        plot_confusion_matrix(final_test_metrics['all_labels'],
                              final_test_metrics['all_preds'],
                              os.path.join(plots_dir, f"confusion_matrix_{subs}_{chromo}.png"))
        
        res = {
            "train_loss": train_losses, "train_accuracy": train_accuracies,
            "test_loss": test_losses, "test_accuracy": test_accuracies,
            "test_f1_micro": test_f1_micros, "test_f1_macro": test_f1_macros,
            "train_f1_micro": train_f1_micros, "train_f1_macro": train_f1_macros,
            "test_auroc": test_aurocs, "train_auroc": train_aurocs,
            "test_precision": test_precisions, "test_recall": test_recalls,
            "train_precision": train_precisions, "train_recall": train_recalls,
            "final_test_labels": final_test_metrics['all_labels'],
            "final_test_preds": final_test_metrics['all_preds'],
            "final_test_probs": final_test_metrics['all_probs']
        }
        
        with open(f"{results_dir}/res_{subs}_{chromo}.pkl", "wb") as f:
            pickle.dump(res, f)

        torch.save(model.state_dict(), f"{results_dir}/checkpoints/model_{subs}_{chromo}.pth")
        
        print("Model saved successfully!")
    
    print("\n-----Training complete! -----\n")

# Main function
if __name__ == "__main__":
    USE_MIXED_DATASETS = True
    if USE_MIXED_DATASETS:
        run_mixed_training()
        sys.exit(0)

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
    DATASET_NAME = "Parcel_BallSqueezingHD_modified"
    preprocessed_path = os.path.join("datasets/processed", 'parcel_BallSqueezingHD_modified')
        
    os.makedirs(f"results/{DATASET_NAME}/checkpoints/", exist_ok=True)

    
    if DATASET_NAME == "parcel_BallSqueezingHD_modified":
        subject_ids = ['sub-170', 'sub-173', 'sub-171', 'sub-174',
                       'sub-176', 'sub-179', 'sub-182', 'sub-177',
                       'sub-181', 'sub-183', 'sub-184', 'sub-185']
    elif DATASET_NAME == "parcel_FreshMotor":
        subject_ids = ['sub-01', 'sub-02', 'sub-03', 'sub-04',
                       'sub-05', 'sub-06', 'sub-07', 'sub-08',
                       'sub-09', 'sub-10']
    elif DATASET_NAME == "BS_Laura":
        subject_ids = ['sub-538', 'sub-580', 'sub-586', 'sub-587',
                       'sub-592', 'sub-613', 'sub-618', 'sub-619',
                       'sub-621', 'sub-633', 'sub-638', 'sub-639',
                       'sub-640']
        
    k = len(subject_ids)  # = 10 FreshMotor, 12 BSQ-HD (LOSO), 13 BS Laura

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
    # exclude_subjects = ['sub-547', 'sub-639', 'sub-588']
    
    exclude_subjects = ['sub-538', 'sub-547', 'sub-549', 'sub-639', 'sub-588'],
    
    chromo = "HbO"
    for fold in folds:
        subs = "_".join(fold)

        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        if fold:
            logging.info(
                f"Excluded test subjects for training ({DATASET_NAME}): {', '.join(fold)}"
            )
        else:
            logging.info(f"Excluded test subjects for training ({DATASET_NAME}): none")
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

        # Calculate class weights for imbalanced data (optional)
        USE_CLASS_WEIGHTS = True  # Set to False to disable class weights
        if USE_CLASS_WEIGHTS:
            class_weights = calculate_class_weights(train_dataset, device)
        else:
            class_weights = None

        # Initialize model, loss, and optimizer
        model = CNN2DImage().to(device)
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        test_f1_micros = []
        test_f1_macros = []
        train_f1_micros = []
        train_f1_macros = []
        test_aurocs = []
        train_aurocs = []
        test_precisions = []
        test_recalls = []
        train_precisions = []
        train_recalls = []

        # Training loop
        for epoch in range(num_epochs):

            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            train_metrics = evaluate_model(model, train_loader, criterion, device)
            test_metrics = evaluate_model(model, test_loader, criterion, device)

            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_metrics['accuracy'])
            test_losses.append(test_metrics['loss'])
            test_accuracies.append(test_metrics['accuracy'])
            
            test_f1_micros.append(test_metrics['f1_micro'])
            test_f1_macros.append(test_metrics['f1_macro'])
            train_f1_micros.append(train_metrics['f1_micro'])
            train_f1_macros.append(train_metrics['f1_macro'])
            
            test_aurocs.append(test_metrics['auroc'])
            train_aurocs.append(train_metrics['auroc'])
            test_precisions.append(test_metrics['precision'])
            test_recalls.append(test_metrics['recall'])
            train_precisions.append(train_metrics['precision'])
            train_recalls.append(train_metrics['recall'])

            logging.info(f"Sub: {subs}, Epoch [{epoch+1}], "
                        f"Train F1: {train_metrics['f1_micro']:.4f}, Test F1: {test_metrics['f1_micro']:.4f}, "
                        f"Train AUROC: {train_metrics['auroc']:.4f}, Test AUROC: {test_metrics['auroc']:.4f}")
       
        # Generate final evaluation plots
        final_train_metrics = evaluate_model(model, train_loader, criterion, device)
        final_test_metrics = evaluate_model(model, test_loader, criterion, device)
        
        plots_dir = os.path.join(f"results/{DATASET_NAME}", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot ROC curve
        plot_roc_curve(
            final_test_metrics['all_labels'],
            final_test_metrics['all_probs'],
            os.path.join(plots_dir, f"roc_curve_{subs}_{chromo}.png"))
        
        # Plot Precision-Recall curve
        plot_precision_recall_curve(
            final_test_metrics['all_labels'],
            final_test_metrics['all_probs'],
            os.path.join(plots_dir, f"pr_curve_{subs}_{chromo}.png"))
        
        # Plot Confusion Matrix
        plot_confusion_matrix(
            final_test_metrics['all_labels'],
            final_test_metrics['all_preds'],
            os.path.join(plots_dir, f"confusion_matrix_{subs}_{chromo}.png"))
        
        res = {
            "train_loss": train_losses, "train_accuracy": train_accuracies,
            "test_loss": test_losses, "test_accuracy": test_accuracies,
            "test_f1_micro": test_f1_micros, "test_f1_macro": test_f1_macros,
            "train_f1_micro": train_f1_micros, "train_f1_macro": train_f1_macros,
            "test_auroc": test_aurocs, "train_auroc": train_aurocs,
            "test_precision": test_precisions, "test_recall": test_recalls,
            "train_precision": train_precisions, "train_recall": train_recalls,
            "final_test_labels": final_test_metrics['all_labels'],
            "final_test_preds": final_test_metrics['all_preds'],
            "final_test_probs": final_test_metrics['all_probs']
        }
        
        with open(f"results/{DATASET_NAME}/res_{subs}_{chromo}.pkl", "wb") as f:
            pickle.dump(res, f)

        torch.save(model.state_dict(), f"results/{DATASET_NAME}/checkpoints/model_{subs}_{chromo}.pth")
        
        print("Model saved successfully!")
    
    print("\n-----Training complete! -----\n")
