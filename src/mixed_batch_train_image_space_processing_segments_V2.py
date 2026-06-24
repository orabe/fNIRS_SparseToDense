from torch.utils.data import DataLoader, ConcatDataset
import pickle
from datasets_v02 import fNIRSChannelSpaceSegmentLoad, fNIRSPreloadDataset
from online_augmentations import ONLINE_AUGMENTATION_GROUPS, ONLINE_AUGMENTATIONS
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # dataset_name = "BallSqueezingHD_modified"
    # dataset_name = "BS_Laura"
    dataset_name = "vfc_hd"
    num_epochs = 400
    learning_rate = 1e-4
    batch_size = 16
    random_state = 42
    chromo = "both"
    USE_CLASS_WEIGHTS= False
    experiment_mode = "online_eeg_aug"  # "legacy" or "online_eeg_aug"
    representation = "channel"  # "channel" or "parcel"
    online_aug_name = "space_shuffle"
    online_aug_params = {"aug_prob": 0.5, "drop_prob": 0.1}

    augmentation_strategy = "imageRecon_params"  # "channel_density" or "imageRecon_params"
    sparse_sample_ratio = 1.0       # 0.1, 0.3, 0.5, 0.7, 1.0

    # Shared sampling ratio for non-baseline image-recon parameter views.
    # The baseline view am_1__as_1 is always kept at 1.0.
    # Must be a float fraction in [0.0, 1.0], e.g. 0.0, 0.1, 0.3, 0.5, 0.7, 1.0.
    recon_param_aug_sample_ratio = 0.0
    
    if not 0.0 <= recon_param_aug_sample_ratio <= 1.0:
        raise ValueError("recon_param_aug_sample_ratio must be a float fraction between 0.0 and 1.0")

    recon_param_sample_ratios = {
        "am_0.1__as_0.1": recon_param_aug_sample_ratio,
        "am_0.1__as_1": recon_param_aug_sample_ratio,
        "am_0.1__as_10": recon_param_aug_sample_ratio,
        "am_1__as_0.1": recon_param_aug_sample_ratio,
        "am_1__as_1": 1.0,
        "am_1__as_10": recon_param_aug_sample_ratio,
        "am_10__as_0.1": recon_param_aug_sample_ratio,
        "am_10__as_1": recon_param_aug_sample_ratio,
        "am_10__as_10": recon_param_aug_sample_ratio,
    }

    ballsqueezing_subjects = [
        'sub-170', 'sub-173', 'sub-171', 'sub-174',
        'sub-176', 'sub-179', 'sub-182', 'sub-177',
        'sub-181', 'sub-183', 'sub-184', 'sub-185',
    ]

    laura_multi_sparse_motor_chs_subjects = [
        'sub-568', 'sub-577',
        'sub-580', 'sub-581', 'sub-583', 'sub-586',
        'sub-587', 'sub-592', 'sub-613',
        'sub-618', 'sub-619', 'sub-621', 'sub-633',
        'sub-638', 'sub-640',
    ]

    vfc_hd_subjects = [
        "sub-01", "sub-06", "sub-08", "sub-09",
        "sub-11", "sub-12", "sub-14", "sub-15",
        "sub-17", "sub-20", "sub-22", "sub-23",
        "sub-24", "sub-25", "sub-26", "sub-27",
    ]

    image_recon_subjects_by_dataset = {
        "BallSqueezingHD_modified": ballsqueezing_subjects,
        "BS_Laura": laura_multi_sparse_motor_chs_subjects,
        "vfc_hd": vfc_hd_subjects,
    }
    online_data_roots = {
        "channel": {
            "BS_Laura": "datasets/processed/channel_space/BS_Laura/full",
            "vfc_hd": "datasets/processed/channel_space/vfc_hd/full",
        },
        "parcel": {
            "BS_Laura": "datasets/processed/imageRecon_params/BS_Laura/full/am_1__as_1",
            "vfc_hd": "datasets/processed/imageRecon_params/vfc_hd/full/am_1__as_1",
        },
    }

    def get_image_recon_subjects(dataset_name):
        if dataset_name not in image_recon_subjects_by_dataset:
            supported = ", ".join(sorted(image_recon_subjects_by_dataset))
            raise ValueError(
                f"No image-recon subject list configured for dataset_name={dataset_name!r}. "
                f"Supported datasets: {supported}"
            )
        return image_recon_subjects_by_dataset[dataset_name]

    recon_param_folders = [
        "am_0.1__as_0.1",
        "am_0.1__as_1",
        "am_0.1__as_10",
        "am_1__as_0.1",
        "am_1__as_1",
        "am_1__as_10",
        "am_10__as_0.1",
        "am_10__as_1",
        "am_10__as_10",
    ]
    default_recon_param_folder = "am_1__as_1"


    def make_dataset_config(root, subjects, dataset_name, sample_ratio=1.0, exclude_subjects=None):
        return {
            "root": root,
            "subjects": subjects,
            "dataset_name": dataset_name,
            "sample_ratio": sample_ratio,
            "exclude_subjects": exclude_subjects or [],
        }


    if experiment_mode == "online_eeg_aug":
        online_subjects = get_image_recon_subjects(dataset_name)
        online_root = online_data_roots[representation][dataset_name]
        train_datasets_config = {
            f"{representation}_online": make_dataset_config(
                online_root,
                online_subjects,
                dataset_name,
                1.0,
            )
        }
        eval_datasets_config = {
            f"{representation}_online": make_dataset_config(
                online_root,
                online_subjects,
                dataset_name,
                1.0,
            )
        }
    elif augmentation_strategy == "imageRecon_params":
        # Train on the 3x3 image-reconstruction parameter grid for the selected dataset.
        # Test only on the default reconstruction parameters: am_1__as_1.
        image_recon_subjects = get_image_recon_subjects(dataset_name)
        base_processed_root = f"datasets/processed/{augmentation_strategy}/{dataset_name}/full"
        train_datasets_config = {
            folder: make_dataset_config(
                os.path.join(base_processed_root, folder),
                image_recon_subjects,
                dataset_name,
                recon_param_sample_ratios[folder],
            )
            for folder in recon_param_folders
        }
        eval_datasets_config = {
            default_recon_param_folder: make_dataset_config(
                os.path.join(base_processed_root, default_recon_param_folder),
                image_recon_subjects,
                dataset_name,
                1.0,
            )
        }

    elif augmentation_strategy == "channel_density":
        # Previous augmentation strategy: pool different BS_Laura channel-density reconstructions.
        train_datasets_config = {
            "full": make_dataset_config(
                "datasets/processed/BS_Laura/full",
                laura_multi_sparse_motor_chs_subjects,
                "BS_Laura",
                1.0,
            ),
            "motor_100chs": make_dataset_config(
                "datasets/processed/BS_Laura/motor_100chs",
                laura_multi_sparse_motor_chs_subjects,
                "BS_Laura",
                sparse_sample_ratio,
            ),
            "motor_91chs": make_dataset_config(
                "datasets/processed/BS_Laura/motor_91chs",
                laura_multi_sparse_motor_chs_subjects,
                "BS_Laura",
                sparse_sample_ratio,
            ),
            "motor_80chs": make_dataset_config(
                "datasets/processed/BS_Laura/motor_80chs",
                laura_multi_sparse_motor_chs_subjects,
                "BS_Laura",
                sparse_sample_ratio,
            ),
            "motor_70chs": make_dataset_config(
                "datasets/processed/BS_Laura/motor_70chs",
                laura_multi_sparse_motor_chs_subjects,
                "BS_Laura",
                sparse_sample_ratio,
            ),
            "motor_59chs": make_dataset_config(
                "datasets/processed/BS_Laura/motor_59chs",
                laura_multi_sparse_motor_chs_subjects,
                "BS_Laura",
                sparse_sample_ratio,
            ),
            "motor_50chs": make_dataset_config(
                "datasets/processed/BS_Laura/motor_50chs",
                laura_multi_sparse_motor_chs_subjects,
                "BS_Laura",
                sparse_sample_ratio,
            ),
        }
        eval_datasets_config = {
            "laura_full": make_dataset_config(
                "datasets/processed/BS_Laura/full",
                laura_multi_sparse_motor_chs_subjects,
                "BS_Laura",
                1.0,
            )
        }

    else:
        raise ValueError(f"Unknown augmentation_strategy: {augmentation_strategy}")

    fold_dataset_name = list(eval_datasets_config.keys())[0]
    subject_ids = eval_datasets_config[fold_dataset_name]["subjects"]
    k = len(subject_ids) # Number of folds

    train_names = list(train_datasets_config.keys())
    eval_names = list(eval_datasets_config.keys())
    if experiment_mode == "online_eeg_aug":
        run_name = f"train_{dataset_name}_{representation}_{online_aug_name}_{chromo}"
    elif augmentation_strategy == "imageRecon_params":
        active_train_names = [
            name for name, cfg in train_datasets_config.items()
            if cfg.get("sample_ratio", 1.0) > 0
        ]
        ratio_tag = f"{recon_param_aug_sample_ratio:.1f}"
        run_name = f"train_{dataset_name}_imgRecon_{len(active_train_names)}am-as_ratio_{chromo}_{ratio_tag}"
    else:
        run_name = f"train_{'+'.join(train_names)}__eval_{'+'.join(eval_names)}_{chromo}"

    unique_dataset_names = sorted(
        {
            cfg["dataset_name"]
            for cfg in list(train_datasets_config.values()) + list(eval_datasets_config.values())
        }
    )
    if experiment_mode == "online_eeg_aug":
        result_group = f"online_eeg_aug__{representation}__{'+'.join(unique_dataset_names)}"
    else:
        result_group = f"{augmentation_strategy}__{'+'.join(unique_dataset_names)}"
    results_dir = os.path.join("results", result_group, run_name)
    csv_dir = os.path.join(results_dir, "csv")
    os.makedirs(os.path.join(results_dir, "checkpoints"), exist_ok=True)

    # Shuffle the subject list
    rng = np.random.default_rng(seed=random_state)
    shuffled_subjects = rng.permutation(subject_ids)

    # Split into k roughly equal folds
    folds = np.array_split(shuffled_subjects, k)
    folds = [list(fold) for fold in folds]

    logging.info(
        f"Experiment mode={experiment_mode}, representation={representation}, "
        f"online_aug_name={online_aug_name}, online_aug_params={online_aug_params}"
    )
    logging.info(f"Available online augmentations: {', '.join(ONLINE_AUGMENTATIONS)}")
    logging.info(
        "Online augmentation groups: "
        + "; ".join(
            f"{group}=[{', '.join(methods)}]"
            for group, methods in ONLINE_AUGMENTATION_GROUPS.items()
        )
    )

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
            # Exclude the held-out LOSO subject from every training root.
            # This prevents leakage across augmentation views of the same subject.
            test_subjects_list = fold
            logging.info(
                f"Excluded test subjects for training ({name}): {', '.join(test_subjects_list)}"
            )
            train_df, _ = create_train_test_segments(
                None,
                cfg["root"],
                test_subjects_list=test_subjects_list,
                exclude_subjects=cfg.get("exclude_subjects", []),
            )
            train_df = filter_df_by_subjects(train_df, cfg.get("subjects"))
            train_df = sample_sparse_by_subject(
                train_df,
                ratio,
                seed=random_state + fold_idx,
            )
            if train_df.empty:
                raise ValueError(f"No training samples found for dataset: {name}")
            train_csv = write_csv(train_df, csv_dir, f"train_{name}_{subs}.csv")
            train_datasets.append(
                fNIRSPreloadDataset(
                    train_csv,
                    chromo=chromo,
                    aug_name=online_aug_name if experiment_mode == "online_eeg_aug" else "none",
                    aug_params=online_aug_params if experiment_mode == "online_eeg_aug" else None,
                    seed=random_state + fold_idx,
                )
            )

        if len(train_datasets) == 1:
            train_dataset = train_datasets[0]
        else:
            train_dataset = ConcatDataset(train_datasets)

        eval_datasets = []
        for name, cfg in eval_datasets_config.items():
            ratio = cfg.get("sample_ratio", 1.0)
            if name == fold_dataset_name:
                test_subjects_list = fold
            else:
                test_subjects_list = cfg.get("subjects", [])
            _, test_df = create_train_test_segments(
                None,
                cfg["root"],
                test_subjects_list=test_subjects_list,
                exclude_subjects=cfg.get("exclude_subjects", []),
            )
            test_df = filter_df_by_subjects(test_df, cfg.get("subjects"))
            if test_df.empty:
                raise ValueError(f"No evaluation samples found for dataset: {name}")
            test_csv = write_csv(test_df, csv_dir, f"test_{name}_{subs}.csv")
            eval_datasets.append(
                fNIRSPreloadDataset(
                    test_csv,
                    mode="test",
                    chromo=chromo,
                    aug_name="none",
                    aug_params=None,
                    seed=random_state + fold_idx,
                )
            )

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
    
    logging.info(f"Training outputs stored in: {results_dir}")
    logging.info(f"CSV files stored in: {csv_dir}")
    logging.info(f"Checkpoints stored in: {os.path.join(results_dir, 'checkpoints')}")
    logging.info(f"Plots stored in: {os.path.join(results_dir, 'plots')}")
    print("\n-----Training complete! -----\n")
    print(f"Training outputs stored in: {results_dir}")

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
    
    chromo = "both"
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
        train_df, test_df = create_train_test_segments(
            None,
            preprocessed_path,
            test_subjects_list=fold,
            exclude_subjects=exclude_subjects
        )
        train_dataset = fNIRSPreloadDataset(
            train_df, chromo=chromo)
        test_dataset = fNIRSPreloadDataset(
            test_df, mode="test", chromo=chromo)
        
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
