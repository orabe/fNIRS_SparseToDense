from torch.utils.data import DataLoader, ConcatDataset
import pickle
import torch
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, precision_score, recall_score, confusion_matrix
import logging
import sys
import warnings

warnings.filterwarnings("ignore")

from utils import create_train_test_segments
from datasets_v02 import fNIRSPreloadDataset
from model import CNN2DImage

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


PROCESSED_ROOT = "datasets/processed"
RESULTS_ROOT = "results"

DATASET_SUBJECTS = {
    "BallSqueezingHD_modified": [
        "sub-170", "sub-173", "sub-171", "sub-174",
        "sub-176", "sub-179", "sub-182", "sub-177",
        "sub-181", "sub-183", "sub-184", "sub-185",
    ],
    "BS_Laura": [
        "sub-568", "sub-577", "sub-580", "sub-581",
        "sub-583", "sub-586", "sub-587", "sub-592",
        "sub-613", "sub-618", "sub-619", "sub-621",
        "sub-633", "sub-638", "sub-640",
    ],
    "vfc_hd": [
        "sub-01", "sub-06", "sub-08", "sub-09",
        "sub-11", "sub-12", "sub-14", "sub-15",
        "sub-17", "sub-20", "sub-22", "sub-23",
        "sub-24", "sub-25", "sub-26", "sub-27",
    ],
    "Anderson_sparse": [
        "sub-1", "sub-2", "sub-3", "sub-4",
        "sub-5", "sub-6", "sub-7", "sub-8",
        "sub-9", "sub-10", "sub-11", "sub-12",
        "sub-13", "sub-14", "sub-15", "sub-16",
        "sub-17",
    ],
}

CHANNEL_DENSITY_SUBSET_FOLDERS = {
    "BS_Laura": [
        "subset_90_chs",
        "subset_80_chs",
        "subset_70_chs",
        "subset_60_chs",
        "subset_50_chs",
    ],
    "BallSqueezingHD_modified": [
        "subset_90_chs",
        "subset_80_chs",
        "subset_70_chs",
        "subset_60_chs",
        "subset_50_chs",
    ],
    "vfc_hd": [
        "subset_90_chs",
        "subset_80_chs",
        "subset_70_chs",
        "subset_60_chs",
        "subset_50_chs",
    ],
    "Anderson_sparse": ["full"],
}

RECON_PARAM_FOLDERS = [
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

ONLINE_AUG_DEFAULT_PARAMS = {
    "none": None,
    "gaussian_noise": {"aug_prob": 0.5, "std": 0.01},
    "smooth_time_mask": {"aug_prob": 0.5, "mask_fraction": 0.1},
    "time_reverse": {"aug_prob": 0.5},
    "sign_flip": {"aug_prob": 0.5},
    "ft_surrogate": {"aug_prob": 0.5, "phase_scale": np.pi},
    "frequency_shift": {"aug_prob": 0.5, "shift_bins": 1},
    "bandstop_filter": {"aug_prob": 0.5, "band_width": 2},
    "space_symmetry": {"aug_prob": 0.5, "symmetry_pairs": None},
    "space_dropout": {"aug_prob": 0.5, "drop_prob": 0.1},
    "space_shuffle": {"aug_prob": 0.5},
}


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
    if subjects is None:
        return df
    subject_set = set(subjects)
    mask = df["snirf_file"].apply(lambda p: extract_subject_id(p) in subject_set)
    return df[mask].reset_index(drop=True)

def sample_sparse_by_subject(df, fraction, seed=42):
    if fraction >= 1.0:
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

def get_subjects(subjects_by_dataset, dataset_name):
    if dataset_name not in subjects_by_dataset:
        supported = ", ".join(sorted(subjects_by_dataset))
        raise ValueError(
            f"No subject list configured for dataset_name={dataset_name!r}. "
            f"Supported datasets: {supported}"
        )
    return subjects_by_dataset[dataset_name]

def make_dataset_config(root, subjects, dataset_name, sample_ratio, exclude_subjects):
    return {
        "root": root,
        "subjects": subjects,
        "dataset_name": dataset_name,
        "sample_ratio": sample_ratio,
        "exclude_subjects": exclude_subjects,
    }

def build_online_eeg_aug_strategy(target_dataset_name, strategy_config, subjects_by_dataset):
    """Build train/eval configs for online augmentation on one base dataset view."""
    subset_name = strategy_config["subset_name"]
    representation = strategy_config["representation"]
    online_aug_name = strategy_config["online_aug_name"]
    online_aug_params = strategy_config["online_aug_params"]
    target_subjects = get_subjects(subjects_by_dataset, target_dataset_name)
    root = f"{PROCESSED_ROOT}/base/{target_dataset_name}/{subset_name}/{representation}_space"
    dataset_config = make_dataset_config(root, target_subjects, target_dataset_name, 1.0, [])
    results_dir = os.path.join(
        RESULTS_ROOT,
        "online_eeg_aug",
        f"target_{target_dataset_name}",
        subset_name,
        f"{representation}_space",
        online_aug_name,
    )
    return (
        {f"{representation}_online": dataset_config},
        {f"{representation}_online": dataset_config},
        results_dir,
        online_aug_name,
        online_aug_params,
    )

def build_base_strategy(target_dataset_name, strategy_config, subjects_by_dataset):
    """Build train/eval configs for no-augmentation baseline training."""
    subset_name = strategy_config["subset_name"]
    representation = strategy_config["representation"]
    target_subjects = get_subjects(subjects_by_dataset, target_dataset_name)
    root = f"{PROCESSED_ROOT}/base/{target_dataset_name}/{subset_name}/{representation}_space"
    dataset_config = make_dataset_config(root, target_subjects, target_dataset_name, 1.0, [])
    results_dir = os.path.join(
        RESULTS_ROOT,
        "base",
        f"target_{target_dataset_name}",
        subset_name,
        f"{representation}_space",
    )
    return (
        {f"{target_dataset_name}_{subset_name}_{representation}": dataset_config},
        {f"{target_dataset_name}_{subset_name}_{representation}": dataset_config},
        results_dir,
        "none",
        None,
    )

def build_image_recon_strategy(target_dataset_name, strategy_config, subjects_by_dataset):
    """Build train/eval configs for image-reconstruction parameter augmentation."""
    recon_param_aug_sample_ratio = strategy_config["recon_param_aug_sample_ratio"]
    if recon_param_aug_sample_ratio < 0 or recon_param_aug_sample_ratio > 1:
        raise ValueError("recon_param_aug_sample_ratio must be between 0 and 1")

    recon_param_folders = strategy_config["recon_param_folders"]
    default_recon_param_folder = strategy_config["default_recon_param_folder"]
    recon_param_sample_ratios = {
        folder: recon_param_aug_sample_ratio
        for folder in recon_param_folders
    }
    recon_param_sample_ratios[default_recon_param_folder] = 1.0
    target_subjects = get_subjects(subjects_by_dataset, target_dataset_name)
    base_processed_root = f"{PROCESSED_ROOT}/imageRecon_params/{target_dataset_name}/full"
    train_datasets_config = {
        folder: make_dataset_config(
            os.path.join(base_processed_root, folder, "parcel_space"),
            target_subjects,
            target_dataset_name,
            recon_param_sample_ratios[folder],
            [],
        )
        for folder in recon_param_folders
    }
    eval_datasets_config = {
        default_recon_param_folder: make_dataset_config(
            os.path.join(base_processed_root, default_recon_param_folder, "parcel_space"),
            target_subjects,
            target_dataset_name,
            1.0,
            [],
        )
    }
    results_dir = os.path.join(
        RESULTS_ROOT,
        "imageRecon_params",
        f"target_{target_dataset_name}",
        "parcel_space",
        f"ratio_{recon_param_aug_sample_ratio:.1f}",
    )
    return train_datasets_config, eval_datasets_config, results_dir, "none", None

def build_channel_density_strategy(
    target_dataset_name,
    strategy_config,
    subjects_by_dataset,
):
    """Build train/eval configs for parcel-space channel-density augmentation."""
    source_dataset_name = strategy_config["source_dataset_name"]
    sparse_sample_ratio = strategy_config["sparse_sample_ratio"]
    subset_folders = strategy_config["subset_folders"]
    if sparse_sample_ratio < 0 or sparse_sample_ratio > 1:
        raise ValueError("sparse_sample_ratio must be between 0 and 1")

    target_subjects = get_subjects(subjects_by_dataset, target_dataset_name)
    source_subjects = get_subjects(subjects_by_dataset, source_dataset_name)
    train_datasets_config = {
        f"target_{target_dataset_name}_full": make_dataset_config(
            f"{PROCESSED_ROOT}/base/{target_dataset_name}/full/parcel_space",
            target_subjects,
            target_dataset_name,
            1.0,
            [],
        )
    }
    source_result_name = source_dataset_name
    if sparse_sample_ratio == 0:
        source_result_name = "none"
    else:
        for source_subset_name in subset_folders[source_dataset_name]:
            train_datasets_config[f"source_{source_dataset_name}_{source_subset_name}"] = make_dataset_config(
                f"{PROCESSED_ROOT}/base/{source_dataset_name}/{source_subset_name}/parcel_space",
                source_subjects,
                source_dataset_name,
                sparse_sample_ratio,
                [],
            )
    eval_datasets_config = {
        f"target_{target_dataset_name}_full": make_dataset_config(
            f"{PROCESSED_ROOT}/base/{target_dataset_name}/full/parcel_space",
            target_subjects,
            target_dataset_name,
            1.0,
            [],
        )
    }
    results_dir = os.path.join(
        RESULTS_ROOT,
        "channel_density",
        f"target_{target_dataset_name}",
        f"source_{source_result_name}",
        "parcel_space",
        f"ratio_{sparse_sample_ratio:.1f}",
    )
    return train_datasets_config, eval_datasets_config, results_dir, "none", None

def run_mixed_training(run_strategy, config, train_params):
    """Build the selected data strategy and run LOSO training/evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = train_params["num_epochs"]
    learning_rate = train_params["learning_rate"]
    batch_size = train_params["batch_size"]
    random_state = train_params["random_state"]
    chromo = train_params["chromo"]
    use_class_weights = train_params["use_class_weights"]
    target_dataset_name = config["target_dataset_name"]
    if run_strategy == "online_eeg_aug":
        (
            train_datasets_config,
            eval_datasets_config,
            results_dir,
            train_aug_name,
            train_aug_params,
        ) = build_online_eeg_aug_strategy(
            target_dataset_name,
            config["online_eeg_aug"],
            DATASET_SUBJECTS,
        )
    elif run_strategy == "base":
        (
            train_datasets_config,
            eval_datasets_config,
            results_dir,
            train_aug_name,
            train_aug_params,
        ) = build_base_strategy(
            target_dataset_name,
            config["base"],
            DATASET_SUBJECTS,
        )
    elif run_strategy == "imageRecon_params":
        (
            train_datasets_config,
            eval_datasets_config,
            results_dir,
            train_aug_name,
            train_aug_params,
        ) = build_image_recon_strategy(
            target_dataset_name,
            config["imageRecon_params"],
            DATASET_SUBJECTS,
        )
    elif run_strategy == "channel_density":
        (
            train_datasets_config,
            eval_datasets_config,
            results_dir,
            train_aug_name,
            train_aug_params,
        ) = build_channel_density_strategy(
            target_dataset_name,
            config["channel_density"],
            DATASET_SUBJECTS,
        )
    else:
        raise ValueError(f"Unknown run_strategy: {run_strategy}")

    fold_dataset_name = list(eval_datasets_config.keys())[0]
    subject_ids = eval_datasets_config[fold_dataset_name]["subjects"]
    k = len(subject_ids) # Number of folds

    csv_dir = os.path.join(results_dir, "csv")
    os.makedirs(os.path.join(results_dir, "checkpoints"), exist_ok=True)

    # Shuffle the subject list
    rng = np.random.default_rng(seed=random_state)
    shuffled_subjects = rng.permutation(subject_ids)

    # Split into k roughly equal folds
    folds = np.array_split(shuffled_subjects, k)
    folds = [list(fold) for fold in folds]

    logging.info(f"Run strategy: {run_strategy}")
    logging.info(f"Target dataset: {target_dataset_name}")
    logging.info(f"Results directory: {results_dir}")

    for fold_idx, fold in enumerate(folds):
        subs = "_".join(fold)

        train_datasets = []
        for name, cfg in train_datasets_config.items():
            ratio = cfg["sample_ratio"]
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
                exclude_subjects=cfg["exclude_subjects"],
            )
            train_df = filter_df_by_subjects(train_df, cfg["subjects"])
            train_df = sample_sparse_by_subject(
                train_df,
                ratio,
                seed=random_state + fold_idx,
            )
            if train_df.empty:
                raise ValueError(f"No training samples found for dataset: {name}")
            train_csv = os.path.join(csv_dir, f"train_{name}_{subs}.csv")
            train_df.to_csv(train_csv, index=False)
            train_datasets.append(
                fNIRSPreloadDataset(
                    train_csv,
                    chromo=chromo,
                    aug_name=train_aug_name,
                    aug_params=train_aug_params,
                    seed=random_state + fold_idx,
                )
            )

        train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)

        eval_datasets = []
        for name, cfg in eval_datasets_config.items():
            if name == fold_dataset_name:
                test_subjects_list = fold
            else:
                test_subjects_list = cfg["subjects"]
            _, test_df = create_train_test_segments(
                None,
                cfg["root"],
                test_subjects_list=test_subjects_list,
                exclude_subjects=cfg["exclude_subjects"],
            )
            test_df = filter_df_by_subjects(test_df, cfg["subjects"])
            if test_df.empty:
                raise ValueError(f"No evaluation samples found for dataset: {name}")
            test_csv = os.path.join(csv_dir, f"test_{name}_{subs}.csv")
            test_df.to_csv(test_csv, index=False)
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

        test_dataset = eval_datasets[0] if len(eval_datasets) == 1 else ConcatDataset(eval_datasets)

        dataset_shapes = [tuple(dataset[0][0].shape) for dataset in train_datasets + eval_datasets]
        if len(set(dataset_shapes)) != 1:
            raise ValueError(f"All train/eval datasets must have the same tensor shape; got {dataset_shapes}")
        input_channels = dataset_shapes[0][0]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # Calculate class weights for imbalanced data (optional)
        if use_class_weights:
            class_weights = calculate_class_weights(train_dataset, device)
        else:
            class_weights = None

        # Initialize model, loss, and optimizer
        model = CNN2DImage(input_channels=input_channels).to(device)
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

def main():
    train_params = {
        "num_epochs": 500,
        "learning_rate": 1e-4,
        "batch_size": 16,
        "random_state": 42,
        "chromo": "both",
        "use_class_weights": False,
    }

    run_strategy = "base"  # "base", "channel_density", "imageRecon_params", or "online_eeg_aug"
    config = {
        
        "target_dataset_name": "BS_Laura",
        
        "base": {
            "subset_name": "full",
            "representation": "parcel",
        },
        "channel_density": {
            "source_dataset_name": "BS_Laura",
            "sparse_sample_ratio": 1.0,
            "subset_folders": CHANNEL_DENSITY_SUBSET_FOLDERS,
        },
        "imageRecon_params": {
            "recon_param_aug_sample_ratio": 0.0,
            "default_recon_param_folder": "am_1__as_1",
            "recon_param_folders": RECON_PARAM_FOLDERS,
        },
        "online_eeg_aug": {
            "subset_name": "full",
            "representation": "channel",
            "online_aug_name": "space_shuffle",
            "online_aug_params": ONLINE_AUG_DEFAULT_PARAMS["space_shuffle"],
        },
    }

    run_mixed_training(run_strategy, config, train_params)


if __name__ == "__main__":
    main()
