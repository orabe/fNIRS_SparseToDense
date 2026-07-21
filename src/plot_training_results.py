#!/usr/bin/env python3
import glob
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, roc_curve, auc


CONFIG = {
    "results_root": "results",
    "run_strategy": "online_eeg_aug",  # "base", "imageRecon_params", "channel_density", or "online_eeg_aug"
    "dataset_name": "BallSqueezingHD_modified", # BS_Laura, BallSqueezingHD_modified, vfc_hd, Anderson_sparse
    "subset_name": "full",
    "representation": "channel",  # "parcel" or "channel"
    "image_recon": {
        "sample_ratio": 1.0,
    },
    "online_eeg_aug": {
        "aug_name": "none",
    },
    "channel_density": {
        "source_dataset_name": "none",
        "sample_ratio": 0.0,
    },
}


def subject_label_from_path(path):
    filename = os.path.basename(path)
    stem, _ = os.path.splitext(filename)
    parts = stem.split("_")
    subject_parts = [part for part in parts if part.startswith("sub-")]
    if subject_parts:
        return "_".join(subject_parts)
    return stem


def run_dir_for_current_config():
    if CONFIG["run_strategy"] == "base":
        return os.path.join(
            CONFIG["results_root"],
            "base",
            f"target_{CONFIG['dataset_name']}",
            CONFIG["subset_name"],
            f"{CONFIG['representation']}_space",
        )
    if CONFIG["run_strategy"] == "imageRecon_params":
        return os.path.join(
            CONFIG["results_root"],
            "imageRecon_params",
            f"target_{CONFIG['dataset_name']}",
            "parcel_space",
            f"ratio_{CONFIG['image_recon']['sample_ratio']:.1f}",
        )
    if CONFIG["run_strategy"] == "online_eeg_aug":
        return os.path.join(
            CONFIG["results_root"],
            "online_eeg_aug",
            f"target_{CONFIG['dataset_name']}",
            CONFIG["subset_name"],
            f"{CONFIG['representation']}_space",
            CONFIG["online_eeg_aug"]["aug_name"],
        )
    if CONFIG["run_strategy"] == "channel_density":
        sample_ratio = CONFIG["channel_density"]["sample_ratio"]
        source_name = "none" if sample_ratio == 0 else CONFIG["channel_density"]["source_dataset_name"]
        return os.path.join(
            CONFIG["results_root"],
            "channel_density",
            f"target_{CONFIG['dataset_name']}",
            f"source_{source_name}",
            "parcel_space",
            f"ratio_{sample_ratio:.1f}",
        )
    raise ValueError(f"Unknown run_strategy: {CONFIG['run_strategy']}")


def output_dir_for_current_config():
    return os.path.join(run_dir_for_current_config(), "analysis")


def experiment_label():
    representation = CONFIG["representation"]
    if CONFIG["run_strategy"] in ["imageRecon_params", "channel_density"]:
        representation = "parcel"
    return f"{CONFIG['dataset_name']} | {representation} space | {CONFIG['run_strategy']}"


def plot_metric(all_results, metric_name, train_key, test_key, output_path=None, show=False):
    n_subjects = len(all_results)
    
    cols = 4
    rows = (n_subjects + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4, rows * 3), squeeze=False, sharey=True
    )

    train_color = "tab:blue"
    test_color = "tab:orange"

    for idx, (label, results) in enumerate(all_results):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        train_vals = results.get(train_key, [])
        test_vals = results.get(test_key, [])
        ax.plot(train_vals, label="train", alpha=0.8, color=train_color)
        ax.plot(test_vals, label="test", alpha=0.8, color=test_color)
        ax.set_title(label, fontsize=11, fontweight="bold")
        if r == rows - 1:
            ax.set_xlabel("Epoch")
        else:
            ax.set_xlabel("")
        ax.tick_params(labelsize=8, labelleft=True)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        ax.legend(fontsize=7)

    for idx in range(n_subjects, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    fig.suptitle(
        f"{experiment_label()} | {metric_name.capitalize()}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def compute_f1_vs_threshold(labels, probs, thresholds):
    """
    Compute F1 scores across different classification thresholds.
    
    Args:
        labels: true labels (array)
        probs: predicted probabilities for positive class (array)
        thresholds: array of threshold values to evaluate
    
    Returns:
        Array of F1 scores corresponding to each threshold
    """
    f1_scores = []
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels, preds, average='micro', zero_division=0)
        f1_scores.append(f1)
    return np.array(f1_scores)


def plot_f1_vs_threshold(all_results, output_path, thresholds=None):
    """
    Plot F1 score vs classification threshold for all subjects.
    
    Args:
        all_results: list of (label, results) tuples
        output_path: path to save the plot
        thresholds: array of threshold values (default: 0 to 1 in 0.01 steps)
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.01)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_f1_curves = []
    
    for label, results in all_results:
        # Get final test predictions
        final_labels = results.get('final_test_labels', None)
        final_probs = results.get('final_test_probs', None)
        
        if final_labels is None or final_probs is None:
            print(f"Skipping {label}: missing final predictions")
            continue
        
        # For binary classification, use probability of positive class
        if final_probs.ndim == 2 and final_probs.shape[1] == 2:
            pos_probs = final_probs[:, 1]
        else:
            pos_probs = final_probs
        
        # Compute F1 at each threshold
        f1_scores = compute_f1_vs_threshold(final_labels, pos_probs, thresholds)
        all_f1_curves.append(f1_scores)
        
        # Plot individual subject curve
        ax.plot(thresholds, f1_scores, alpha=0.6, linewidth=1.5, label=label)
    
    if all_f1_curves:
        # Compute and plot mean curve
        mean_f1 = np.mean(all_f1_curves, axis=0)
        std_f1 = np.std(all_f1_curves, axis=0)
        
        # Plot std band
        f1_upper = np.minimum(mean_f1 + std_f1, 1)
        f1_lower = np.maximum(mean_f1 - std_f1, 0)
        ax.fill_between(thresholds, f1_lower, f1_upper, color='grey',
                        alpha=0.2, label='± 1 std. dev.', zorder=99)
        
        ax.plot(thresholds, mean_f1, color='black', linewidth=3, 
                label='Mean', linestyle='--', zorder=100)
        
        # Find optimal threshold (max mean F1)
        optimal_idx = np.argmax(mean_f1)
        optimal_thresh = thresholds[optimal_idx]
        optimal_f1 = mean_f1[optimal_idx]
        std_f1_at_optimal = std_f1[optimal_idx]
        
        ax.axvline(optimal_thresh, color='red', linestyle=':', linewidth=2, 
                   label=f'Optimal threshold: {optimal_thresh:.2f}')
        ax.plot(optimal_thresh, optimal_f1, 'r*', markersize=15, zorder=101)
        
        ax.text(optimal_thresh + 0.02, optimal_f1, 
                f'F1={optimal_f1:.3f}±{std_f1_at_optimal:.3f}', fontsize=10, color='red')
    
    ax.set_xlabel('Classification Threshold', fontsize=12)
    ax.set_ylabel('F1 Score (Micro)', fontsize=12)
    ax.set_title(
        f"{experiment_label()} | F1 Score vs Classification Threshold",
        fontsize=14,
        fontweight='bold',
    )
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Legend outside plot area
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"F1 vs Threshold plot saved to {output_path}")
    if all_f1_curves:
        print(f"Optimal threshold: {optimal_thresh:.3f} (F1={optimal_f1:.3f})")


def plot_roc_curves(all_results, output_path):
    """
    Plot ROC curves for all subjects on one figure.
    
    Args:
        all_results: list of (label, results) tuples
        output_path: path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    all_tprs = []
    all_aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for label, results in all_results:
        # Get final test predictions
        final_labels = results.get('final_test_labels', None)
        final_probs = results.get('final_test_probs', None)
        
        if final_labels is None or final_probs is None:
            print(f"Skipping {label}: missing final predictions")
            continue
        
        # For binary classification, use probability of positive class
        if final_probs.ndim == 2 and final_probs.shape[1] == 2:
            pos_probs = final_probs[:, 1]
        else:
            pos_probs = final_probs
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(final_labels, pos_probs)
        roc_auc = auc(fpr, tpr)
        all_aucs.append(roc_auc)
        
        # Interpolate to common FPR values for averaging
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        all_tprs.append(interp_tpr)
        
        # Plot individual subject curve
        ax.plot(fpr, tpr, alpha=0.4, linewidth=1.5, 
                label=f'{label} (AUC={roc_auc:.3f})')
    
    if all_tprs:
        # Compute mean ROC curve
        mean_tpr = np.mean(all_tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(all_aucs)
        
        # Plot mean curve
        ax.plot(mean_fpr, mean_tpr, color='black', linewidth=3,
                label=f'Mean (AUC={mean_auc:.3f} ± {std_auc:.3f})',
                linestyle='--', zorder=100)
        
        # Plot std band
        std_tpr = np.std(all_tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey',
                        alpha=0.2, label='± 1 std. dev.', zorder=99)
    
    # Plot random classifier baseline
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, 
            alpha=0.7, zorder=98)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(
        f"{experiment_label()} | ROC Curves - All Subjects",
        fontsize=14,
        fontweight='bold',
    )
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
    
    # Legend outside plot area
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"ROC curves plot saved to {output_path}")
    if all_tprs:
        print(f"Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}")


def plot_final_mean_std_bar(all_results, metric_label, train_key, test_key, output_path):
    train_final = []
    test_final = []
    for _, results in all_results:
        train_values = results.get(train_key, [])
        test_values = results.get(test_key, [])
        if train_values:
            train_final.append(train_values[-1])
        if test_values:
            test_final.append(test_values[-1])

    if not train_final or not test_final:
        return

    train_mean = float(sum(train_final) / len(train_final))
    test_mean = float(sum(test_final) / len(test_final))
    train_std = float((sum((x - train_mean) ** 2 for x in train_final) / len(train_final)) ** 0.5)
    test_std = float((sum((x - test_mean) ** 2 for x in test_final) / len(test_final)) ** 0.5)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(
        ["Train", "Test"],
        [train_mean, test_mean],
        yerr=[train_std, test_std],
        capsize=5,
        color=["tab:blue", "tab:orange"],
    )
    ax.text(
        0,
        train_mean / 2,
        f"{train_mean:.3f} ± {train_std:.3f}",
        ha="center",
        va="center",
        rotation=90,
        color="white",
        fontsize=9,
    )
    ax.text(
        1,
        test_mean / 2,
        f"{test_mean:.3f} ± {test_std:.3f}",
        ha="center",
        va="center",
        rotation=90,
        color="white",
        fontsize=9,
    )
    ax.set_title(
        f"{experiment_label()} | Final {metric_label} Mean ± Std",
        fontsize=7,
        fontweight="bold",
    )
    ax.set_ylabel(metric_label)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    run_dir = run_dir_for_current_config()
    output_dir = output_dir_for_current_config()
    os.makedirs(output_dir, exist_ok=True)

    result_files = sorted(glob.glob(os.path.join(run_dir, "metrics", "res_*.pkl")))
    if not result_files:
        raise RuntimeError(f"No metric pickle files found in: {os.path.join(run_dir, 'metrics')}")

    all_results = []
    for path in result_files:
        with open(path, "rb") as handle:
            results = pickle.load(handle)
        label = subject_label_from_path(path)
        all_results.append((label, results))

    metrics = {
        "loss": ("train_loss", "test_loss"),
        "accuracy": ("train_accuracy", "test_accuracy"),
        "f1_micro": ("train_f1_micro", "test_f1_micro"),
        "f1_macro": ("train_f1_macro", "test_f1_macro"),
        "auroc": ("train_auroc", "test_auroc"),
        "precision": ("train_precision", "test_precision"),
        "recall": ("train_recall", "test_recall"),
    }

    for metric_name, (train_key, test_key) in metrics.items():
        output_path = os.path.join(output_dir, f"all_subjects_{metric_name}.png")
        plot_metric(
            all_results,
            metric_name,
            train_key,
            test_key,
            output_path=output_path,
            show=False,
        )

    plot_final_mean_std_bar(
        all_results,
        "F1 Macro",
        "train_f1_macro",
        "test_f1_macro",
        os.path.join(output_dir, "f1_macro_mean_std_bar.png"),
    )
    plot_final_mean_std_bar(
        all_results,
        "AUROC",
        "train_auroc",
        "test_auroc",
        os.path.join(output_dir, "auroc_mean_std_bar.png"),
    )

    # Plot F1 vs threshold
    f1_threshold_path = os.path.join(output_dir, "f1_vs_threshold.png")
    plot_f1_vs_threshold(all_results, f1_threshold_path)

    # Plot ROC curves
    roc_path = os.path.join(output_dir, "roc_curves_all_subjects.png")
    plot_roc_curves(all_results, roc_path)

    print(f"Plots saved in {output_dir}/")


if __name__ == "__main__":
    main()
