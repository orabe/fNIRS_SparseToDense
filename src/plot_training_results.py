#!/usr/bin/env python3
import glob
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, roc_curve, auc


def subject_label_from_path(path):
    filename = os.path.basename(path)
    stem, _ = os.path.splitext(filename)
    parts = stem.split("_")
    subject_parts = [part for part in parts if part.startswith("sub-")]
    if subject_parts:
        return "_".join(subject_parts)
    return stem

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

    fig.suptitle(metric_name.capitalize(), fontsize=14, fontweight="bold")
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
    ax.set_title('F1 Score vs Classification Threshold', fontsize=14, fontweight='bold')
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
    ax.set_title('ROC Curves - All Subjects', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
    
    # Legend outside plot area
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"ROC curves plot saved to {output_path}")
    if all_tprs:
        print(f"Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}")


def main():
    # Old channel-density augmentation results:
    # DATASET_NAME = "train_full__eval_laura_full_1.0"
    # DATASET_NAME = "train_full+motor_100chs+motor_91chs+motor_80chs+motor_70chs+motor_59chs+motor_50chs__eval_laura_full_0.1"
    # DATASET_NAME = "train_full+motor_100chs+motor_91chs+motor_80chs+motor_70chs+motor_59chs+motor_50chs__eval_laura_full_0.3"
    # DATASET_NAME = "train_full+motor_100chs+motor_91chs+motor_80chs+motor_70chs+motor_59chs+motor_50chs__eval_laura_full_0.5"
    # DATASET_NAME = "train_full+motor_100chs+motor_91chs+motor_80chs+motor_70chs+motor_59chs+motor_50chs__eval_laura_full_0.7"
    # DATASET_NAME = "train_full+motor_100chs+motor_91chs+motor_80chs+motor_70chs+motor_59chs+motor_50chs__eval_laura_full_1.0"

    # New image-reconstruction parameter augmentation results.
    # dataset_name = "BallSqueezingHD_modified"
    # dataset_name = "BS_Laura"
    dataset_name = "vfc_hd"
    
    chromo_mode = "both"
    augmentation_strategy = "imageRecon_params" #"imageRecon_params" or "channelDensity_aug"
    
    # Must match the training folder convention: 0.0, 0.1, 0.3, 0.5, 0.7, 1.0.
    recon_param_aug_sample_ratio = 1.0
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
    active_views = [name for name, ratio in recon_param_sample_ratios.items() if ratio > 0]
    ratio_tag = f"{recon_param_aug_sample_ratio:.1f}"
    result_group = f"{augmentation_strategy}__{dataset_name}"
    DATASET_NAME = f"train_{dataset_name}_imgRecon_{len(active_views)}am-as_ratio_{chromo_mode}_{ratio_tag}"

    result_patterns = [
        # "results/*/res_*.pkl",
        f"results/{result_group}/{DATASET_NAME}/res_*.pkl",
    ]
    output_dir = f"figures/{result_group}/{DATASET_NAME}"
    os.makedirs(output_dir, exist_ok=True)

    result_files = []
    for pattern in result_patterns:
        result_files.extend(glob.glob(pattern))

    all_results = []
    for path in sorted(result_files):
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

    train_f1_final = []
    test_f1_final = []
    for _, results in all_results:
        train_f1 = results.get("train_f1_micro", [])
        test_f1 = results.get("test_f1_micro", [])
        if train_f1:
            train_f1_final.append(train_f1[-1])
        if test_f1:
            test_f1_final.append(test_f1[-1])

    if train_f1_final and test_f1_final:
        train_mean = float(sum(train_f1_final) / len(train_f1_final))
        test_mean = float(sum(test_f1_final) / len(test_f1_final))

        train_std = float((sum((x - train_mean) ** 2 for x in train_f1_final) / len(train_f1_final)) ** 0.5)
        test_std = float((sum((x - test_mean) ** 2 for x in test_f1_final) / len(test_f1_final)) ** 0.5)

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
        ax.set_title("Final F1 Micro Mean ± Std", fontsize=13, fontweight="bold")
        ax.set_ylabel("F1 Micro")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "f1_mean_std_bar.png"), dpi=300)
        plt.close(fig)

    # Plot F1 vs threshold
    f1_threshold_path = os.path.join(output_dir, "f1_vs_threshold.png")
    plot_f1_vs_threshold(all_results, f1_threshold_path)

    # Plot ROC curves
    roc_path = os.path.join(output_dir, "roc_curves_all_subjects.png")
    plot_roc_curves(all_results, roc_path)

    print(f"Plots saved in {output_dir}/")


if __name__ == "__main__":
    main()
