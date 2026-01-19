#!/usr/bin/env python3
import glob
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from datasets_v02 import fNIRSPreloadDataset
from model import CNN2DImage


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalDataset:
    name: str
    root: str
    subjects: list[str]


def label_from_filename(path: str) -> int:
    labels = {"Left": 1, "Right": 0, "left": 1, "right": 0}
    base = os.path.basename(path)
    lower = base.lower()
    if "_left_" in lower:
        return 1
    if "_right_" in lower:
        return 0
    parts = base.split("_")
    key = parts[-3] if base.endswith("_test.nc") else parts[-2]
    if key not in labels:
        raise KeyError(f"Could not infer label from filename: {base}")
    return labels[key]


def collect_files(root: str, subjects: list[str]) -> list[str]:
    files = []
    for subject in subjects:
        files += glob.glob(os.path.join(root, subject, "**", "*.nc"), recursive=True)
    test_files = [f for f in files if os.path.basename(f).endswith("_test.nc")]
    return test_files if test_files else files


def build_test_csv(root: str, subjects: list[str], out_dir: str, tag: str) -> str:
    files = collect_files(root, subjects)
    if not files:
        raise ValueError(f"No .nc files found under {root} for subjects {subjects}")
    labels = [label_from_filename(f) for f in files]
    df = pd.DataFrame({"snirf_file": files, "trial_type": labels})
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"test_segments_{tag}.csv")
    df.to_csv(out_path, index=False)
    return out_path


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    all_preds = []
    all_labels = []
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    accuracy = correct / max(total, 1)
    f1 = f1_score(all_labels, all_preds, average="micro") if total else 0.0
    return total_loss / max(len(loader), 1), accuracy, f1


def main():
    # Model / training config.
    model_dir = "results/fullParcel_BallSqueezingHD_modified/checkpoints"
    chromo = "yuanyuan_v2"
    batch_size = 16

    # Dataset roots (update if your paths differ).
    datasets = [
        EvalDataset(
            name="BallSqueezing_modified_dense",
            root="datasets/full_processed/BallSqueezingHD_modified",
            subjects=[
                "sub-170", "sub-173", "sub-171", "sub-174",
                "sub-176", "sub-179", "sub-182", "sub-177",
                "sub-181", "sub-183", "sub-184", "sub-185",
            ],
        ),
        EvalDataset(
            name="BallSqueezing_modified_sparseSim",
            root="datasets/subset_2_processed/BallSqueezingHD_modified",
            subjects=[
                "sub-170", "sub-173", "sub-171", "sub-174",
                "sub-176", "sub-179", "sub-182", "sub-177",
                "sub-181", "sub-183", "sub-184", "sub-185",
            ],
        ),
        EvalDataset(
            name="FreshMotor_sparse",
            root="datasets/full_processed/FreshMotor",
            subjects=[
                "sub-01", "sub-02", "sub-03", "sub-04", "sub-05",
                "sub-06", "sub-07", "sub-08", "sub-09", "sub-10",
            ],
        ),
        EvalDataset(
            name="FreshMotor_denseGen",
            root="src/subset/vae_results/subset_2_train_all_Conv2D_VAE/parcel_vae_inference/freshmotor",
            subjects=[
                "sub-01", "sub-02", "sub-03", "sub-04", "sub-05",
                "sub-06", "sub-07", "sub-08", "sub-09", "sub-10",
            ],
        ),
    ]

    save_dir = "src/subset/vae_results/parcel_classifier_eval"
    plot_dir = os.path.join(save_dir, "plots")
    csv_dir = os.path.join(save_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(save_dir, "eval.log")),
            logging.StreamHandler(),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LOSO folds match BallSqueezing_modified_dense subjects (one subject per fold).
    folds = [[s] for s in datasets[0].subjects]

    rows = []
    for fold in folds:
        fold_name = "_".join(fold)
        model_path = os.path.join(model_dir, f"model_{fold_name}_{chromo}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model: {model_path}")
        model = CNN2DImage().to(device)
        # Build the dynamic classifier head before loading weights.
        sample_csv = build_test_csv(datasets[0].root, fold, csv_dir, f"shape_{fold_name}")
        sample_ds = fNIRSPreloadDataset(sample_csv, mode="test", chromo="HbO")
        sample_x, _ = sample_ds[0]
        _ = model(sample_x.unsqueeze(0).to(device))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        for dataset in datasets:
            # Use LOSO fold subjects when evaluating BallSqueezing variants.
            if "BallSqueezing" in dataset.name:
                subjects = fold
                test_csv = build_test_csv(dataset.root, subjects, csv_dir, f"{dataset.name}_{fold_name}")
                test_ds = fNIRSPreloadDataset(test_csv, mode="test", chromo="HbO")
                test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
                loss, acc, f1 = evaluate_model(model, test_loader, device)
                LOGGER.info(
                    "[%s][%s] loss=%.4f acc=%.4f f1=%.4f",
                    fold_name,
                    dataset.name,
                    loss,
                    acc,
                    f1,
                )
                rows.append(
                    {
                        "fold": fold_name,
                        "dataset": dataset.name,
                        "subject": fold_name,
                        "loss": loss,
                        "accuracy": acc,
                        "f1": f1,
                    }
                )
            else:
                for subject in dataset.subjects:
                    test_csv = build_test_csv(
                        dataset.root, [subject], csv_dir, f"{dataset.name}_{fold_name}_{subject}"
                    )
                    test_ds = fNIRSPreloadDataset(test_csv, mode="test", chromo="HbO")
                    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
                    loss, acc, f1 = evaluate_model(model, test_loader, device)
                    LOGGER.info(
                        "[%s][%s][%s] loss=%.4f acc=%.4f f1=%.4f",
                        fold_name,
                        dataset.name,
                        subject,
                        loss,
                        acc,
                        f1,
                    )
                    rows.append(
                        {
                            "fold": fold_name,
                            "dataset": dataset.name,
                            "subject": subject,
                            "loss": loss,
                            "accuracy": acc,
                            "f1": f1,
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "metrics_by_fold.csv"), index=False)
    summary = df.groupby("dataset")[["loss", "accuracy", "f1"]].mean().reset_index()
    summary.to_csv(os.path.join(save_dir, "metrics_summary.csv"), index=False)
    f1_stats = df.groupby("dataset")["f1"].agg(["mean", "std"]).reset_index()

    # Bar chart summary across folds per dataset (F1 only).
    f1_sorted = f1_stats.set_index("dataset").loc[[d.name for d in datasets]]
    x = np.arange(len(f1_sorted.index))
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(
        x,
        f1_sorted["mean"],
        yerr=f1_sorted["std"],
        capsize=6,
        label="f1",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(f1_sorted.index, rotation=20, ha="right")
    ax.set_ylabel("f1")
    ax.set_title("Classifier Summary (F1 mean ± std)")
    for idx, bar in enumerate(bars):
        mean = f1_sorted["mean"].iloc[idx]
        std = f1_sorted["std"].iloc[idx]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean:.3f}\n±{std:.3f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "summary_f1.png"), dpi=150)
    plt.close(fig)

    # Line plots per fold for each dataset.
    fold_order = [f[0] for f in folds]
    df_fold = df.groupby(["dataset", "fold"])[["loss", "f1"]].mean().reset_index()
    for metric in ["f1", "loss"]:
        fig, ax = plt.subplots(figsize=(10, 4))
        for dataset in datasets:
            if "FreshMotor" in dataset.name:
                continue
            subset = df_fold[df_fold["dataset"] == dataset.name].set_index("fold").reindex(fold_order)
            ax.plot(fold_order, subset[metric], marker="o", label=dataset.name)
        ax.set_xlabel("fold")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by fold")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{metric}_by_fold.png"), dpi=150)
        plt.close(fig)

    # FreshMotor per-subject boxplots (per dataset).
    fm_names = ["FreshMotor_sparse", "FreshMotor_denseGen"]
    for metric in ["f1", "loss"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        data = []
        labels = []
        for name in fm_names:
            values = df[df["dataset"] == name][metric].values
            data.append(values)
            labels.append(name)
        ax.boxplot(data, labels=labels)
        ax.set_ylabel(metric)
        ax.set_title(f"FreshMotor {metric} per subject (all folds)")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"freshmotor_{metric}_boxplot.png"), dpi=150)
        plt.close(fig)

    # FreshMotor per-subject mean across folds (F1 only).
    subject_order = next(d.subjects for d in datasets if d.name == fm_names[0])
    fig, ax = plt.subplots(figsize=(10, 4))
    for name in fm_names:
        fm_df = df[df["dataset"] == name]
        fm_subject = fm_df.groupby("subject")["f1"].mean().reindex(subject_order)
        ax.plot(fm_subject.index, fm_subject.values, marker="o", label=name)
    ax.set_xlabel("subject")
    ax.set_ylabel("f1")
    ax.set_title("FreshMotor F1 by subject (mean over folds)")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "freshmotor_f1_by_subject.png"), dpi=150)
    plt.close(fig)

    print("Evaluation complete. Results are saved to:", save_dir)

if __name__ == "__main__":
    main()
