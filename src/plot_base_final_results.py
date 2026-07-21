#!/usr/bin/env python3
import csv
import glob
import os
import pickle
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np


CONFIG = {
    "results_root": "results",
    "output_dir": "results/base/analysis",
    "subset_name": "full",
    "datasets": [
        "BallSqueezingHD_modified",
        "BS_Laura",
        "vfc_hd",
        "Anderson_sparse",
    ],
    "representations": ["channel", "parcel"],
    "metrics": {
        "F1 Macro": {
            "train_key": "train_f1_macro",
            "test_key": "test_f1_macro",
            "output_name": "f1_macro",
        },
        "AUROC": {
            "train_key": "train_auroc",
            "test_key": "test_auroc",
            "output_name": "auroc",
        },
    },
}


DATASET_LABELS = {
    "BallSqueezingHD_modified": "BallSqueezingHD",
    "BS_Laura": "BS_Laura",
    "vfc_hd": "VFC-HD",
    "Anderson_sparse": "Anderson",
}

REPRESENTATION_LABELS = {
    "channel": "Channel",
    "parcel": "Parcel",
}

SPLIT_LABELS = {
    "train": "Train",
    "test": "Test",
}

SPLIT_COLORS = {
    "train": "#4C78A8",
    "test": "#F58518",
}


def run_dir(dataset_name, representation):
    return os.path.join(
        CONFIG["results_root"],
        "base",
        f"target_{dataset_name}",
        CONFIG["subset_name"],
        f"{representation}_space",
    )


def collect_final_values(dataset_name, representation, metric_config):
    metric_dir = os.path.join(run_dir(dataset_name, representation), "metrics")
    result_files = sorted(glob.glob(os.path.join(metric_dir, "res_*.pkl")))
    if not result_files:
        raise RuntimeError(f"No metric files found in: {metric_dir}")

    values = {"train": [], "test": []}
    for path in result_files:
        with open(path, "rb") as handle:
            result = pickle.load(handle)

        train_values = result[metric_config["train_key"]]
        test_values = result[metric_config["test_key"]]
        if not train_values or not test_values:
            raise RuntimeError(f"Empty metric series in: {path}")

        values["train"].append(float(train_values[-1]))
        values["test"].append(float(test_values[-1]))

    return values


def summarize_values(values):
    return {
        "mean": mean(values),
        "std": pstdev(values),
        "n": len(values),
    }


def build_summary():
    summary = {}
    rows = []
    for dataset_name in CONFIG["datasets"]:
        summary[dataset_name] = {}
        for representation in CONFIG["representations"]:
            summary[dataset_name][representation] = {}
            for metric_name, metric_config in CONFIG["metrics"].items():
                values = collect_final_values(dataset_name, representation, metric_config)
                train_summary = summarize_values(values["train"])
                test_summary = summarize_values(values["test"])
                summary[dataset_name][representation][metric_name] = {
                    "train": train_summary,
                    "test": test_summary,
                }
                for split_name, split_summary in [
                    ("train", train_summary),
                    ("test", test_summary),
                ]:
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "representation": representation,
                            "metric": metric_name,
                            "split": split_name,
                            "mean": f"{split_summary['mean']:.6f}",
                            "std": f"{split_summary['std']:.6f}",
                            "n": split_summary["n"],
                        }
                    )
    return summary, rows


def write_summary_csv(rows):
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    output_path = os.path.join(CONFIG["output_dir"], "base_final_channel_vs_parcel_summary.csv")
    headers = ["dataset", "representation", "metric", "split", "mean", "std", "n"]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def plot_summary(summary):
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    png_path = os.path.join(CONFIG["output_dir"], "base_final_channel_vs_parcel.png")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    dataset_names = CONFIG["datasets"]
    metric_names = list(CONFIG["metrics"])
    fig, axes = plt.subplots(
        len(dataset_names),
        len(metric_names),
        figsize=(7.2, 8.6),
        squeeze=False,
        sharey="col",
    )

    x_positions = np.arange(len(CONFIG["representations"]))
    bar_width = 0.32
    split_offsets = {"train": -bar_width / 2, "test": bar_width / 2}

    for row_idx, dataset_name in enumerate(dataset_names):
        for col_idx, metric_name in enumerate(metric_names):
            ax = axes[row_idx][col_idx]
            for split_name in ["train", "test"]:
                means = [
                    summary[dataset_name][representation][metric_name][split_name]["mean"]
                    for representation in CONFIG["representations"]
                ]
                stds = [
                    summary[dataset_name][representation][metric_name][split_name]["std"]
                    for representation in CONFIG["representations"]
                ]
                positions = x_positions + split_offsets[split_name]
                ax.bar(
                    positions,
                    means,
                    width=bar_width,
                    yerr=stds,
                    capsize=3,
                    error_kw={"elinewidth": 1.0, "capthick": 1.0},
                    label=SPLIT_LABELS[split_name],
                    color=SPLIT_COLORS[split_name],
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.95,
                )
                for position, metric_mean, metric_std in zip(positions, means, stds):
                    ax.text(
                        position,
                        min(metric_mean + metric_std + 0.025, 1.02),
                        f"{metric_mean:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

            if row_idx == 0:
                ax.set_title(metric_name, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.text(
                    -0.34,
                    0.5,
                    DATASET_LABELS.get(dataset_name, dataset_name),
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=10,
                    fontweight="bold",
                )
            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                [REPRESENTATION_LABELS[name] for name in CONFIG["representations"]]
            )
            ax.set_ylim(0, 1.05)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.set_ylabel("Score" if col_idx == 0 else "")
            ax.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.25)
            ax.set_axisbelow(True)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.54, 0.995),
    )
    fig.suptitle("Base Experiment: Final Train/Test Performance", y=1.025, fontsize=12, fontweight="bold")
    fig.supxlabel("Input representation", y=0.025, fontsize=9)
    fig.tight_layout(rect=[0.08, 0.04, 1.0, 0.965], h_pad=1.0, w_pad=1.2)
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main():
    summary, rows = build_summary()
    csv_path = write_summary_csv(rows)
    png_path = plot_summary(summary)
    print(f"Saved summary CSV to {csv_path}")
    print(f"Saved PNG plot to {png_path}")


if __name__ == "__main__":
    main()
