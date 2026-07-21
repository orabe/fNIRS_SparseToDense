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
    "output_dir": "results/online_eeg_aug/analysis",
    "subset_name": "full",
    "datasets": [
        "BallSqueezingHD_modified",
        "BS_Laura",
        "vfc_hd",
        "Anderson_sparse",
    ],
    "representations": ["channel", "parcel"],
    "augmentations": [
        "none",
        "gaussian_noise",
        "smooth_time_mask",
        "time_reverse",
        "sign_flip",
        "ft_surrogate",
        "frequency_shift",
        "bandstop_filter",
        "space_symmetry",
        "space_dropout",
        "space_shuffle",
    ],
    "metrics": {
        "F1 Macro": {
            "key": "test_f1_macro",
            "output_name": "f1_macro",
        },
        "AUROC": {
            "key": "test_auroc",
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

AUGMENTATION_LABELS = {
    "none": "None",
    "gaussian_noise": "Gaussian\nnoise",
    "smooth_time_mask": "Smooth\ntime mask",
    "time_reverse": "Time\nreverse",
    "sign_flip": "Sign\nflip",
    "ft_surrogate": "FT\nsurrogate",
    "frequency_shift": "Frequency\nshift",
    "bandstop_filter": "Bandstop\nfilter",
    "space_symmetry": "Space\nsymmetry",
    "space_dropout": "Space\ndropout",
    "space_shuffle": "Space\nshuffle",
}

REPRESENTATION_LABELS = {
    "channel": "Channel",
    "parcel": "Parcel",
}


def run_dir(dataset_name, representation, augmentation):
    return os.path.join(
        CONFIG["results_root"],
        "online_eeg_aug",
        f"target_{dataset_name}",
        CONFIG["subset_name"],
        f"{representation}_space",
        augmentation,
    )


def collect_final_metric(dataset_name, representation, augmentation, metric_key):
    metric_dir = os.path.join(run_dir(dataset_name, representation, augmentation), "metrics")
    result_files = sorted(glob.glob(os.path.join(metric_dir, "res_*.pkl")))
    if not result_files:
        return None

    values = []
    for path in result_files:
        with open(path, "rb") as handle:
            result = pickle.load(handle)
        metric_values = result.get(metric_key, [])
        if metric_values:
            values.append(float(metric_values[-1]))

    if not values:
        return None

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
            for augmentation in CONFIG["augmentations"]:
                summary[dataset_name][representation][augmentation] = {}
                for metric_name, metric_config in CONFIG["metrics"].items():
                    metric_summary = collect_final_metric(
                        dataset_name,
                        representation,
                        augmentation,
                        metric_config["key"],
                    )
                    summary[dataset_name][representation][augmentation][metric_name] = metric_summary
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "representation": representation,
                            "augmentation": augmentation,
                            "metric": metric_name,
                            "mean": "" if metric_summary is None else f"{metric_summary['mean']:.6f}",
                            "std": "" if metric_summary is None else f"{metric_summary['std']:.6f}",
                            "n": "" if metric_summary is None else metric_summary["n"],
                        }
                    )
    return summary, rows


def write_summary_csv(rows):
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    output_path = os.path.join(CONFIG["output_dir"], "online_eeg_aug_final_summary.csv")
    headers = ["dataset", "representation", "augmentation", "metric", "mean", "std", "n"]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def best_augmentation_for_metric(summary, dataset_name, representation, metric_name):
    best_name = None
    best_summary = None
    for augmentation in CONFIG["augmentations"]:
        if augmentation == "none":
            continue
        metric_summary = summary[dataset_name][representation][augmentation][metric_name]
        if metric_summary is None:
            continue
        if best_summary is None or metric_summary["mean"] > best_summary["mean"]:
            best_name = augmentation
            best_summary = metric_summary
    return best_name, best_summary


def plot_best_methods(summary, metric_names, output_name, title):
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    output_path = os.path.join(CONFIG["output_dir"], output_name)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    dataset_names = CONFIG["datasets"]
    representations = CONFIG["representations"]
    fig, axes = plt.subplots(
        len(representations),
        len(dataset_names),
        figsize=(13.5, 5.6) if len(metric_names) > 1 else (13.5, 6.6),
        squeeze=False,
        sharey=True,
    )

    group_centers = np.arange(len(metric_names))
    bar_width = 0.34 if len(metric_names) > 1 else 0.10
    split_offsets = {"none": -bar_width / 2, "best": bar_width / 2}
    bar_colors = {"none": "#9E9E9E", "best": "#4C78A8"}

    for row_idx, representation in enumerate(representations):
        for col_idx, dataset_name in enumerate(dataset_names):
            ax = axes[row_idx][col_idx]
            has_any_metric = False
            for metric_name in metric_names:
                none_summary = summary[dataset_name][representation]["none"][metric_name]
                best_augmentation, best_summary = best_augmentation_for_metric(
                    summary,
                    dataset_name,
                    representation,
                    metric_name,
                )
                group_idx = metric_names.index(metric_name)
                bar_items = [
                    ("none", "None", none_summary),
                    (
                        "best",
                        "missing" if best_augmentation is None else AUGMENTATION_LABELS[best_augmentation].replace("\n", " "),
                        best_summary,
                    ),
                ]

                for bar_kind, method_label, metric_summary in bar_items:
                    x_pos = group_centers[group_idx] + split_offsets[bar_kind]
                    if metric_summary is None:
                        ax.bar(
                            x_pos,
                            0.0,
                            width=bar_width,
                            color="#f2f2f2",
                            edgecolor="#d0d0d0",
                            linewidth=0.6,
                        )
                        ax.text(
                            x_pos,
                            0.04,
                            "missing",
                            ha="center",
                            va="bottom",
                            rotation=90,
                            fontsize=5,
                            color="#777777",
                        )
                        continue

                    has_any_metric = True
                    value = metric_summary["mean"]
                    std_value = metric_summary["std"]
                    ax.bar(
                        x_pos,
                        value,
                        width=bar_width,
                        yerr=std_value,
                        capsize=3,
                        color=bar_colors[bar_kind],
                        edgecolor="black",
                        linewidth=0.7,
                        error_kw={"elinewidth": 0.9, "capthick": 0.9},
                    )
                    ax.text(
                        x_pos,
                        max(value * 0.5, 0.08),
                        f"{value:.2f}±{std_value:.2f}",
                        ha="center",
                        va="center",
                        rotation=90,
                        fontsize=6,
                        color="white",
                        fontweight="bold",
                    )
                    ax.text(
                        x_pos,
                        0.02,
                        method_label,
                        ha="center",
                        va="bottom",
                        fontsize=5,
                        color="white" if value > 0.16 else "#555555",
                    )

                if none_summary is not None and best_summary is not None:
                    improvement = (best_summary["mean"] - none_summary["mean"]) * 100
                    best_x = group_centers[group_idx] + split_offsets["best"]
                    best_y = min(best_summary["mean"] + best_summary["std"] + 0.035, 1.03)
                    ax.text(
                        best_x,
                        best_y,
                        f"{improvement:+.1f} pp",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#D62728" if improvement >= 0 else "#555555",
                        fontweight="bold",
                    )

            if not has_any_metric:
                ax.text(
                    0.5,
                    0.5,
                    "No completed runs",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#777777",
                )

            if row_idx == 0:
                ax.set_title(DATASET_LABELS.get(dataset_name, dataset_name), fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{REPRESENTATION_LABELS[representation]}\nScore")
            ax.set_ylim(0, 1.05)
            ax.set_xticks(group_centers)
            ax.set_xticklabels(metric_names)
            ax.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.25)
            ax.set_axisbelow(True)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=bar_colors["none"], ec="black", linewidth=0.7),
        plt.Rectangle((0, 0), 1, 1, color=bar_colors["best"], ec="black", linewidth=0.7),
        plt.Line2D([], [], linestyle="none", label="PP = percentage points"),
    ]
    fig.legend(
        handles,
        ["No augmentation", "Best augmentation", "Improvement above bars; PP = percentage points"],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.55, 0.975),
    )

    fig.suptitle(
        title,
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.94], h_pad=1.0, w_pad=0.8)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    summary, rows = build_summary()
    csv_path = write_summary_csv(rows)
    combined_plot_path = plot_best_methods(
        summary,
        list(CONFIG["metrics"]),
        "online_eeg_aug_best_methods.png",
        "Online EEG-Inspired Augmentations: None vs Best Method",
    )
    f1_plot_path = plot_best_methods(
        summary,
        ["F1 Macro"],
        "online_eeg_aug_best_methods_f1_macro_only.png",
        "Online EEG-Inspired Augmentations: None vs Best Method (F1 Macro)",
    )
    print(f"Saved summary CSV to {csv_path}")
    print(f"Saved combined best-method plot to {combined_plot_path}")
    print(f"Saved F1-only best-method plot to {f1_plot_path}")


if __name__ == "__main__":
    main()
