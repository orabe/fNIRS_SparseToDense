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
    "output_dir": "results/channel_density/analysis",
    "ratios": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "dataset_pairs": [
        {"target_dataset_name": "BS_Laura", 
         "source_dataset_name": "BS_Laura"},
        
        {"target_dataset_name": "BS_Laura",
         "source_dataset_name": "BallSqueezingHD_modified"},
        
        {"target_dataset_name": "BallSqueezingHD_modified",
         "source_dataset_name": "BallSqueezingHD_modified"},
        
        {"target_dataset_name": "BallSqueezingHD_modified",
         "source_dataset_name": "BS_Laura"},
        
        {"target_dataset_name": "vfc_hd", "source_dataset_name": "vfc_hd"},
        {"target_dataset_name": "vfc_hd", "source_dataset_name": "Anderson_sparse"},
    ],
    "metrics": {
        "F1 Macro": {
            "key": "test_f1_macro",
        },
        "Accuracy": {
            "key": "test_accuracy",
        },
        "AUROC": {
            "key": "test_auroc",
        },
    },
}


DATASET_LABELS = {
    "BallSqueezingHD_modified": "BallSqueezingHD",
    "BS_Laura": "BS_Laura",
    "vfc_hd": "VFC-HD",
    "Anderson_sparse": "Anderson",
}


def dataset_label(dataset_name):
    return DATASET_LABELS.get(dataset_name, dataset_name)


def pair_label(dataset_pair):
    target_name = dataset_label(dataset_pair["target_dataset_name"])
    source_name = dataset_label(dataset_pair["source_dataset_name"])
    return f"Target: {target_name}\nAugmentation Source: {source_name}"


def dataset_pair_groups():
    intra_pairs = []
    inter_pairs = []
    for pair_idx, dataset_pair in enumerate(CONFIG["dataset_pairs"]):
        if dataset_pair["target_dataset_name"] == dataset_pair["source_dataset_name"]:
            intra_pairs.append(pair_idx)
        else:
            inter_pairs.append(pair_idx)
    return [
        ("Intra-dataset", intra_pairs),
        ("Inter-dataset", inter_pairs),
    ]


def gain_color(value):
    if value > 0:
        return "#2CA02C"
    if value < 0:
        return "#D62728"
    return "black"


def text_color_for_background(color):
    red, green, blue, _ = plt.matplotlib.colors.to_rgba(color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.55 else "white"


def run_dir(dataset_pair, ratio):
    source_name = "none" if ratio == 0 else dataset_pair["source_dataset_name"]
    return os.path.join(
        CONFIG["results_root"],
        "channel_density",
        f"target_{dataset_pair['target_dataset_name']}",
        f"source_{source_name}",
        "parcel_space",
        f"ratio_{ratio:.1f}",
    )


def collect_metric_by_subject(dataset_pair, ratio, metric_key):
    metric_dir = os.path.join(run_dir(dataset_pair, ratio), "metrics")
    result_files = sorted(glob.glob(os.path.join(metric_dir, "res_*.pkl")))
    values = {}
    for path in result_files:
        subject = os.path.basename(path).replace("res_", "").replace("_both.pkl", "")
        with open(path, "rb") as handle:
            result = pickle.load(handle)
        metric_values = result.get(metric_key, [])
        if metric_values:
            values[subject] = float(metric_values[-1])
    return values


def summarize_values(values):
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
    for pair_idx, dataset_pair in enumerate(CONFIG["dataset_pairs"]):
        summary[pair_idx] = {}
        for ratio in CONFIG["ratios"]:
            summary[pair_idx][ratio] = {}
            for metric_name, metric_config in CONFIG["metrics"].items():
                by_subject = collect_metric_by_subject(dataset_pair, ratio, metric_config["key"])
                metric_summary = summarize_values(list(by_subject.values()))
                summary[pair_idx][ratio][metric_name] = {
                    "by_subject": by_subject,
                    "summary": metric_summary,
                }
                rows.append(
                    {
                        "target_dataset": dataset_pair["target_dataset_name"],
                        "source_dataset": "none" if ratio == 0 else dataset_pair["source_dataset_name"],
                        "representation": "parcel",
                        "ratio": f"{ratio:.1f}",
                        "metric": metric_name,
                        "mean": "" if metric_summary is None else f"{metric_summary['mean']:.6f}",
                        "std": "" if metric_summary is None else f"{metric_summary['std']:.6f}",
                        "n": "" if metric_summary is None else metric_summary["n"],
                    }
                )
    return summary, rows


def write_summary_csv(rows):
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    output_path = os.path.join(CONFIG["output_dir"], "channel_density_final_summary.csv")
    headers = ["target_dataset", "source_dataset", "representation", "ratio", "metric", "mean", "std", "n"]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def paired_gain_pp(pair_summary, metric_name, ratio):
    baseline = pair_summary[0.0][metric_name]["by_subject"]
    augmented = pair_summary[ratio][metric_name]["by_subject"]
    paired_subjects = sorted(set(baseline) & set(augmented))
    if not paired_subjects:
        return None
    gains = [augmented[subject] - baseline[subject] for subject in paired_subjects]
    return mean(gains) * 100


def set_plot_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def draw_metric_axis(ax, pair_summary, metric_name, show_ylabel, show_xlabel):
    ratios = CONFIG["ratios"]
    bar_width = 0.11
    x_positions = np.arange(len(ratios)) * 0.16
    colors = ["#B8B8B8"] + [
        plt.get_cmap("Blues")(0.35 + 0.5 * idx / max(1, len(ratios) - 2))
        for idx in range(len(ratios) - 1)
    ]
    has_values = False

    for ratio_idx, ratio in enumerate(ratios):
        metric_summary = pair_summary[ratio][metric_name]["summary"]
        x_pos = x_positions[ratio_idx]
        if metric_summary is None:
            ax.bar(
                x_pos,
                0.0,
                width=bar_width,
                color="#f2f2f2",
                edgecolor="#d0d0d0",
                linewidth=0.5,
            )
            ax.text(x_pos, 0.04, "missing", ha="center", va="bottom", rotation=90, fontsize=5, color="#777777")
            continue

        has_values = True
        value = metric_summary["mean"]
        std_value = metric_summary["std"]
        ax.bar(
            x_pos,
            value,
            width=bar_width,
            yerr=std_value,
            capsize=2,
            color=colors[ratio_idx],
            edgecolor="black",
            linewidth=0.55,
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
        )
        ax.text(
            x_pos,
            max(value * 0.5, 0.08),
            f"{value:.2f}±{std_value:.2f}",
            ha="center",
            va="center",
            rotation=90,
            fontsize=5,
            color=text_color_for_background(colors[ratio_idx]),
            fontweight="bold",
        )
        if ratio != 0:
            gain = paired_gain_pp(pair_summary, metric_name, ratio)
            if gain is not None:
                ax.text(
                    x_pos,
                    min(value + std_value + 0.035, 1.08),
                    f"{gain:+.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=gain_color(gain),
                    fontweight="bold",
                )

    if not has_values:
        ax.text(0.5, 0.5, "No completed runs", transform=ax.transAxes, ha="center", va="center", color="#777777")

    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{ratio * 100:g}%" for ratio in ratios], rotation=0)
    if show_ylabel:
        ax.set_ylabel("Parcel-space score")
    if show_xlabel:
        ax.set_xlabel("Sparse subset ratio")
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.25)
    ax.set_axisbelow(True)


def plot_all_pairs(summary, metric_names, output_name, title):
    set_plot_style()
    groups = dataset_pair_groups()
    n_rows = len(metric_names) * len(groups)
    n_cols = max(len(pair_indices) for _, pair_indices in groups)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * n_cols, 2.8 * n_rows + 0.8),
        sharey=True,
        squeeze=False,
    )

    for metric_idx, metric_name in enumerate(metric_names):
        for group_idx, (group_name, pair_indices) in enumerate(groups):
            row_idx = metric_idx * len(groups) + group_idx
            for col_idx in range(n_cols):
                ax = axes[row_idx][col_idx]
                if col_idx >= len(pair_indices):
                    ax.axis("off")
                    continue
                pair_idx = pair_indices[col_idx]
                dataset_pair = CONFIG["dataset_pairs"][pair_idx]
                draw_metric_axis(
                    ax,
                    summary[pair_idx],
                    metric_name,
                    show_ylabel=col_idx == 0,
                    show_xlabel=row_idx == n_rows - 1,
                )
                ax.set_title(pair_label(dataset_pair), fontweight="bold", loc="left")
                if col_idx == 0:
                    ax.text(
                        -0.28,
                        0.5,
                        f"{metric_name}\n{group_name}",
                        transform=ax.transAxes,
                        ha="right",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                    )

    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.965,
        "Numbers above non-baseline bars show paired gain vs 0% baseline in percentage points (pp)",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.tight_layout(rect=[0.05, 0.03, 1.0, 0.94])

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    output_path = os.path.join(CONFIG["output_dir"], output_name)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_single_metric_grid(summary, metric_name, output_name, title):
    set_plot_style()
    groups = dataset_pair_groups()
    n_rows = len(groups)
    n_cols = max(len(pair_indices) for _, pair_indices in groups)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 3.1 * n_rows + 0.8),
        sharey=True,
        squeeze=False,
    )

    for row_idx, (group_name, pair_indices) in enumerate(groups):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx >= len(pair_indices):
                ax.axis("off")
                continue
            pair_idx = pair_indices[col_idx]
            dataset_pair = CONFIG["dataset_pairs"][pair_idx]
            draw_metric_axis(
                ax,
                summary[pair_idx],
                metric_name,
                show_ylabel=col_idx == 0,
                show_xlabel=row_idx == n_rows - 1,
            )
            ax.set_title(pair_label(dataset_pair), fontweight="bold", loc="left")
            if col_idx == 0:
                ax.text(
                    -0.28,
                    0.5,
                    group_name,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.965,
        "Numbers above non-baseline bars show paired gain vs 0% baseline in percentage points (pp)",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.tight_layout(rect=[0.04, 0.03, 1.0, 0.94])

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    output_path = os.path.join(CONFIG["output_dir"], output_name)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    summary, rows = build_summary()
    csv_path = write_summary_csv(rows)
    combined_plot_path = plot_all_pairs(
        summary,
        list(CONFIG["metrics"]),
        "channel_density_all_datasets_all_ratios.png",
        "Channel-Density Augmentation Across Dataset Pairs",
    )
    f1_plot_path = plot_single_metric_grid(
        summary,
        "F1 Macro",
        "channel_density_all_datasets_all_ratios_f1_macro_only.png",
        "Channel-Density Augmentation Across Dataset Pairs (F1 Macro)",
    )
    accuracy_plot_path = plot_single_metric_grid(
        summary,
        "Accuracy",
        "channel_density_all_datasets_all_ratios_accuracy_only.png",
        "Channel-Density Augmentation Across Dataset Pairs (Accuracy)",
    )
    print(f"Saved summary CSV to {csv_path}")
    print(f"Saved combined all-ratios plot to {combined_plot_path}")
    print(f"Saved F1-only all-ratios plot to {f1_plot_path}")
    print(f"Saved accuracy-only all-ratios plot to {accuracy_plot_path}")


if __name__ == "__main__":
    main()
