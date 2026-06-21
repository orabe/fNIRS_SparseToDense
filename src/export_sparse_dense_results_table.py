#!/usr/bin/env python3
import csv
import glob
import os
import pickle
import re
from statistics import mean, pstdev

from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import ttest_rel


RESULTS_ROOT = "results"
EXPORT_STRATEGY = "img_recon"  # "img_recon" or "sparse_dense"

DATASET_NAME = "vfc_hd"
DATASET_NAME = "BS_Laura"
DATASET_NAME = "BallSqueezingHD_modified"

CHROMO_MODE = "both"
IMG_RECON_RESULT_GROUP = f"imageRecon_params__{DATASET_NAME}"
IMG_RECON_RESULTS_ROOT = os.path.join(RESULTS_ROOT, IMG_RECON_RESULT_GROUP)
IMG_RECON_EXPORT_DIR = os.path.join(
    IMG_RECON_RESULTS_ROOT,
    f"{DATASET_NAME}_imgrecon_{CHROMO_MODE}_analysis",
)

SPARSE_DENSE_DATASET_CONFIGS = [
    ("train_full__eval_laura_full_1.0", "0%", "100%"),
    (
        "train_full+motor_100chs+motor_91chs+motor_80chs+motor_70chs+motor_59chs+motor_50chs__eval_laura_full_0.1",
        "10%",
        "100%",
    ),
    (
        "train_full+motor_100chs+motor_91chs+motor_80chs+motor_70chs+motor_59chs+motor_50chs__eval_laura_full_0.3",
        "30%",
        "100%",
    ),
    (
        "train_full+motor_100chs+motor_91chs+motor_80chs+motor_70chs+motor_59chs+motor_50chs__eval_laura_full_0.5",
        "50%",
        "100%",
    ),
    (
        "train_full+motor_100chs+motor_91chs+motor_80chs+motor_70chs+motor_59chs+motor_50chs__eval_laura_full_0.7",
        "70%",
        "100%",
    ),
    (
        "train_full+motor_100chs+motor_91chs+motor_80chs+motor_70chs+motor_59chs+motor_50chs__eval_laura_full_1.0",
        "100%",
        "100%",
    ),
]

# The baseline am_1__as_1 view is always included; ratio controls the other 8 views.
IMG_RECON_AUG_RATIOS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]


def result_name_for_img_recon_ratio(ratio):
    active_views = 9 if ratio > 0 else 1
    ratio_tag = f"{ratio:.1f}"
    return f"train_{DATASET_NAME}_imgRecon_{active_views}am-as_ratio_{CHROMO_MODE}_{ratio_tag}"


def format_ratio_label(ratio):
    return f"{ratio * 100:g}%"


def collect_final_metric_values(results_root, dataset_name, metric_key):
    pattern = os.path.join(results_root, dataset_name, "res_*.pkl")
    values = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "rb") as handle:
            result = pickle.load(handle)
        series = result.get(metric_key, [])
        if series:
            values.append(float(series[-1]))
    return values


def collect_final_metric_by_subject(results_root, dataset_name, metric_key):
    pattern = os.path.join(results_root, dataset_name, "res_*.pkl")
    values_by_subject = {}
    for path in sorted(glob.glob(pattern)):
        match = re.search(r"(sub-[^_]+)", os.path.basename(path))
        if not match:
            print(f"Skipping result with unknown subject id: {path}")
            continue

        with open(path, "rb") as handle:
            result = pickle.load(handle)

        series = result.get(metric_key, [])
        if series:
            values_by_subject[match.group(1)] = float(series[-1])

    return values_by_subject


def format_mean_std(values):
    metric_mean = mean(values)
    metric_std = pstdev(values)
    return metric_mean, metric_std, f"{metric_mean:.3f} ± {metric_std:.3f}"


def write_csv(path, rows, headers):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, headers):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def write_subject_gain_heatmap(path, subject_rows, gain_column="Subject Gain", metric_label="F1"):
    if not subject_rows:
        return False

    try:
        import matplotlib
    except ModuleNotFoundError:
        print("Skipping heatmap: matplotlib is not installed in this Python environment.")
        return False

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    gains = {}
    ratios = []
    subjects = []
    for row in subject_rows:
        ratio = row["Image-Recon Aug Ratio"]
        subject = row["Subject"]
        gain = float(row[gain_column].split()[0])
        gains[(ratio, subject)] = gain
        if ratio not in ratios:
            ratios.append(ratio)
        if subject not in subjects:
            subjects.append(subject)

    subject_matrix = np.full((len(ratios), len(subjects)), np.nan)
    for row_idx, ratio in enumerate(ratios):
        for col_idx, subject in enumerate(subjects):
            if (ratio, subject) in gains:
                subject_matrix[row_idx, col_idx] = gains[(ratio, subject)]

    if len(subjects) >= 2 and len(ratios) >= 2:
        cluster_matrix = np.nan_to_num(subject_matrix, nan=0.0)
        subject_order = leaves_list(linkage(cluster_matrix.T, method="average"))
        subjects = [subjects[idx] for idx in subject_order]

    columns = subjects + ["mean"]
    heatmap = np.full((len(ratios), len(columns)), np.nan)
    for row_idx, ratio in enumerate(ratios):
        row_values = []
        for col_idx, subject in enumerate(subjects):
            if (ratio, subject) in gains:
                gain = gains[(ratio, subject)]
                heatmap[row_idx, col_idx] = gain
                row_values.append(gain)
        if row_values:
            heatmap[row_idx, len(columns) - 1] = mean(row_values)

    max_abs_gain = np.nanmax(np.abs(heatmap))
    if not np.isfinite(max_abs_gain) or max_abs_gain == 0:
        max_abs_gain = 1.0

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(columns) * 0.75), max(3, len(ratios) * 0.7)))
    cmap = plt.get_cmap("bwr_r")
    norm = plt.Normalize(vmin=-max_abs_gain, vmax=max_abs_gain)
    im = ax.imshow(heatmap, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(ratios)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(ratios)
    ax.set_xlabel("held-out subject")
    ax.set_ylabel("image-recon augmentation ratio")
    ax.set_title(f"{DATASET_NAME}: subject-wise {metric_label} gain vs baseline")

    for row_idx in range(len(ratios)):
        for col_idx in range(len(columns)):
            value = heatmap[row_idx, col_idx]
            if np.isfinite(value):
                r, g, b, _ = cmap(norm(value))
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "black" if luminance > 0.6 else "white"
                label = f"{value:+.3f}" if col_idx == len(columns) - 1 else f"{value:+.2f}"
                ax.text(
                    col_idx,
                    row_idx,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

    ax.axvline(len(subjects) - 0.5, color="black", linewidth=1.5)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{metric_label} gain")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def write_subject_performance_heatmap(path, performance_rows, score_column="Test F1", metric_label="F1"):
    if not performance_rows:
        return False

    try:
        import matplotlib
    except ModuleNotFoundError:
        print("Skipping performance heatmap: matplotlib is not installed in this Python environment.")
        return False

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    scores = {}
    ratios = []
    subjects = []
    for row in performance_rows:
        ratio = row["Image-Recon Aug Ratio"]
        subject = row["Subject"]
        score = float(row[score_column])
        scores[(ratio, subject)] = score
        if ratio not in ratios:
            ratios.append(ratio)
        if subject not in subjects:
            subjects.append(subject)

    subject_matrix = np.full((len(ratios), len(subjects)), np.nan)
    for row_idx, ratio in enumerate(ratios):
        for col_idx, subject in enumerate(subjects):
            if (ratio, subject) in scores:
                subject_matrix[row_idx, col_idx] = scores[(ratio, subject)]

    if len(subjects) >= 2 and len(ratios) >= 2:
        cluster_matrix = np.nan_to_num(subject_matrix, nan=np.nanmean(subject_matrix))
        subject_order = leaves_list(linkage(cluster_matrix.T, method="average"))
        subjects = [subjects[idx] for idx in subject_order]

    columns = subjects + ["mean"]
    heatmap = np.full((len(ratios), len(columns)), np.nan)
    for row_idx, ratio in enumerate(ratios):
        row_values = []
        for col_idx, subject in enumerate(subjects):
            if (ratio, subject) in scores:
                score = scores[(ratio, subject)]
                heatmap[row_idx, col_idx] = score
                row_values.append(score)
        if row_values:
            heatmap[row_idx, len(columns) - 1] = mean(row_values)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(columns) * 0.75), max(3, len(ratios) * 0.7)))
    cmap = plt.get_cmap("Blues")
    norm = plt.Normalize(vmin=np.nanmin(heatmap), vmax=np.nanmax(heatmap))
    im = ax.imshow(heatmap, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(ratios)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(ratios)
    ax.set_xlabel("held-out subject")
    ax.set_ylabel("image-recon augmentation ratio")
    ax.set_title(f"{DATASET_NAME}: subject-wise test {metric_label}")

    for row_idx in range(len(ratios)):
        for col_idx in range(len(columns)):
            value = heatmap[row_idx, col_idx]
            if np.isfinite(value):
                r, g, b, _ = cmap(norm(value))
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "black" if luminance > 0.6 else "white"
                label = f"{value:.3f}" if col_idx == len(columns) - 1 else f"{value:.2f}"
                ax.text(
                    col_idx,
                    row_idx,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

    ax.axvline(len(subjects) - 0.5, color="black", linewidth=1.5)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"test {metric_label}")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def write_best_aug_vs_baseline_bar_chart(
    path,
    summary,
    metric_key="f1",
    metric_label="F1",
):
    baseline_items = [item for item in summary if item["ratio"] == 0.0]
    aug_items = [item for item in summary if item["ratio"] > 0.0]
    mean_key = f"{metric_key}_mean"
    std_key = f"{metric_key}_std"
    aug_items = [item for item in aug_items if item.get(mean_key) is not None]
    if not baseline_items or not aug_items or baseline_items[0].get(mean_key) is None:
        return False

    try:
        import matplotlib
    except ModuleNotFoundError:
        print("Skipping best-ratio bar chart: matplotlib is not installed in this Python environment.")
        return False

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    baseline_item = baseline_items[0]
    best_item = max(aug_items, key=lambda item: item[mean_key])

    labels = [
        "baseline\n0%",
        f"best augmentation\n{format_ratio_label(best_item['ratio'])}",
    ]
    means = [baseline_item[mean_key], best_item[mean_key]]
    stds = [baseline_item[std_key], best_item[std_key]]
    improvement_pct = (best_item[mean_key] - baseline_item[mean_key]) * 100

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.5))
    bar_width = 0.14
    x = np.array([0.0, bar_width])
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        width=bar_width,
        capsize=8,
        color=["#9e9e9e", "#2f6fbb"],
        edgecolor="black",
        linewidth=1.0,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"test {metric_label}")
    ax.set_title(f"{DATASET_NAME}: baseline vs best image-recon augmentation", fontsize=10)
    ax.set_ylim(0, min(1.05, max(mean_value + std_value for mean_value, std_value in zip(means, stds)) + 0.08))
    ax.set_xlim(-0.12, 0.26)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, mean_value, std_value in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean_value / 2,
            f"{mean_value:.3f} ± {std_value:.3f}",
            ha="center",
            va="center",
            fontsize=9,
            rotation=90,
            color="white",
            fontweight="bold",
        )

    ax.text(
        x[1],
        max(mean_value + std_value for mean_value, std_value in zip(means, stds)) + 0.025,
        f"{improvement_pct:+.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True

def build_img_recon_rows(results_root):
    summary = []
    for ratio in IMG_RECON_AUG_RATIOS:
        dataset_name = result_name_for_img_recon_ratio(ratio)
        f1_by_subject = collect_final_metric_by_subject(results_root, dataset_name, "test_f1_micro")
        auc_by_subject = collect_final_metric_by_subject(results_root, dataset_name, "test_auroc")
        f1_values = list(f1_by_subject.values())
        auc_values = list(auc_by_subject.values())

        if not f1_values:
            print(f"Skipping missing result folder or empty metrics: {dataset_name}")
            continue

        f1_mean, f1_std, f1_text = format_mean_std(f1_values)
        auc_mean, auc_std, auc_text = format_mean_std(auc_values) if auc_values else (None, None, "n/a")
        summary.append(
            {
                "ratio": ratio,
                "active_views": 9 if ratio > 0 else 1,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "f1_text": f1_text,
                "f1_by_subject": f1_by_subject,
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "auc_text": auc_text,
                "auc_by_subject": auc_by_subject,
            }
        )

    if not summary:
        raise RuntimeError("No result metrics found for any configured image-recon ratio.")

    baseline_items = [item for item in summary if item["ratio"] == 0.0]
    baseline_item = baseline_items[0] if baseline_items else summary[0]
    baseline_f1 = baseline_item["f1_mean"]
    baseline_by_subject = baseline_item["f1_by_subject"]
    baseline_auc_by_subject = baseline_item["auc_by_subject"]

    rows = []
    subject_rows = []
    performance_rows = []
    for item in summary:
        for subject, f1_value in sorted(item["f1_by_subject"].items()):
            auc_value = item["auc_by_subject"].get(subject)
            performance_rows.append(
                {
                    "Image-Recon Aug Ratio": format_ratio_label(item["ratio"]),
                    "Subject": subject,
                    "Test F1": f"{f1_value:.3f}",
                    "Test AUROC": f"{auc_value:.3f}" if auc_value is not None else "n/a",
                }
            )

        if item["ratio"] == 0.0:
            paired_n = "reference"
            paired_gain = "reference"
            paired_t = "reference"
            paired_p = "reference"
            paired_auc_gain = "reference"
            paired_auc_t = "reference"
            paired_auc_p = "reference"
            improved = "reference"
            worsened = "reference"
            unchanged = "reference"
            auc_improved = "reference"
            auc_worsened = "reference"
            auc_unchanged = "reference"
        else:
            paired_subjects = sorted(set(baseline_by_subject) & set(item["f1_by_subject"]))
            baseline_values = [baseline_by_subject[subject] for subject in paired_subjects]
            augmented_values = [item["f1_by_subject"][subject] for subject in paired_subjects]
            gains = [
                augmented - baseline
                for augmented, baseline in zip(augmented_values, baseline_values)
            ]

            paired_n = str(len(paired_subjects))
            paired_gain = f"{mean(gains):+.3f} ({mean(gains) * 100:+.1f}%)" if gains else "n/a"
            improved = str(sum(gain > 0 for gain in gains))
            worsened = str(sum(gain < 0 for gain in gains))
            unchanged = str(sum(gain == 0 for gain in gains))

            if len(gains) >= 2:
                t_stat, p_value = ttest_rel(augmented_values, baseline_values)
                paired_t = f"{t_stat:.3f}"
                paired_p = f"{p_value:.4f}"
            else:
                paired_t = "n/a"
                paired_p = "n/a"

            paired_auc_subjects = sorted(set(baseline_auc_by_subject) & set(item["auc_by_subject"]))
            baseline_auc_values = [baseline_auc_by_subject[subject] for subject in paired_auc_subjects]
            augmented_auc_values = [item["auc_by_subject"][subject] for subject in paired_auc_subjects]
            auc_gains = [
                augmented - baseline
                for augmented, baseline in zip(augmented_auc_values, baseline_auc_values)
            ]

            paired_auc_gain = f"{mean(auc_gains):+.3f} ({mean(auc_gains) * 100:+.1f}%)" if auc_gains else "n/a"
            auc_improved = str(sum(gain > 0 for gain in auc_gains))
            auc_worsened = str(sum(gain < 0 for gain in auc_gains))
            auc_unchanged = str(sum(gain == 0 for gain in auc_gains))

            if len(auc_gains) >= 2:
                auc_t_stat, auc_p_value = ttest_rel(augmented_auc_values, baseline_auc_values)
                paired_auc_t = f"{auc_t_stat:.3f}"
                paired_auc_p = f"{auc_p_value:.4f}"
            else:
                paired_auc_t = "n/a"
                paired_auc_p = "n/a"

            for subject, baseline_value, augmented_value, subject_gain in zip(
                paired_subjects, baseline_values, augmented_values, gains
            ):
                baseline_auc = baseline_auc_by_subject.get(subject)
                augmented_auc = item["auc_by_subject"].get(subject)
                subject_auc_gain = (
                    augmented_auc - baseline_auc
                    if baseline_auc is not None and augmented_auc is not None
                    else None
                )
                subject_rows.append(
                    {
                        "Image-Recon Aug Ratio": format_ratio_label(item["ratio"]),
                        "Subject": subject,
                        "Baseline F1": f"{baseline_value:.3f}",
                        "Augmented F1": f"{augmented_value:.3f}",
                        "Subject Gain": f"{subject_gain:+.3f} ({subject_gain * 100:+.1f}%)",
                        "Baseline AUROC": f"{baseline_auc:.3f}" if baseline_auc is not None else "n/a",
                        "Augmented AUROC": f"{augmented_auc:.3f}" if augmented_auc is not None else "n/a",
                        "Subject AUROC Gain": (
                            f"{subject_auc_gain:+.3f} ({subject_auc_gain * 100:+.1f}%)"
                            if subject_auc_gain is not None
                            else "n/a"
                        ),
                    }
                )

        rows.append(
            {
                "Image-Recon Aug Ratio": format_ratio_label(item["ratio"]),
                "Test F1 (mean ± std)": item["f1_text"],
                "Test AUROC (mean ± std)": item["auc_text"],
                "Paired Subjects": paired_n,
                "Mean Paired F1 Gain Across Subjects": paired_gain,
                "Mean Paired AUROC Gain Across Subjects": paired_auc_gain,
                "Subjects Improved (F1)": improved,
                "Subjects Worsened (F1)": worsened,
                "Subjects Unchanged (F1)": unchanged,
                "Subjects Improved (AUROC)": auc_improved,
                "Subjects Worsened (AUROC)": auc_worsened,
                "Subjects Unchanged (AUROC)": auc_unchanged,
                "Paired F1 t-statistic": paired_t,
                "Paired F1 t-test p-value": paired_p,
                "Paired AUROC t-statistic": paired_auc_t,
                "Paired AUROC t-test p-value": paired_auc_p,
            }
        )

    headers = [
        "Image-Recon Aug Ratio",
        "Test F1 (mean ± std)",
        "Test AUROC (mean ± std)",
        "Paired Subjects",
        "Mean Paired F1 Gain Across Subjects",
        "Mean Paired AUROC Gain Across Subjects",
        "Subjects Improved (F1)",
        "Subjects Worsened (F1)",
        "Subjects Unchanged (F1)",
        "Subjects Improved (AUROC)",
        "Subjects Worsened (AUROC)",
        "Subjects Unchanged (AUROC)",
        "Paired F1 t-statistic",
        "Paired F1 t-test p-value",
        "Paired AUROC t-statistic",
        "Paired AUROC t-test p-value",
    ]
    subject_headers = [
        "Image-Recon Aug Ratio",
        "Subject",
        "Baseline F1",
        "Augmented F1",
        "Subject Gain",
        "Baseline AUROC",
        "Augmented AUROC",
        "Subject AUROC Gain",
    ]
    return (
        rows,
        headers,
        os.path.join(IMG_RECON_EXPORT_DIR, "summary_table"),
        subject_rows,
        subject_headers,
        os.path.join(IMG_RECON_EXPORT_DIR, "subjectwise_gains"),
        performance_rows,
        summary,
    )


def build_sparse_dense_rows(results_root):
    summary = []
    for dataset_name, sparse_ratio, dense_ratio in SPARSE_DENSE_DATASET_CONFIGS:
        f1_values = collect_final_metric_values(results_root, dataset_name, "test_f1_micro")

        if not f1_values:
            print(f"Skipping missing result folder or empty metrics: {dataset_name}")
            continue

        f1_mean, _, f1_text = format_mean_std(f1_values)
        summary.append(
            {
                "sparse_ratio": sparse_ratio,
                "dense_ratio": dense_ratio,
                "f1_mean": f1_mean,
                "f1_text": f1_text,
            }
        )

    if not summary:
        raise RuntimeError("No result metrics found for any configured sparse/dense result folder.")

    baseline_f1 = summary[0]["f1_mean"]

    rows = []
    for idx, item in enumerate(summary):
        if idx == 0:
            f1_gain = "reference"
        else:
            gain = item["f1_mean"] - baseline_f1
            f1_gain = f"{gain:+.3f} ({gain * 100:+.1f}%)"

        rows.append(
            {
                "Sparse Data Ratio": item["sparse_ratio"],
                "Dense Data Ratio": item["dense_ratio"],
                "Test F1 (mean ± std)": item["f1_text"],
                "Absolute F1 Gain vs. 0%": f1_gain,
            }
        )

    headers = [
        "Sparse Data Ratio",
        "Dense Data Ratio",
        "Test F1 (mean ± std)",
        "Absolute F1 Gain vs. 0%",
    ]
    return rows, headers, "results/laura_sparse_dense_summary_table"


def main():
    if EXPORT_STRATEGY == "img_recon":
        (
            rows,
            headers,
            output_prefix,
            subject_rows,
            subject_headers,
            subject_output_prefix,
            performance_rows,
            img_recon_summary,
        ) = build_img_recon_rows(IMG_RECON_RESULTS_ROOT)
    elif EXPORT_STRATEGY == "sparse_dense":
        rows, headers, output_prefix = build_sparse_dense_rows(RESULTS_ROOT)
        subject_rows = []
        subject_headers = []
        subject_output_prefix = None
        performance_rows = []
        img_recon_summary = []
    else:
        raise ValueError(f"Unknown EXPORT_STRATEGY: {EXPORT_STRATEGY}")

    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    csv_path = output_prefix + ".csv"
    md_path = output_prefix + ".md"

    write_csv(csv_path, rows, headers)
    write_markdown(md_path, rows, headers)

    print(f"Saved table to {csv_path}")
    print(f"Saved table to {md_path}")

    if subject_output_prefix and subject_rows:
        subject_csv_path = subject_output_prefix + ".csv"
        subject_md_path = subject_output_prefix + ".md"
        write_csv(subject_csv_path, subject_rows, subject_headers)
        write_markdown(subject_md_path, subject_rows, subject_headers)
        heatmap_path = os.path.join(IMG_RECON_EXPORT_DIR, "subjectwise_gain_heatmap.png")
        heatmap_written = write_subject_gain_heatmap(
            heatmap_path,
            subject_rows,
            gain_column="Subject Gain",
            metric_label="F1",
        )
        print(f"Saved subject-wise gains to {subject_csv_path}")
        print(f"Saved subject-wise gains to {subject_md_path}")
        if heatmap_written:
            print(f"Saved subject-wise gain heatmap to {heatmap_path}")

        performance_heatmap_path = os.path.join(
            IMG_RECON_EXPORT_DIR, "subjectwise_f1_heatmap.png"
        )
        performance_heatmap_written = write_subject_performance_heatmap(
            performance_heatmap_path,
            performance_rows,
            score_column="Test F1",
            metric_label="F1",
        )
        if performance_heatmap_written:
            print(f"Saved subject-wise F1 heatmap to {performance_heatmap_path}")

        auc_gain_heatmap_path = os.path.join(
            IMG_RECON_EXPORT_DIR, "subjectwise_auc_gain_heatmap.png"
        )
        auc_gain_heatmap_written = write_subject_gain_heatmap(
            auc_gain_heatmap_path,
            subject_rows,
            gain_column="Subject AUROC Gain",
            metric_label="AUROC",
        )
        if auc_gain_heatmap_written:
            print(f"Saved subject-wise AUROC gain heatmap to {auc_gain_heatmap_path}")

        auc_performance_heatmap_path = os.path.join(
            IMG_RECON_EXPORT_DIR, "subjectwise_auc_heatmap.png"
        )
        auc_performance_heatmap_written = write_subject_performance_heatmap(
            auc_performance_heatmap_path,
            performance_rows,
            score_column="Test AUROC",
            metric_label="AUROC",
        )
        if auc_performance_heatmap_written:
            print(f"Saved subject-wise AUROC heatmap to {auc_performance_heatmap_path}")

        f1_bar_chart_path = os.path.join(
            IMG_RECON_EXPORT_DIR, "best_aug_vs_baseline_f1_bar.png"
        )
        f1_bar_chart_written = write_best_aug_vs_baseline_bar_chart(
            f1_bar_chart_path,
            img_recon_summary,
            metric_key="f1",
            metric_label="F1",
        )
        if f1_bar_chart_written:
            print(f"Saved best-ratio F1 bar chart to {f1_bar_chart_path}")

        auc_bar_chart_path = os.path.join(
            IMG_RECON_EXPORT_DIR, "best_aug_vs_baseline_auc_bar.png"
        )
        auc_bar_chart_written = write_best_aug_vs_baseline_bar_chart(
            auc_bar_chart_path,
            img_recon_summary,
            metric_key="auc",
            metric_label="AUROC",
        )
        if auc_bar_chart_written:
            print(f"Saved best-ratio AUROC bar chart to {auc_bar_chart_path}")


if __name__ == "__main__":
    main()
