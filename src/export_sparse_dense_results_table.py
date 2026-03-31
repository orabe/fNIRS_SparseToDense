#!/usr/bin/env python3
import csv
import glob
import os
import pickle
from statistics import mean, pstdev


DATASET_CONFIGS = [
    (   "train_full__eval_laura_full_1.0",
        0, 
        100
    ),
    (
        "train_full+motor_100chs+motor_91chs+motor_80chs+motor_70chs+motor_59chs+motor_50chs__eval_laura_full_0.1",
        10,
        100,
    ),
    (
        "train_full+motor_100chs+motor_91chs+motor_80chs+motor_70chs+motor_59chs+motor_50chs__eval_laura_full_0.3",
        30,
        100,
    ),
    (
        "train_full+motor_100chs+motor_91chs+motor_80chs+motor_70chs+motor_59chs+motor_50chs__eval_laura_full_0.5",
        50,
        100,
    ),
    (
        "train_full+motor_100chs+motor_91chs+motor_80chs+motor_70chs+motor_59chs+motor_50chs__eval_laura_full_0.7",
        70,
        100,
    ),
    (
        "train_full+motor_100chs+motor_91chs+motor_80chs+motor_70chs+motor_59chs+motor_50chs__eval_laura_full_1.0",
        100,
        100,
    ),
]

RESULTS_ROOT = "results"
OUTPUT_PREFIX = "results/laura_sparse_dense_summary_table"

HEADERS = [
    "Sparse Data Ratio",
    "Dense Data Ratio",
    "Test F1 (mean ± std)",
    "Mean F1 Test Improvement vs. 0% (%)",
]


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


def format_mean_std(values):
    metric_mean = mean(values)
    metric_std = pstdev(values)
    return metric_mean, metric_std, f"{metric_mean:.3f} ± {metric_std:.3f}"


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(HEADERS) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(HEADERS)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(row[h] for h in HEADERS) + " |\n")


def build_rows(results_root):
    summary = []
    for dataset_name, sparse_ratio, dense_ratio in DATASET_CONFIGS:
        f1_values = collect_final_metric_values(results_root, dataset_name, "test_f1_micro")

        if not f1_values:
            raise RuntimeError(
                f"Missing metrics for dataset '{dataset_name}'. "
                "Expected non-empty 'test_f1_micro' in res_*.pkl files."
            )

        f1_mean, _, f1_text = format_mean_std(f1_values)

        summary.append(
            {
                "dataset_name": dataset_name,
                "sparse_ratio": sparse_ratio,
                "dense_ratio": dense_ratio,
                "f1_mean": f1_mean,
                "f1_text": f1_text,
            }
        )

    baseline_f1 = summary[0]["f1_mean"]

    rows = []
    for i, item in enumerate(summary):
        if i == 0:
            f1_improvement = "reference"
        else:
            improvement_pct = ((item['f1_mean'] - baseline_f1) / baseline_f1) * 100
            f1_improvement = f"{improvement_pct:+.2f}%"

        rows.append(
            {
                "Sparse Data Ratio": f"{item['sparse_ratio']}%",
                "Dense Data Ratio": f"{item['dense_ratio']}%",
                "Test F1 (mean ± std)": item["f1_text"],
                "Mean F1 Test Improvement vs. 0% (%)": f1_improvement,
            }
        )

    return rows


def main():
    rows = build_rows(RESULTS_ROOT)

    output_dir = os.path.dirname(OUTPUT_PREFIX)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    csv_path = OUTPUT_PREFIX + ".csv"
    md_path = OUTPUT_PREFIX + ".md"

    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    print(f"Saved table to {csv_path}")
    print(f"Saved table to {md_path}")


if __name__ == "__main__":
    main()
