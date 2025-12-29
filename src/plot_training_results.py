#!/usr/bin/env python3
import glob
import os
import pickle
import matplotlib.pyplot as plt

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
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main():
    # DATASET_NAME = "BallSqueezingHD_modified"
    DATASET_NAME = "channel_BallSqueezingHD_modified"
    result_patterns = [
        # "results/*/res_*.pkl",
        f"results/{DATASET_NAME}/res_*.pkl",
    ]
    output_dir = f"figures/{DATASET_NAME}"
    os.makedirs(output_dir, exist_ok=True)

    result_files = []
    for pattern in result_patterns:
        result_files.extend(glob.glob(pattern))

    all_results = []
    for path in sorted(result_files):
        with open(path, "rb") as handle:
            results = pickle.load(handle)
        label = path.split("/")[-1].split("_")[-3]
        all_results.append((label, results))

    metrics = {
        "loss": ("train_loss", "test_loss"),
        "accuracy": ("train_accuracy", "test_accuracy"),
        "f1": ("train_f1", "test_f1"),
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
        train_f1 = results.get("train_f1", [])
        test_f1 = results.get("test_f1", [])
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
        ax.set_title("Final F1 Mean ± Std", fontsize=13, fontweight="bold")
        ax.set_ylabel("F1")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "f1_mean_std_bar.png"), dpi=150)
        plt.close(fig)

    print(f"Plots saved in {output_dir}/")


if __name__ == "__main__":
    main()
