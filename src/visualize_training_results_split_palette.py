import math
from matplotlib.lines import Line2D
def build_subject_grid(subjects, subject_cols):
    """Create a grid of subplots for per-subject plots."""
    n_subjects = len(subjects)
    n_rows = math.ceil(n_subjects / subject_cols)
    fig, axes = plt.subplots(
        n_rows,
        subject_cols,
        figsize=(4 * subject_cols, SUBJECT_ROW_HEIGHT * n_rows),
        dpi=400,
    )
    axes = np.array(axes).reshape(n_rows, subject_cols)
    return fig, axes
#!/usr/bin/env python3
"""
Aggregate model evaluation metrics stored inside the training/ directory and
generate multi-figure summaries. The script automatically discovers datasets,
models, and subjects by recursively scanning for `loss/mats_*.npy` (or JSON)
files. Each dataset is assigned a single base color that is reused across all
figures; model-specific colors are simple lightness variations of that base so
datasets remain visually consistent everywhere.

Configuration is handled via the constants near the top of this file—edit them
directly instead of supplying CLI flags.
"""

# from __future__ import annotations
import json
import re
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple


def place_legend_below_title(
    fig: plt.Figure,
    legend_items: Sequence[object],
    legend_labels: Sequence[str],
    *,
    ncol: int = 1,
    handler_map: Mapping[object, object] | None = None,
) -> None:
    """Place a figure-level legend below the title with generous spacing."""
    if not legend_items:
        return
    legend_kwargs = dict(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=ncol,
        frameon=True,
        fontsize=8,
        borderpad=0.3,
        labelspacing=0.2,
        handlelength=1.0,
        handletextpad=0.3,
    )
    if handler_map:
        legend_kwargs["handler_map"] = handler_map
    fig.legend(legend_items, legend_labels, **legend_kwargs)

try:  # Switch to a non-interactive backend when possible.
    plt.switch_backend("Agg")
except Exception:
    pass

# --- Selection configuration ---
# Dictionary mapping dataset name to list of model names to include.
# --- Selection configuration ---
# Dictionary mapping dataset name to list of model names to include.
# If the list is empty, all models for that dataset are included.
INCLUDE = {
    # "BallSqueezingHD_modified": [
    #   "CNN2D_BaselineV2_lr1e-5__20251218_032322", # parcel space
    #   "CNN2D_BaselineV2_lr1e-5__20251218_032710", # channel space
    # ],
        
    #     # --------------- FreshMotor ---------------
    "BallSqueezingHD_modified__FreshMotor": [
        # "CNN2D_BaselineV2_lr1e-5__aug20%__20251218_044850",
        "CNN2D_BaselineV2_lr1e-5__aug100%__20251218_045119",
    ],
    # "FreshMotor": [
        # "CNN2D_BaselineV2_lr1e-5__20251218_032839",
        # "CNN2D_BaselineV2_lr1e-5__20251218_040819", # channel space
    # ],
}

SPLIT_COLUMNS = OrderedDict(
    [
        ("train_loss", 0),
        ("train_f1", 1),
        ("val_loss", 2),
        ("val_f1", 3),
        ("test_loss", 4),
        ("test_f1", 5),
    ]
)
SPLIT_ORDER = ("train", "val", "test")
SUBJECT_PATTERN = re.compile(r"(sub-[a-zA-Z0-9]+)")

SPLIT_BASE_COLORS = {
    "train": (0.0, 0.45, 0.9),  # blue
    "val": (0.1, 0.7, 0.3),     # green
    "test": (0.9, 0.35, 0.2),   # orange/red
}
SPLIT_GRADIENT_LEVEL_LABELS = {
    "train": "train",
    "val": "val",
    "test": "test",
}
GRADIENT_LEGEND_BASE_COLOR = SPLIT_BASE_COLORS["train"]

MODEL_HATCHES = ["", "//", "\\\\", "xx", "..", "++", "oo", "**", "--", "||", "..//", "\\\\..", "//\\\\"]


def assign_model_hatches(
    combos: Sequence[Tuple[str, str]]
) -> Dict[Tuple[str, str], str]:
    hatches: Dict[Tuple[str, str], str] = {}
    for idx, combo in enumerate(combos):
        hatches[combo] = MODEL_HATCHES[idx % len(MODEL_HATCHES)]
    return hatches


def split_legend_label(split_key: str) -> str:
    level = SPLIT_GRADIENT_LEVEL_LABELS[split_key]
    return f"{split_key.capitalize()} ({level} shade)"


def split_color_for_combo(
    combo_key: Tuple[str, str],
    model_colors: Mapping[Tuple[str, str], Tuple[float, float, float]],
    split_key: str,
) -> Tuple[float, float, float]:
    return SPLIT_BASE_COLORS[split_key]


def build_split_line_handles(
    split_config: Sequence[Tuple[str, str, str, Mapping[str, float]]],
    combos: Sequence[Tuple[str, str]],
    model_colors: Mapping[Tuple[str, str], Tuple[float, float, float]],
) -> List[Tuple[Tuple[Line2D, ...], str]]:
    handles: List[Tuple[Tuple[Line2D, ...], str]] = []
    for _, _, split_key, style in split_config:
        lines = tuple(
            Line2D([], [], color=split_color_for_combo(combo, model_colors, split_key), **style)
            for combo in combos
        )
        if lines:
            handles.append((lines, split_legend_label(split_key)))
    return handles


def build_split_patch_handles(
    combos: Sequence[Tuple[str, str]],
    model_colors: Mapping[Tuple[str, str], Tuple[float, float, float]],
) -> List[Tuple[Tuple[Patch, ...], str]]:
    handles: List[Tuple[Tuple[Patch, ...], str]] = []
    for split_key in SPLIT_ORDER:
        patches = tuple(
            Patch(
                facecolor=split_color_for_combo(combo, model_colors, split_key),
                edgecolor="black",
            )
            for combo in combos
        )
        if patches:
            handles.append((patches, split_legend_label(split_key)))
    return handles

# Simple configuration block; update these values before executing the script.
TRAINING_ROOT = Path("training")
OUTPUT_DIR = Path("results/analysis_results")
SUBJECT_GRID_COLUMNS = 4
SUBJECT_ROW_HEIGHT = 4.5
RUN_TAG_RE = re.compile(r"__([0-9]{8}_[0-9]{6})")


def discover_runs(training_root: Path) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Recursively search for loss directories and load per-subject arrays."""
    runs: Dict[str, Dict[str, Dict[str, np.ndarray]]] = defaultdict(dict)
    if not training_root.exists():
        raise FileNotFoundError(f"Training root '{training_root}' does not exist.")

    loss_dirs = sorted(
        [
            path
            for path in training_root.rglob("loss")
            if path.is_dir() and any(path.glob("mats_*.*"))
        ]
    )
    for loss_dir in loss_dirs:
        rel = loss_dir.relative_to(training_root)
        parts = rel.parts
        if len(parts) < 2:
            continue
        dataset = parts[0]
        model_parts = parts[1:-1]  # drop trailing "loss"
        if not model_parts:
            continue
        model = "/".join(model_parts)
        subject_arrays = load_subject_arrays(loss_dir)
        if subject_arrays:
            runs[dataset][model] = subject_arrays
    return runs


def load_subject_arrays(loss_dir: Path) -> Dict[str, np.ndarray]:
    """Load subject-specific metric arrays from .npy or JSON files."""
    subject_data: Dict[str, np.ndarray] = {}
    files = sorted(loss_dir.glob("mats_*.*"))
    if not files:
        files = sorted(loss_dir.glob("*sub-*.*_new"))
    for file_path in files:
        subject = extract_subject_name(file_path.stem)
        if file_path.suffix == ".npy":
            arr = np.load(file_path)
        elif file_path.suffix == ".json":
            arr = metrics_json_to_array(file_path)
        else:
            continue
        if arr is None:
            continue
        subject_data[subject] = arr
    return subject_data


def extract_subject_name(filename_stem: str) -> str:
    match = SUBJECT_PATTERN.search(filename_stem)
    return match.group(1) if match else filename_stem


def metrics_json_to_array(json_path: Path) -> np.ndarray | None:
    with open(json_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, list) and payload and isinstance(payload[0], Mapping):
        rows = []
        for entry in payload:
            rows.append([float(entry.get(name, np.nan)) for name in SPLIT_COLUMNS])
        return np.asarray(rows, dtype=float)
    if isinstance(payload, Mapping):
        lengths = {len(v) for v in payload.values() if isinstance(v, Sequence)}
        if lengths:
            steps = max(lengths)
        else:
            return None
        arr = np.full((steps, len(SPLIT_COLUMNS)), np.nan, dtype=float)
        for name, column in SPLIT_COLUMNS.items():
            values = payload.get(name, [])
            for idx, value in enumerate(values):
                arr[idx, column] = float(value)
        return arr
    return None


def assign_dataset_colors(datasets: Sequence[str]) -> Dict[str, Tuple[float, float, float]]:
    """Map each dataset to a distinct shade of blue."""
    cmap = plt.colormaps.get_cmap("Blues")
    colors: Dict[str, Tuple[float, float, float]] = {}
    for idx, dataset in enumerate(sorted(datasets)):
        # Skip the very lightest shades to keep contrast
        shade = 0.4 + 0.5 * (idx % cmap.N) / max(1, cmap.N - 1)
        colors[dataset] = cmap(shade)[:3]
    return colors


def derive_model_colors(
    runs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    dataset_colors: Mapping[str, Tuple[float, float, float]],
) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
    model_colors: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
    for dataset, models in runs.items():
        base = dataset_colors[dataset]
        model_names = sorted(models.keys())
        shades = compute_lightness_variations(len(model_names))
        for i, model_name in enumerate(model_names):
            # Use increasing/decreasing lightness for each model
            model_colors[(dataset, model_name)] = adjust_lightness(base, shades[i])
    return model_colors


def compute_lightness_variations(count: int) -> List[float]:
    if count <= 1:
        return [1.0]
    return np.linspace(0.65, 1.15, count).tolist()


def adjust_lightness(color: Tuple[float, float, float], factor: float) -> Tuple[float, float, float]:
    # Convert RGB -> HLS and tweak lightness to keep dataset hue the same.
    import colorsys

    r, g, b = color[:3]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.05, min(0.95, l * factor))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (r2, g2, b2)


def flatten_subjects(runs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]) -> List[str]:
    subjects = set()
    for models in runs.values():
        for subject_map in models.values():
            subjects.update(subject_map.keys())
    return sorted(subjects)


def compute_best_epoch_scores(arr: np.ndarray) -> Dict[str, float]:
    if arr.ndim != 2 or arr.shape[1] < len(SPLIT_COLUMNS):
        raise ValueError("Metric arrays must have shape (epochs, metrics).")
    val_idx = SPLIT_COLUMNS["val_f1"]
    val_track = arr[:, val_idx]
    if np.all(np.isnan(val_track)):
        best_epoch = arr.shape[0] - 1
    else:
        best_epoch = int(np.nanargmax(val_track))
    return {
        "train": float(arr[best_epoch, SPLIT_COLUMNS["train_f1"]]),
        "val": float(arr[best_epoch, SPLIT_COLUMNS["val_f1"]]),
        "test": float(arr[best_epoch, SPLIT_COLUMNS["test_f1"]]),
    }


def collect_best_stats(runs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]) -> Dict[Tuple[str, str], Dict[str, Dict[str, float]]]:
    stats: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    for dataset, models in runs.items():
        for model_name, subject_map in models.items():
            key = (dataset, model_name)
            stats[key] = {}
            for subject, arr in subject_map.items():
                try:
                    stats[key][subject] = compute_best_epoch_scores(arr)
                except ValueError:
                    continue
    return stats


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def display_model_name(name: str) -> str:
    """Remove timestamp suffix from a model/run name if present."""
    m = RUN_TAG_RE.search(name)
    return name[: m.start()] if m else name


def display_dataset_model_label(dataset: str, model_name: str) -> str:
    """Include dataset context alongside the cleaned model name."""
    return f"{dataset} | {display_model_name(model_name)}"


def extract_run_tags(runs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]) -> List[str]:
    tags = set()
    for models in runs.values():
        for model_name in models:
            m = RUN_TAG_RE.search(model_name)
            if m:
                tags.add(m.group(1))
    return sorted(tags)


def dataset_subplot_grid(n_items: int) -> Tuple[plt.Figure, np.ndarray]:
    cols = min(3, max(1, n_items))
    rows = math.ceil(n_items / cols) if n_items else 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * SUBJECT_ROW_HEIGHT), squeeze=False, sharey=True)
    return fig, axes


def annotate_bar(ax, bar, text: str) -> None:
    xpos = bar.get_x() + bar.get_width() / 2
    ypos = bar.get_height() + max(0.01, bar.get_height() * 0.02)
    ax.text(xpos, ypos, text, ha="center", va="bottom", rotation=90, fontsize=8)


def plot_across_subject_boxplots(
    runs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    best_stats: Mapping[Tuple[str, str], Mapping[str, Mapping[str, float]]],
    model_colors: Mapping[Tuple[str, str], Tuple[float, float, float]],
    output_dir: Path,
) -> None:
    # Plot all models/datasets in a single subplot
    fig, ax = plt.subplots(figsize=(8, 6))
    model_handles: Dict[str, Patch] = {}
    split_handles = [
        Patch(facecolor=color, edgecolor="black", label=split.upper())
        for split, color in SPLIT_BASE_COLORS.items()
    ]
    base_positions = np.arange(len(SPLIT_ORDER))
    # Use INCLUDE dict order for model display
    def get_defined_order_keys(best_stats):
        order = []
        for dataset in INCLUDE:
            for model in INCLUDE[dataset] if INCLUDE[dataset] else [None]:
                for key in best_stats:
                    if key[0] == dataset and (model is None or key[1] == model):
                        order.append(key)
        # Add any keys not in INCLUDE (for empty lists)
        for key in best_stats:
            if key not in order:
                order.append(key)
        return order
    all_keys = get_defined_order_keys(best_stats)
    model_hatches = assign_model_hatches(all_keys)
    box_data = []
    box_positions = []
    box_colors = []
    box_hatches = []
    n_models = len(all_keys)
    # Offset for each model so boxes are side-by-side
    offsets = np.linspace(-0.25, 0.25, n_models) if n_models > 1 else [0]
    for m_idx, key in enumerate(all_keys):
        split_values: Dict[str, List[float]] = {split: [] for split in SPLIT_ORDER}
        for subject_stats in best_stats.get(key, {}).values():
            for split in SPLIT_ORDER:
                value = subject_stats.get(split)
                if value is not None and not np.isnan(value):
                    split_values[split].append(value)
        for split_idx, split in enumerate(SPLIT_ORDER):
            values = split_values[split]
            if not values:
                continue
            position = base_positions[split_idx] + offsets[m_idx]
            box_data.append(values)
            box_positions.append(position)
            box_colors.append(split_color_for_combo(key, model_colors, split))
            box_hatches.append(model_hatches[key])
            label = display_dataset_model_label(key[0], key[1])
            if label not in model_handles:
                model_handles[label] = Patch(
                    facecolor="white",
                    edgecolor="black",
                    hatch=model_hatches[key],
                    label=label,
                )
    if not box_data:
        ax.axis("off")
    else:
        # Side-by-side boxes at each tick
        bp = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=0.5 / max(1, n_models),
            patch_artist=True,
            boxprops=dict(alpha=0.8),
            medianprops=dict(color="black"),
        )
        for patch, color, hatch in zip(bp["boxes"], box_colors, box_hatches):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_hatch(hatch)
        for median in bp["medians"]:
            median.set_color("black")
        ax.set_xticks(base_positions, [split.upper() for split in SPLIT_ORDER])
        ax.set_ylabel("F1 Score")
        ax.set_title("Across-Subjects F1 Distribution", pad=18)
        ax.grid(axis="y", alpha=0.3)
    legend_items: List[object] = list(model_handles.values())
    legend_labels: List[str] = [handle.get_label() for handle in legend_items]
    for patch in split_handles:
        legend_items.append(patch)
        legend_labels.append(patch.get_label())
    fig.tight_layout(rect=[0, 0, 1, 0.78])
    place_legend_below_title(fig, legend_items, legend_labels)
    save_path = output_dir / "across_subjects_f1_boxplots.png"
    fig.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close(fig)


def plot_across_subject_bars(
    runs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    best_stats: Mapping[Tuple[str, str], Mapping[str, Mapping[str, float]]],
    model_colors: Mapping[Tuple[str, str], Tuple[float, float, float]],
    output_dir: Path,
) -> None:
    # Correct single-axis bar plot implementation
    fig, ax = plt.subplots(figsize=(8, 6))
    model_handles: Dict[str, Patch] = {}
    split_handles = [
        Patch(facecolor=color, edgecolor="black", label=split.upper())
        for split, color in SPLIT_BASE_COLORS.items()
    ]
    x = np.arange(len(SPLIT_ORDER))
    # Use INCLUDE dict order for model display
    def get_defined_order_keys(best_stats):
        order = []
        for dataset in INCLUDE:
            for model in INCLUDE[dataset] if INCLUDE[dataset] else [None]:
                for key in best_stats:
                    if key[0] == dataset and (model is None or key[1] == model):
                        order.append(key)
        # Add any keys not in INCLUDE (for empty lists)
        for key in best_stats:
            if key not in order:
                order.append(key)
        return order
    all_keys = get_defined_order_keys(best_stats)
    model_hatches = assign_model_hatches(all_keys)
    width = 0.8 / max(1, len(all_keys))
    bar_text_size = 10
    for m_idx, key in enumerate(all_keys):
        means, stds = [], []
        for split in SPLIT_ORDER:
            values = [stat[split] for stat in best_stats.get(key, {}).values() if split in stat]
            if values:
                means.append(float(np.nanmean(values)))
                stds.append(float(np.nanstd(values)))
            else:
                means.append(np.nan)
                stds.append(0.0)
        offset = (m_idx - (len(all_keys) - 1) / 2) * width
        split_colors = [SPLIT_BASE_COLORS[split] for split in SPLIT_ORDER]
        bars = ax.bar(
            x + offset,
            means,
            width=width * 0.95,
            yerr=stds,
            capsize=4,
            color=split_colors,
            edgecolor="black",
            linewidth=1.2,
            label=display_dataset_model_label(key[0], key[1]),
            hatch=model_hatches[key],
        )
        for bar, mean, std in zip(bars, means, stds):
            if np.isnan(mean):
                continue
            xpos = bar.get_x() + bar.get_width() / 2
            ypos = bar.get_height() / 2
            txt = f"{mean:.2f} ± {std:.2f}"
            ax.text(
                xpos,
                ypos,
                txt,
                ha='center',
                va='center',
                rotation=90,
                fontsize=bar_text_size,
                color='black',
            )
        model_label = display_dataset_model_label(key[0], key[1])
        if model_label not in model_handles:
            model_handles[model_label] = Patch(
                facecolor="white",
                edgecolor="black",
                hatch=model_hatches[key],
                label=model_label,
            )
    ax.set_xticks(x, [split.upper() for split in SPLIT_ORDER])
    ax.set_ylabel("F1 Score")
    ax.set_title("Across-Subjects Mean ± STD F1")
    ax.grid(axis="y", alpha=0.3)
    legend_items: List[object] = list(model_handles.values())
    legend_labels: List[str] = [handle.get_label() for handle in legend_items]
    for patch in split_handles:
        legend_items.append(patch)
        legend_labels.append(patch.get_label())
    fig.tight_layout(rect=[0, 0, 1, 0.78])
    place_legend_below_title(
        fig,
        legend_items,
        legend_labels,
    )
    save_path = output_dir / "across_subjects_f1_bar.png"
    fig.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close(fig)


def plot_subject_curves(
    subjects: Sequence[str],
    runs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    model_colors: Mapping[Tuple[str, str], Tuple[float, float, float]],
    output_dir: Path,
    subject_cols: int,
    metric: str,
    filename: str,
    combos: Sequence[Tuple[str, str]],
) -> None:
    if not subjects:
        return
    fig, axes = build_subject_grid(subjects, subject_cols)
    combo_handles: Dict[str, Line2D] = {}
    split_config = {
        "f1": [
            ("Train", "train_f1", "train", {"lw": 2.4, "ls": "-", "alpha": 1.0}),
            ("Validation", "val_f1", "val", {"lw": 2.2, "ls": "-", "alpha": 0.9}),
            ("Test", "test_f1", "test", {"lw": 2.2, "ls": "-", "alpha": 0.9}),
        ],
        "loss": [
            ("Train", "train_loss", "train", {"lw": 2.4, "ls": "-", "alpha": 1.0}),
            ("Validation", "val_loss", "val", {"lw": 2.2, "ls": "-", "alpha": 0.9}),
            ("Test", "test_loss", "test", {"lw": 2.2, "ls": "-", "alpha": 0.9}),
        ],
    }[metric]
    split_handles = build_split_line_handles(split_config, combos, model_colors)
    ylabel = "F1 Score" if metric == "f1" else "Loss"
    flat_axes = axes.flatten()
    for ax, subject in zip(flat_axes, subjects):
        plotted = False
        for dataset, models in runs.items():
            for model_name, subject_map in models.items():
                arr = subject_map.get(subject)
                if arr is None:
                    continue
                combo_key = (dataset, model_name)
                base_color = model_colors.get(combo_key, GRADIENT_LEGEND_BASE_COLOR)
                epochs = np.arange(arr.shape[0])
                label = f"{dataset} | {display_model_name(model_name)}"
                model_plotted = False
                # Always use best validation F1 epoch for vertical line, regardless of metric
                val_f1_idx = SPLIT_COLUMNS["val_f1"]
                val_f1_track = arr[:, val_f1_idx]
                if np.all(np.isnan(val_f1_track)):
                    best_val_epoch = arr.shape[0] - 1
                else:
                    best_val_epoch = int(np.nanargmax(val_f1_track))
                for _, column_key, split_key, style in split_config:
                    values = arr[:, SPLIT_COLUMNS[column_key]]
                    if np.all(np.isnan(values)):
                        continue
                    split_color = split_color_for_combo(combo_key, model_colors, split_key)
                    ax.plot(
                        epochs,
                        values,
                        color=split_color,
                        **style,
                    )
                    plotted = True
                    model_plotted = True
                    # Add dashed vertical line at best validation epoch
                    if best_val_epoch is not None:
                        ax.axvline(best_val_epoch, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="Best Val Epoch")
                    if model_plotted and label not in combo_handles:
                        combo_handles[label] = Line2D(
                            [],
                            [],
                            color=base_color,
                            linestyle="-",
                            linewidth=2.5,
                            label=label,
                        )
        if not plotted:
            ax.axis("off")
            continue
        ax.set_title(subject, pad=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax in flat_axes[len(subjects) :]:
        ax.axis("off")
    legend_items: List[object] = list(combo_handles.values())
    legend_labels: List[str] = [h.get_label() for h in legend_items]
    # Add legend for vertical dashed line (best validation F1 epoch)
    best_val_line = Line2D([], [], color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="Best Val F1 Epoch")
    legend_items.append(best_val_line)
    legend_labels.append("Best Val F1 Epoch")
    for handles, label in split_handles:
        legend_items.append(handles)
        legend_labels.append(label)
    handler_map = {tuple: HandlerTuple(ndivide=None)}
    title = "Per-Subject F1 Curves" if metric == "f1" else "Per-Subject Loss Curves"
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.78])
    place_legend_below_title(
        fig,
        legend_items,
        legend_labels,
        handler_map=handler_map,
    )
    save_path = output_dir / filename
    fig.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    runs = discover_runs(TRAINING_ROOT)
    if not runs:
        raise SystemExit("No training runs with metrics were found. Ensure loss/mats_*.npy files exist.")
    # --- Filter runs by INCLUDE dict ---
    if INCLUDE:
        filtered_runs = {}
        for dataset, models in runs.items():
            if dataset not in INCLUDE:
                continue
            model_list = INCLUDE[dataset]
            if model_list:
                filtered_models = {m: arrs for m, arrs in models.items() if m in model_list}
            else:
                filtered_models = dict(models)
            if filtered_models:
                filtered_runs[dataset] = filtered_models
        if not filtered_runs:
            raise SystemExit("No runs matched INCLUDE selection.")
    else:
        filtered_runs = runs

    run_tags = extract_run_tags(filtered_runs)
    output_dir = OUTPUT_DIR
    if len(run_tags) == 1:
        output_dir = Path(f"{OUTPUT_DIR}_{run_tags[0]}")

    ensure_output_dir(output_dir)
    per_subject_dir = output_dir / "per_subject"
    ensure_output_dir(per_subject_dir)

    dataset_colors = assign_dataset_colors(filtered_runs.keys())
    model_colors = derive_model_colors(filtered_runs, dataset_colors)
    best_stats = collect_best_stats(filtered_runs)
    subjects = flatten_subjects(filtered_runs)
    # Sort subjects: BallSqueezing first, then FreshMotor
    def subject_dataset(subject):
        for dataset in filtered_runs:
            for model in filtered_runs[dataset]:
                if subject in filtered_runs[dataset][model]:
                    return dataset
        return ""
    subjects = sorted(
        subjects,
        key=lambda s: (0 if "BallSqueezing" in subject_dataset(s) else 1, s)
    )

    plot_across_subject_boxplots(filtered_runs, best_stats, model_colors, output_dir)
    plot_across_subject_bars(filtered_runs, best_stats, model_colors, output_dir)
    plot_per_subject_bars(
        subjects,
        best_stats,
        per_subject_dir,
        subject_cols=SUBJECT_GRID_COLUMNS,
        filtered_runs=filtered_runs,
        model_colors=model_colors,
    )


def plot_per_subject_bars(
    subjects: Sequence[str],
    best_stats: Mapping[Tuple[str, str], Mapping[str, Mapping[str, float]]],
    output_dir: Path,
    subject_cols: int,
    filtered_runs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    model_colors: Mapping[Tuple[str, str], Tuple[float, float, float]],
) -> None:
    if not subjects:
        return
    fig, axes = build_subject_grid(subjects, subject_cols)
    flat_axes = axes.flatten()
    combos = sorted(best_stats.keys(), key=lambda k: (0 if "BallSqueezing" in k[0] else 1, k[0], k[1]))
    model_hatches = assign_model_hatches(combos)
    split_handles = [
        Patch(facecolor=color, edgecolor="black", label=split.upper())
        for split, color in SPLIT_BASE_COLORS.items()
    ]
    model_handles: List[object] = []
    model_labels: List[str] = []
    for combo in combos:
        label = display_dataset_model_label(combo[0], combo[1])
        model_handles.append(
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=model_hatches[combo],
                label=label,
            )
        )
        model_labels.append(label)
    for ax, subject in zip(flat_axes, subjects):
        plotted = False
        x = np.arange(len(SPLIT_ORDER))
        width = 0.8 / max(1, len(combos))
        for idx_combo, combo in enumerate(combos):
            offset = (idx_combo - (len(combos) - 1) / 2) * width
            subject_stats = best_stats[combo].get(subject)
            if not subject_stats:
                continue
            means = [subject_stats.get(split, np.nan) for split in SPLIT_ORDER]
            values_arr = np.asarray(means, dtype=float)
            if np.isnan(values_arr).all():
                continue
            colors = [SPLIT_BASE_COLORS[split] for split in SPLIT_ORDER]
            bars = ax.bar(
                x + offset,
                means,
                width=width * 0.9,
                color=colors,
                edgecolor="black",
                linewidth=0.6,
                hatch=model_hatches[combo],
            )
            for bar, mean in zip(bars, means):
                if np.isnan(mean):
                    continue
                # Center F1 score text inside bar, with good contrast
                xpos = bar.get_x() + bar.get_width() / 2
                ypos = bar.get_height() / 2
                txt = f"{mean:.2f}"
                face_rgb = np.array(bar.get_facecolor()[:3])
                # Use luminance for contrast: Y = 0.2126 R + 0.7152 G + 0.0722 B
                luminance = 0.2126 * face_rgb[0] + 0.7152 * face_rgb[1] + 0.0722 * face_rgb[2]
                text_color = 'white' if luminance < 0.5 else 'black'
                ax.text(xpos, ypos, txt, ha='center', va='center', rotation=90, fontsize=8, color=text_color)
            plotted = True
        ax.set_xticks(x, [split.upper() for split in SPLIT_ORDER])
        ax.set_ylim(0, 1.05)
        ax.set_title(subject, pad=14)
        ax.grid(axis="y", alpha=0.25)
        if not plotted:
            ax.axis("off")
    for ax in flat_axes[len(subjects):]:
        ax.axis("off")
    legend_items: List[object] = list(model_handles)
    legend_labels: List[str] = list(model_labels)
    for handle in split_handles:
        legend_items.append(handle)
        legend_labels.append(handle.get_label())
    fig.suptitle("Per-Subject F1", y=0.98, fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.8])
    place_legend_below_title(fig, legend_items, legend_labels)
    save_path = output_dir / "per_subject_f1_bars.png"
    fig.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    plot_subject_curves(
        subjects,
        filtered_runs,
        model_colors,
        output_dir,
        subject_cols=subject_cols,
        metric="f1",
        filename="per_subject_f1_curves.png",
        combos=combos,
    )
    plot_subject_curves(
        subjects,
        filtered_runs,
        model_colors,
        output_dir,
        subject_cols=subject_cols,
        metric="loss",
        filename="per_subject_loss_curves.png",
        combos=combos,
    )
    print(f"Figures saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
