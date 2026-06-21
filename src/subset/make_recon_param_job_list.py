from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

AUGMENTATION_STRATEGY = "imageRecon_params"


def list_dataset_files(dataset_name: str) -> list[str]:
    raw_path = Path(f"datasets/raw/{dataset_name}")
    if dataset_name == "BallSqueezingHD_modified":
        raw_dir = f"{raw_path}/sub-*/nirs/sub-*.snirf"
    elif dataset_name == "BS_Laura":
        raw_dir = f"{raw_path}/sub-*/nirs/sub-*.snirf"
    elif dataset_name == "Electrical_Thermal":
        raw_dir = f"{raw_path}/sub-*/ses-*/nirs/sub-*_ses-*_task-Electrical*_nirs.snirf"
    elif dataset_name == "FreshMotor":
        raw_dir = f"{raw_path}/sub-*/ses-*/nirs/sub-*_ses-*_task-FRESHMOTOR_nirs.snirf"
    elif dataset_name == "vfc_hd":
        raw_dir = f"{raw_path}/sub-*/nirs/sub-*_ses-*_task-WordStroop_run-*_nirs.snirf"
    elif dataset_name == "Anderson_sparse":
        raw_dir = f"{raw_path}/sub-*/nirs/sub-*_ses-*_task-WordStroop_run-*_nirs.snirf"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    files = glob.glob(raw_dir)
    if dataset_name == "BS_Laura":
        files = [p for p in files if "BS" in os.path.basename(p)]
        files = [p for p in files if "_acq-4NN_nirs" not in os.path.basename(p)]
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a SLURM file list for recon-parameter preprocessing.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. BS_Laura")
    parser.add_argument("--subset", default="full", help="Subset type, e.g. full")
    parser.add_argument("--out", default=None, help="Output text file. Defaults to the corresponding preprocessed folder/recon_param_files.txt")
    args = parser.parse_args()

    files = list_dataset_files(args.dataset)
    if not files:
        raise RuntimeError(f"No files found for dataset: {args.dataset}")

    pre_processed_path = Path(f"datasets/pre_processed/{AUGMENTATION_STRATEGY}/{args.dataset}/{args.subset}")
    pre_processed_path.mkdir(parents=True, exist_ok=True)

    out = Path(args.out) if args.out else pre_processed_path / "recon_param_files.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(files) + "\n")

    print(f"dataset: {args.dataset}")
    print(f"subset: {args.subset}")
    print(f"n_files: {len(files)}")
    print(f"job_array: 0-{len(files) - 1}")
    print(f"file_list: {out}")
    print(f"pre_processed_path: {pre_processed_path}")


if __name__ == "__main__":
    main()
