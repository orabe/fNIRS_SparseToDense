from pathlib import Path

import cedalion
import cedalion.sigproc.quality as quality
import pandas as pd

from cedalion import units


CONFIG = {
    "datasets": [
        "BallSqueezingHD_modified",
        "BS_Laura",
        "vfc_hd",
    ],
    "raw_root": Path("datasets/raw"),
    "output_root": Path("datasets/pre_processed/qc"),
    "sci_threshold": 0.6,
    "psp_threshold": 0.1,
    "window_length_seconds": 5,
    "min_clean_time_fraction": 0.5,
    "source_detector_distance_cm": [1, 4.5],
    "subject_clean_channel_thresholds": [0.4, 0.6],
}


def find_recordings(dataset_name):
    """Return the full-montage task recordings used by the processing pipeline."""
    dataset_root = CONFIG["raw_root"] / dataset_name
    if dataset_name == "BallSqueezingHD_modified":
        files = dataset_root.glob("sub-*/nirs/sub-*.snirf")
    elif dataset_name == "BS_Laura":
        files = dataset_root.glob("sub-*/nirs/sub-*.snirf")
        files = (
            path
            for path in files
            if "BS" in path.name and "_acq-4NN_nirs" not in path.name
        )
    elif dataset_name == "vfc_hd":
        files = dataset_root.glob(
            "sub-*/nirs/sub-*_ses-*_task-WordStroop_run-*_nirs.snirf"
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return sorted(files)


def compute_dataset_qc(dataset_name):
    """Compute full-montage recording-, channel-, and subject-level SCI/PSP QC."""
    files = find_recordings(dataset_name)
    if not files:
        raise ValueError(f"No raw recordings found for {dataset_name}")

    channel_rows = []
    window_length = CONFIG["window_length_seconds"] * units.s
    sd_range = CONFIG["source_detector_distance_cm"] * units.cm

    print(f"\n{dataset_name}: processing {len(files)} recordings")
    for file_index, file_path in enumerate(files, start=1):
        print(f"[{file_index}/{len(files)}] {file_path}")
        subject_id = next(part for part in file_path.parts if part.startswith("sub-"))
        rec = cedalion.io.read_snirf(file_path)[0]
        amplitudes = rec["amp"]

        _, distance_mask = quality.sd_dist(amplitudes, rec.geo3d, sd_range)
        amplitudes, _ = quality.prune_ch(amplitudes, [distance_mask], "all")

        sci, sci_mask = quality.sci(
            amplitudes,
            window_length,
            CONFIG["sci_threshold"],
        )
        psp, psp_mask = quality.psp(
            amplitudes,
            window_length,
            CONFIG["psp_threshold"],
        )
        combined_mask = sci_mask & psp_mask
        n_windows = combined_mask.sizes["time"]

        for channel in combined_mask.channel.values:
            channel_sci_mask = sci_mask.sel(channel=channel)
            channel_psp_mask = psp_mask.sel(channel=channel)
            channel_combined_mask = combined_mask.sel(channel=channel)
            n_sci_clean = int(channel_sci_mask.sum("time").item())
            n_psp_clean = int(channel_psp_mask.sum("time").item())
            n_combined_clean = int(channel_combined_mask.sum("time").item())
            clean_time_fraction = n_combined_clean / n_windows

            channel_rows.append(
                {
                    "dataset": dataset_name,
                    "subject_id": subject_id,
                    "recording": file_path.stem,
                    "snirf_file": str(file_path),
                    "channel": str(channel),
                    "n_windows": n_windows,
                    "n_sci_clean_windows": n_sci_clean,
                    "n_psp_clean_windows": n_psp_clean,
                    "n_combined_clean_windows": n_combined_clean,
                    "sci_clean_time_fraction": n_sci_clean / n_windows,
                    "psp_clean_time_fraction": n_psp_clean / n_windows,
                    "combined_clean_time_fraction": clean_time_fraction,
                    "mean_sci": float(sci.sel(channel=channel).mean("time").item()),
                    "mean_psp": float(psp.sel(channel=channel).mean("time").item()),
                    "channel_clean": clean_time_fraction
                    >= CONFIG["min_clean_time_fraction"],
                }
            )

    recording_channel_df = pd.DataFrame(channel_rows)
    subject_channel_df = (
        recording_channel_df.groupby(
            ["dataset", "subject_id", "channel"],
            as_index=False,
        )
        .agg(
            n_recordings=("recording", "nunique"),
            n_windows=("n_windows", "sum"),
            n_sci_clean_windows=("n_sci_clean_windows", "sum"),
            n_psp_clean_windows=("n_psp_clean_windows", "sum"),
            n_combined_clean_windows=("n_combined_clean_windows", "sum"),
        )
    )
    subject_channel_df["sci_clean_time_fraction"] = (
        subject_channel_df["n_sci_clean_windows"] / subject_channel_df["n_windows"]
    )
    subject_channel_df["psp_clean_time_fraction"] = (
        subject_channel_df["n_psp_clean_windows"] / subject_channel_df["n_windows"]
    )
    subject_channel_df["combined_clean_time_fraction"] = (
        subject_channel_df["n_combined_clean_windows"]
        / subject_channel_df["n_windows"]
    )
    subject_channel_df["channel_clean"] = (
        subject_channel_df["combined_clean_time_fraction"]
        >= CONFIG["min_clean_time_fraction"]
    )

    recording_df = (
        recording_channel_df.groupby(
            ["dataset", "subject_id", "recording", "snirf_file"],
            as_index=False,
        )
        .agg(
            n_channels=("channel", "nunique"),
            n_clean_channels=("channel_clean", "sum"),
        )
    )
    recording_df["clean_channel_fraction"] = (
        recording_df["n_clean_channels"] / recording_df["n_channels"]
    )

    subject_df = (
        subject_channel_df.groupby(["dataset", "subject_id"], as_index=False)
        .agg(
            n_channels=("channel", "nunique"),
            n_clean_channels=("channel_clean", "sum"),
        )
    )
    recording_counts = (
        recording_channel_df.groupby(["dataset", "subject_id"])["recording"]
        .nunique()
        .rename("n_recordings")
        .reset_index()
    )
    subject_df = subject_df.merge(
        recording_counts,
        on=["dataset", "subject_id"],
        validate="one_to_one",
    )
    subject_df["clean_channel_fraction"] = (
        subject_df["n_clean_channels"] / subject_df["n_channels"]
    )

    for threshold in CONFIG["subject_clean_channel_thresholds"]:
        threshold_percent = int(round(threshold * 100))
        recording_df[f"passes_{threshold_percent}_percent"] = (
            recording_df["clean_channel_fraction"] >= threshold
        )
        subject_df[f"passes_{threshold_percent}_percent"] = (
            subject_df["clean_channel_fraction"] >= threshold
        )

    output_dir = CONFIG["output_root"] / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    recording_channel_df.to_csv(
        output_dir / "recording_channel_qc.csv",
        index=False,
    )
    subject_channel_df.to_csv(
        output_dir / "subject_channel_qc.csv",
        index=False,
    )
    recording_df.to_csv(output_dir / "recording_qc.csv", index=False)
    subject_df.to_csv(output_dir / "subject_qc.csv", index=False)

    print(f"Saved QC outputs to {output_dir}")
    return subject_df


def main():
    """Run full-montage SCI/PSP QC for all configured dense datasets."""
    CONFIG["output_root"].mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "datasets": ",".join(CONFIG["datasets"]),
                "sci_threshold": CONFIG["sci_threshold"],
                "psp_threshold": CONFIG["psp_threshold"],
                "window_length_seconds": CONFIG["window_length_seconds"],
                "min_clean_time_fraction": CONFIG["min_clean_time_fraction"],
                "source_detector_distance_cm": "-".join(
                    str(value) for value in CONFIG["source_detector_distance_cm"]
                ),
                "subject_clean_channel_thresholds": ",".join(
                    str(value)
                    for value in CONFIG["subject_clean_channel_thresholds"]
                ),
            }
        ]
    ).to_csv(CONFIG["output_root"] / "qc_config.csv", index=False)
    subject_tables = [compute_dataset_qc(name) for name in CONFIG["datasets"]]
    combined_subject_df = pd.concat(subject_tables, ignore_index=True)
    combined_subject_df.to_csv(
        CONFIG["output_root"] / "all_datasets_subject_qc.csv",
        index=False,
    )
    print(f"\nSaved combined subject QC to {CONFIG['output_root']}")


if __name__ == "__main__":
    main()
