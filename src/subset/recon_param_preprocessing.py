from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
from pathlib import Path, PureWindowsPath
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

import cedalion
import cedalion.dot as dot
import cedalion.dot.image_recon as dot_image_recon
import cedalion.nirs as nirs
import cedalion.sigproc.motion as motion_correct
import cedalion.sigproc.physio as physio
import cedalion.sigproc.quality as quality
from cedalion import units
from cedalion.io import read_events_from_tsv
from cedalion.io.forward_model import load_Adot

AUGMENTATION_STRATEGY = "imageRecon_params"
K_ALPHA_MEAS = 0.01
ALPHA_MEAS_MULTIPLIERS = [0.1, 1.0, 10.0]
ALPHA_SPATIAL_MULTIPLIERS = [0.1, 1.0, 10.0]
BASELINE_ALPHA_SPATIAL = 1e-2

BS_LAURA_TRANSFORM = np.array(
    [
        [-9.57882733e-01, -7.20806358e-03, 6.20193531e-03, 2.21208571e02],
        [-2.02271710e-02, 6.03819925e-02, 9.94046165e-01, -2.03010603e01],
        [-8.79481533e-03, -1.02761992e00, 6.59199998e-02, 2.87749135e02],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def get_paths(dataset_name: str, subset_type: str) -> tuple[Path, Path]:
    raw_path = Path(f"datasets/raw/{dataset_name}")
    pre_processed_path = Path(
        f"datasets/pre_processed/{AUGMENTATION_STRATEGY}/{dataset_name}/{subset_type}"
    )
    return raw_path, pre_processed_path


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


def get_bad_ch_mask(int_data, ch_preproc) -> list:
    dark_sat_thresh = [1e-3, 0.84]
    amp_threshs_sat = [0.0, dark_sat_thresh[1]]
    amp_threshs_low = [dark_sat_thresh[0], 1]
    _, amp_mask_sat = quality.mean_amp(int_data, amp_threshs_sat)
    _, amp_mask_low = quality.mean_amp(int_data, amp_threshs_low)
    _, snr_mask = quality.snr(int_data, 10)
    amp_mask = amp_mask_sat & amp_mask_low
    _, list_bad_ch = quality.prune_ch(int_data, [amp_mask, snr_mask], "all")
    return list_bad_ch


def standardize_trial_types(dataset_name: str, file: str, stim: pd.DataFrame, rec):
    if dataset_name == "FreshMotor":
        m = re.search(r"(?i)(left|right)", file)
        rec.stim.trial_type = m.group(1).lower()
    elif dataset_name in ["BallSqueezingHD", "BallSqueezingHD_modified"]:
        mapping = {"Right": "right", "Left": "left"}
        rec.stim["trial_type"] = rec.stim["trial_type"].replace(mapping)
    elif dataset_name == "BS_Laura":
        stim = stim.copy()
        stim["duration"] = 10.0
        rec.stim = stim
    elif dataset_name == "Electrical_Thermal":
        mapping = {"1": "WordCongruent", "2": "WordIncongruent"}
        rec.stim["trial_type"] = rec.stim["trial_type"].replace(mapping)
    elif dataset_name in ["vfc_hd", "Anderson_sparse"]:
        mapping = {"1": "WordCongruent", "2": "WordIncongruent"}
        rec.stim["trial_type"] = rec.stim["trial_type"].replace(mapping)

    rec.stim.sort_values(by="onset", ignore_index=True, inplace=True)
    return stim, rec


def get_param_config_folder(alpha_meas_multiplier: float, alpha_spatial_multiplier: float) -> str:
    return f"am_{float(alpha_meas_multiplier):g}__as_{float(alpha_spatial_multiplier):g}"


def estimate_alpha_meas(c_meas, k_alpha_meas: float = K_ALPHA_MEAS) -> float:
    try:
        return float(dot_image_recon.estimate_alpha_meas(c_meas, K=k_alpha_meas))
    except AttributeError:
        return float(k_alpha_meas / np.median(c_meas.values))


def get_recon_settings(c_meas) -> list[tuple]:
    alpha_meas_0 = estimate_alpha_meas(c_meas)
    alpha_spatial_0 = float(BASELINE_ALPHA_SPATIAL)
    settings = []
    for m_meas in ALPHA_MEAS_MULTIPLIERS:
        for m_spatial in ALPHA_SPATIAL_MULTIPLIERS:
            settings.append(
                (
                    get_param_config_folder(m_meas, m_spatial),
                    float(alpha_meas_0 * m_meas),
                    float(alpha_spatial_0 * m_spatial),
                    float(alpha_meas_0),
                    alpha_spatial_0,
                    float(m_meas),
                    float(m_spatial),
                )
            )
    return settings


def get_subset_channels(dataset_name: str, subset_type: str, files: Iterable[str]) -> list:
    if subset_type.startswith("motor_") and dataset_name == "BS_Laura":
        n_chs = subset_type.replace("motor_", "").replace("chs", "")
        sparsified_data_path = (
            "datasets/pre_processed/BS_Laura/"
            f"channel_subset_BS_Laura_k=2_c3c4k=51_dist4.5_{n_chs}chs.npy"
        )
        return np.load(sparsified_data_path, allow_pickle=True).tolist()

    first_file = next(iter(files))
    rec = cedalion.io.read_snirf(first_file)[0]
    return rec["amp"]["channel"].values.tolist()


def make_forward_model(dataset_name: str, subset_channels: list, files: list[str], pre_processed_path: Path):
    first_file = files[0]
    rec = cedalion.io.read_snirf(first_file)[0]
    rec["amp"] = rec["amp"].sel(channel=subset_channels)

    meas_list = rec._measurement_lists["amp"]
    meas_list = meas_list[meas_list["channel"].isin(subset_channels)].reset_index(drop=True)

    head_icbm152 = dot.get_standard_headmodel("icbm152")
    if dataset_name == "BS_Laura":
        ninja_aligned = rec.geo3d.points.apply_transform(BS_LAURA_TRANSFORM)
        geo3d_snapped_ijk = head_icbm152.align_and_snap_to_scalp(ninja_aligned)
    else:
        geo3d_snapped_ijk = head_icbm152.align_and_snap_to_scalp(rec.geo3d)

    fwm = cedalion.dot.forward_model.ForwardModel(head_icbm152, geo3d_snapped_ijk, meas_list)
    sensitivity_fname = pre_processed_path / f"sensitivity_{dataset_name}.h5"
    if not sensitivity_fname.exists():
        raise FileNotFoundError(
            f"Missing sensitivity file: {sensitivity_fname}. Run the notebook forward-model setup first."
        )
    adot = load_Adot(str(sensitivity_fname))
    return fwm, adot


def metadata_path(pre_processed_path: Path, file: str) -> Path:
    path = PureWindowsPath(file)
    subject_dir = path.parts[-3]
    filename = path.stem
    if "ses-" in path.parts[-3]:
        subject_dir = path.parts[-4]
    return pre_processed_path / "_job_metadata" / f"{subject_dir}__{filename}.pkl"


def write_metadata(pre_processed_path: Path, file: str, metadata: dict) -> None:
    out = metadata_path(pre_processed_path, file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as handle:
        pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)


def process_file(dataset_name: str, subset_type: str, file: str) -> dict:
    all_files = list_dataset_files(dataset_name)
    if file not in all_files:
        raise ValueError(f"File is not part of dataset file list after notebook filters: {file}")

    _, pre_processed_path = get_paths(dataset_name, subset_type)
    pre_processed_path.mkdir(parents=True, exist_ok=True)

    if dataset_name == "vfc_hd":
        subj = PureWindowsPath(file).parts[-3]
        if subj == "sub-13":
            metadata = {
                "file": file,
                "status": "skipped",
                "skip_reason": "hard-coded bad subject exclusion for vfc_hd sub-13",
            }
            write_metadata(pre_processed_path, file, metadata)
            return metadata

    subset_channels = get_subset_channels(dataset_name, subset_type, all_files)
    fwm, adot = make_forward_model(dataset_name, subset_channels, all_files, pre_processed_path)

    records = cedalion.io.read_snirf(file)
    rec = records[0]
    rec["amp"] = rec["amp"].sel(channel=subset_channels)

    stim = read_events_from_tsv(file.replace("nirs.snirf", "events.tsv"))
    rec.stim = rec.stim.sort_values(by="onset")
    stim, rec = standardize_trial_types(dataset_name, file, stim, rec)

    rec["rep_amp"] = quality.repair_amp(rec["amp"], median_len=3, method="linear")
    rec["od_amp"], baseline = nirs.cw.int2od(rec["rep_amp"], return_baseline=True)
    rec["od_tddr"] = motion_correct.tddr(rec["od_amp"])
    rec["od_tddr_wavel"] = motion_correct.wavelet(rec["od_tddr"])
    rec["od_hpfilt"] = rec["od_tddr_wavel"].cd.freq_filter(fmin=0.008, fmax=0, butter_order=4)
    rec["amp_clean"] = cedalion.nirs.cw.od2int(rec["od_hpfilt"], baseline)

    ch_preproc = {
        "sci_thresh": 0.5,
        "psp_thresh": 0.1,
        "window_len": 5 * units.s,
        "dark_sat_thresh": [1e-3, 0.84],
        "perc_time_clean": 0.5,
    }
    list_bad_ch = get_bad_ch_mask(rec["amp_clean"], ch_preproc)
    print("the list of bad channels:", len(list_bad_ch))

    dpf = xr.DataArray([6, 6], dims="wavelength", coords={"wavelength": rec["amp"].wavelength})
    rec["conc"] = cedalion.nirs.cw.od2conc(rec["od_hpfilt"], rec.geo3d, dpf, spectrum="prahl")

    try:
        chromo_var = quality.measurement_variance(
            rec["conc"],
            list_bad_channels=list_bad_ch,
            bad_rel_var=1e6,
            calc_covariance=False,
        )
    except Exception as exc:
        metadata = {"file": file, "status": "skipped", "skip_reason": f"chromo_var error: {exc}"}
        write_metadata(pre_processed_path, file, metadata)
        return metadata

    rec["conc_pcr"], _ = physio.global_component_subtract(
        rec["conc"],
        ts_weights=1 / chromo_var,
        k=0,
        spatial_dim="channel",
        spectral_dim="chromo",
    )
    rec["od_pcr1"] = cedalion.nirs.cw.conc2od(rec["conc_pcr"], rec.geo3d, dpf, spectrum="prahl")

    c_meas = quality.measurement_variance(
        rec["od_hpfilt"],
        list_bad_channels=list_bad_ch,
        bad_rel_var=1e6,
        calc_covariance=False,
    )
    recon_settings = get_recon_settings(c_meas)

    _, parcel_mask = fwm.parcel_sensitivity(
        adot,
        [],
        dOD_thresh=0.001,
        minCh=1,
        dHbO=10,
        dHbR=-3,
    )
    sensitive_parcels = parcel_mask.where(parcel_mask, drop=True)["parcel"].values.tolist()
    dropped_parcels = parcel_mask.where(~parcel_mask, drop=True)["parcel"].values.tolist()

    total_parcels = len(sensitive_parcels) + len(dropped_parcels)
    dropped_ratio = (len(dropped_parcels) / total_parcels) if total_parcels else 1.0
    if dropped_ratio > 0.99:
        metadata = {
            "file": file,
            "status": "skipped",
            "skip_reason": f"dropped parcel ratio > 0.99: {len(dropped_parcels)} / {total_parcels}",
            "sensitive_parcels": sensitive_parcels,
            "dropped_parcels": dropped_parcels,
        }
        write_metadata(pre_processed_path, file, metadata)
        return metadata

    path = PureWindowsPath(file)
    subject_dir = path.parts[-3]
    filename = path.stem
    if dataset_name == "FreshMotor":
        subject_dir = path.parts[-4]
        session_label = path.parts[-3]
        task_fragment = next(
            (part for part in filename.split("_") if part.startswith("task-")),
            f"task-{dataset_name.replace('_', '').upper()}",
        )
        run_fragment = session_label.replace("ses-", "run-")
        filename = f"{subject_dir}_{task_fragment}_{run_fragment}_nirs"

    saved_files = []
    for view_id, alpha_meas, alpha_spatial, alpha_meas_0, alpha_spatial_0, m_meas, m_spatial in recon_settings:
        recon = dot.ImageRecon(
            adot,
            recon_mode="mua2conc",
            brain_only=True,
            alpha_meas=alpha_meas,
            alpha_spatial=alpha_spatial,
            apply_c_meas=True,
            spatial_basis_functions=None,
        )

        delta_conc_view = recon.reconstruct(rec["od_pcr1"], c_meas)
        delta_conc_view.time.attrs["units"] = units.s
        dC_brain = delta_conc_view.cd.freq_filter(fmin=0.01, fmax=0.5, butter_order=4)
        dC_brain = dC_brain.sel(time=slice(rec.stim.onset.values[0] - 3, rec.stim.onset.values[-1] + 13))
        dC_brain = dC_brain.where(dC_brain.is_brain == True)
        dC_brain = dC_brain.pint.quantify().pint.to("uM").pint.dequantify()

        hbr = dC_brain.sel(chromo="HbR").groupby("parcel").mean()
        hbo = dC_brain.sel(chromo="HbO").groupby("parcel").mean()
        signal_raw = xr.concat([hbo, hbr], dim="chromo")
        signal_raw = signal_raw.sel(parcel=signal_raw.parcel != "Background+FreeSurfer_Defined_Medial_Wall_LH")
        signal_raw = signal_raw.sel(parcel=signal_raw.parcel != "Background+FreeSurfer_Defined_Medial_Wall_RH")

        delta_conc, _ = physio.global_component_subtract(
            signal_raw,
            ts_weights=None,
            k=0,
            spatial_dim="parcel",
            spectral_dim="chromo",
        )
        delta_conc = delta_conc / np.abs(delta_conc).max()
        delta_conc = delta_conc.fillna(0)
        delta_conc = delta_conc.transpose("time", "parcel", "chromo")

        data = {
            "conc_pcr": rec["conc_pcr"],
            "delta_conc": delta_conc,
            "rec_stim": rec.stim,
            "sensitive_parcels": sensitive_parcels,
            "view_id": view_id,
            "alpha_meas": float(alpha_meas),
            "alpha_spatial": float(alpha_spatial),
            "alpha_meas_0": float(alpha_meas_0),
            "alpha_spatial_0": float(alpha_spatial_0),
            "alpha_meas_multiplier": float(m_meas),
            "alpha_spatial_multiplier": float(m_spatial),
            "k_alpha_meas": float(K_ALPHA_MEAS),
        }

        config_folder = get_param_config_folder(m_meas, m_spatial)
        out_dir = pre_processed_path / config_folder / subject_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{filename}.pkl"
        with open(out_file, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        saved_files.append(str(out_file))

    metadata = {
        "file": file,
        "subject": subject_dir,
        "filename": filename,
        "status": "ok",
        "skip_reason": None,
        "sensitive_parcels": sensitive_parcels,
        "dropped_parcels": dropped_parcels,
        "saved_files": saved_files,
        "n_saved_files": len(saved_files),
    }
    write_metadata(pre_processed_path, file, metadata)
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run image-recon parameter preprocessing for one fNIRS file.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. BS_Laura")
    parser.add_argument("--subset", default="full", help="Subset type, e.g. full")
    parser.add_argument("--file", required=True, help="Raw .snirf file to process")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        meta = process_file(args.dataset, args.subset, args.file)
    except Exception as exc:
        _, pre_processed_path = get_paths(args.dataset, args.subset)
        pre_processed_path.mkdir(parents=True, exist_ok=True)
        meta = {
            "file": args.file,
            "status": "failed",
            "skip_reason": repr(exc),
        }
        write_metadata(pre_processed_path, args.file, meta)
        raise
    finally:
        if "meta" in locals():
            print(f"status: {meta['status']}")
            print(f"file: {meta['file']}")
            if meta.get("skip_reason"):
                print(f"skip_reason: {meta['skip_reason']}")
            print(f"n_saved_files: {meta.get('n_saved_files', 0)}")


if __name__ == "__main__":
    main()
