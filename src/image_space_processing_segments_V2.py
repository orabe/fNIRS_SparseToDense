09 
import torch
import pickle
import cedalion
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cedalion.sigproc.motion_correct as motion_correct
import cedalion.sigproc.quality as quality
import cedalion.sigproc.physio as physio
from cedalion.io.forward_model import load_Adot,save_Adot
import cedalion.dot as dot
from cedalion import units

import cedalion.nirs as nirs
import os
import xarray as xr
import glob
import warnings
warnings.filterwarnings("ignore")


from scipy.interpolate import interp1d


def get_bad_ch_mask(int_data):
    # Saturated and Dark Channels

    dark_sat_thresh = [1e-3, 0.84]
    amp_threshs_sat = [0., dark_sat_thresh[1]]
    amp_threshs_low = [dark_sat_thresh[0], 1]
    _, amp_mask_sat = quality.mean_amp(int_data, amp_threshs_sat)
    _, amp_mask_low = quality.mean_amp(int_data, amp_threshs_low)
    _, snr_mask = quality.snr(int_data, 10)
    amp_mask=amp_mask_sat & amp_mask_low

    _, list_bad_ch = quality.prune_ch(int_data, [amp_mask, snr_mask], "all")
   
    return list_bad_ch

base_dir = "/home/"
# processed_data = os.path.join(base_dir, "data/yuanyuan_v2_processed_partial") ## Yuanyuan Dataset
processed_data = os.path.join(base_dir, "data/combined_fnirs_partial_head") ## Laura Dataset
if not os.path.exists(processed_data):
    os.makedirs(processed_data)
dataset_path = os.path.join(base_dir, "data/BallSqueezingHD_modified") ## Yuanyuan Dataset
# dataset_path = os.path.join(base_dir, "data/BS_Laura") ## Laura Dataset

sensitivity_fname = os.path.join("/data/sensitivity_yuanyuan_bs_v2.h5") ## Yuanyuan sensitivity
# sensitivity_fname = os.path.join("/data/sensitivity_laura_bs_v2.h5") ## Laura sensitivity

sensitive_parcels_path = os.path.join(base_dir, "data/sensitive_parcels_fh.pkl")
keep_parcels_path = os.path.join(base_dir, "data/sensitive_parcels.pkl")

# # Get all snirf files
import sys
sub = sys.argv[1]

print("processing subject: ", sub)
all_files = glob.glob(os.path.join(dataset_path, sub) + "/**/*.snirf", recursive=True)

with open(sensitive_parcels_path, "rb") as f:
    sensitive_parcels = pickle.load(f)
with open(keep_parcels_path, "rb") as f:
    keep_parcels = pickle.load(f)

Adot = load_Adot(sensitivity_fname)
recon = dot.ImageRecon(
    Adot,
    recon_mode="mua2conc",
    brain_only=True,
    alpha_meas=10,
    alpha_spatial=10e-3,
    apply_c_meas=True,
    spatial_basis_functions=None,
)

for file in all_files:
    records = cedalion.io.read_snirf(file)
    rec = records[0]

    rec.stim = rec.stim.sort_values(by="onset") ## Yuanyuan dataset

    rec['rep_amp'] = quality.repair_amp(rec['amp'], median_len=3, method='linear')  # Repair Amp
    rec['od_amp'], baseline= nirs.cw.int2od(rec['rep_amp'],return_baseline=True)

    # motion correct [TDDR + WAVELET]
    rec["od_tddr"] = motion_correct.tddr(rec["od_amp"])
    rec["od_tddr_wavel"] = motion_correct.wavelet(rec["od_tddr"])

    #-----------------------------------------highpass filter--------------------------------
    rec['od_hpfilt'] = rec['od_tddr_wavel'].cd.freq_filter(fmin=0.008,fmax=0,butter_order=4)
    #----------------------------------------------------------------------------------------

    # clean amplitude data
    rec['amp_clean'] = cedalion.nirs.cw.od2int(rec['od_hpfilt'], baseline)

    # get bad channel mask
    list_bad_ch = get_bad_ch_mask(rec["amp_clean"]) # this has custom paramerers!? 
    print('the list of bad channels: ', len(list_bad_ch))

    # channel variance
    od_var_vec = quality.measurement_variance(rec["od_hpfilt"], list_bad_channels=list_bad_ch, bad_rel_var=1e6,calc_covariance=False)

    #---------------------------------------------------------------------------------------
    dpf = xr.DataArray(
        [6, 6],
        dims="wavelength",
        coords={"wavelength": rec["amp"].wavelength},
    )
    rec['conc'] = cedalion.nirs.cw.od2conc(rec['od_hpfilt'], rec.geo3d, dpf, spectrum="prahl")

    # conc_pr vs conc 
    chromo_var = quality.measurement_variance(rec['conc'], list_bad_channels = list_bad_ch, bad_rel_var = 1e6, calc_covariance = False)
    rec['conc_pcr'], gb_comp_rem = physio.global_component_subtract(rec['conc'],ts_weights=1/chromo_var,k=0,spatial_dim='channel',spectral_dim='chromo')

    rec['od_pcr1'] = cedalion.nirs.cw.conc2od(rec['conc_pcr'], rec.geo3d, dpf, spectrum="prahl")#     delta_conc = chunked_eff_xr_matmult(od_stacked, B, contract_dim="flat_channel", sample_dim="time", chunksize=300)
    c_meas = quality.measurement_variance(rec['od_hpfilt'], list_bad_channels=list_bad_ch, bad_rel_var=1e6,calc_covariance=False)

    delta_conc = recon.reconstruct(rec['od_pcr1'], c_meas) 
    delta_conc.time.attrs["units"] = units.s

    dC_brain = delta_conc.cd.freq_filter(fmin=0.01, fmax=0.5, butter_order=4)
    dC_brain = dC_brain.sel(time=slice(rec.stim.onset.values[0]-3 , rec.stim.onset.values[-1]+13))
    dC_brain = dC_brain.where(dC_brain.is_brain == True)
    # alternatively use 1/conc_var to weight vertex sensitivity and then normalize by sum of weights
    dC_brain = dC_brain.pint.quantify().pint.to("uM").pint.dequantify()

    hbr = dC_brain.sel(chromo='HbR').groupby('parcel').mean()
    hbo = dC_brain.sel(chromo='HbO').groupby('parcel').mean()
    signal_raw = xr.concat([hbo, hbr], dim='chromo')

    # revised matrix
    signal_raw = signal_raw.sel(parcel=signal_raw.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_LH')
    signal_raw = signal_raw.sel(parcel=signal_raw.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_RH')
    
    delta_conc, global_comp = physio.global_component_subtract(signal_raw, ts_weights=None, k=0, 
                                                        spatial_dim='parcel', spectral_dim= 'chromo')

    delta_conc = delta_conc / np.abs(delta_conc).max()
    delta_conc = delta_conc.fillna(0)
    delta_conc = delta_conc.transpose("time", "parcel", "chromo")
    delta_brain = delta_conc.copy()

    # create a boolean mask along 'parcel' dimension
    delta_brain = delta_brain.sel(parcel=keep_parcels)


    i = 0
    for index, row in rec.stim.iterrows():
        label = row["trial_type"]
        duration = 10
        start_list = np.linspace(-2.5, 2.5, 9)
        for s in start_list:
            start_time = row["onset"] + s
            end_time = start_time + duration + 5   # in seconds
            baseline = delta_brain.sel(time=slice(row["onset"] - 2.5 , row["onset"])).mean("time")
            # Then, trimming is easy with `.sel()`:
            x = delta_brain.sel(time=slice(start_time, end_time)) - baseline
            x = x.isel(time=slice(0, 87))
            x = x.transpose("parcel", "chromo", "time")
            del x.time.attrs['units']
            if not os.path.exists(os.path.dirname(file.replace(dataset_path, processed_data))):
                os.makedirs(os.path.dirname(file.replace(dataset_path, processed_data)))
            if s == 0:
                x.to_netcdf(file.replace(dataset_path, processed_data).replace(".snirf", "_" + label + "_"+str(i)+"_test.nc"))
                i += 1
            else:
                x.to_netcdf(file.replace(dataset_path, processed_data).replace(".snirf", "_" + label + "_"+str(i)+".nc"))
                i += 1
    print("finished processing file: ", os.path.basename(file).replace(".snirf",".npy"))