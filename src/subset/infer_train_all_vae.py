#!/usr/bin/env python3
import glob
import os

import numpy as np
import torch
import xarray as xr

from train_parcel_vae import Conv2D_VAE


def main():
    # Paths: update these to your trained model and input/output dirs.
    model_path = "src/subset/vae_results/subset_2_train_all_Conv2D_VAE/models/train_all_vae.pth"
    stats_path = "src/subset/vae_results/subset_2_train_all_Conv2D_VAE/train_all_stats.npz"
    input_dir = "datasets/full_processed/FreshMotor"
    output_dir = "src/subset/vae_results/subset_2_train_all_Conv2D_VAE/parcel_vae_inference/freshmotor"

    latent_dim = 32
    hidden_dim = 16

    os.makedirs(output_dir, exist_ok=True)
    shared_scale = float(np.load(stats_path)["shared_scale"])

    files = glob.glob(os.path.join(input_dir, "**", "*.nc"), recursive=True)

    sample_da = xr.open_dataarray(files[0])
    parcels, chromos, times = sample_da.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conv2D_VAE(parcels, chromos, times, latent_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        for path in files:
            da = xr.open_dataarray(path) # (parcels, chromos, times)
            x = da.values.astype(np.float32) / shared_scale
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)
            recon, _, _ = model(x_t)
            y = recon.squeeze(0).cpu().numpy() * shared_scale

            out_da = xr.DataArray(
                y,
                dims=("parcel", "chromo", "time"),
                coords={k: da.coords[k] for k in da.coords if k in ("parcel", "chromo", "time")},
            )
            rel_path = os.path.relpath(path, input_dir)
            rel_dir = os.path.dirname(rel_path)
            out_dir = os.path.join(output_dir, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            out_name = os.path.basename(path).replace(".nc", "_recon.nc")
            out_path = os.path.join(out_dir, out_name)
            out_da.to_netcdf(out_path)
    
    print("Inference complete. Results are saved to:", output_dir)


if __name__ == "__main__":
    main()
