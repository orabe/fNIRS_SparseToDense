#!/usr/bin/env python3
import glob
import os
import re
import math
import csv
import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import xarray as xr

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class SegmentKey:
    subject: str
    run: str
    segment: str


def parse_segment_key(path: str) -> SegmentKey:
    """Extract (subject, run, segment) from a filename."""
    name = os.path.basename(path)
    pattern = r"(sub-[^_]+).*?run-([^_]+).*?_(\d+)(?:_test)?\.nc$"
    match = re.search(pattern, name)
    if not match:
        raise ValueError(f"Could not parse subject/run/segment from {name}")
    return SegmentKey(match.group(1), match.group(2), match.group(3))


def list_pairs(sparse_root: str, dense_root: str) -> list[tuple[str, str, SegmentKey]]:
    # Scan both roots and align by subject/run/segment keys.
    sparse_files = glob.glob(os.path.join(sparse_root, "**", "*.nc"), recursive=True)
    dense_files = glob.glob(os.path.join(dense_root, "**", "*.nc"), recursive=True)

    sparse_map = {parse_segment_key(f): f for f in sparse_files}
    dense_map = {parse_segment_key(f): f for f in dense_files}
    keys = sorted(set(sparse_map.keys()) & set(dense_map.keys()), key=lambda k: (k.subject, k.run, k.segment))
    pairs = [(sparse_map[k], dense_map[k], k) for k in keys]
    if not pairs:
        raise RuntimeError("No aligned sparse/dense pairs found.")
    return pairs

class ParcelPairPreloadDataset(Dataset):
    """preloads all data into memory"""
    def __init__(self, pairs: list[tuple[str, str, SegmentKey]], stats: dict | None = None, label: str = "train"):
        self.pairs = pairs
        self.stats = stats
        self.all_x = []
        self.all_y = []

        LOGGER.info("Preloading %d paired segments into memory (%s)...", len(self.pairs), label)
        for idx, (sparse_path, dense_path, _) in enumerate(self.pairs, start=1):
            x = xr.open_dataarray(sparse_path).values.astype(np.float32)
            y = xr.open_dataarray(dense_path).values.astype(np.float32)

            if self.stats is not None:
                # x = (x - self.stats["x_mean"]) / self.stats["x_std"]
                # y = (y - self.stats["y_mean"]) / self.stats["y_std"]
                x = x / self.stats["shared_scale"]
                y = y / self.stats["shared_scale"]
                x = x.astype(np.float32)
                y = y.astype(np.float32)

            self.all_x.append(torch.from_numpy(x))
            self.all_y.append(torch.from_numpy(y))
            if idx % 500 == 0 or idx == len(self.pairs):
                LOGGER.info("preload (%s): %d/%d pairs", label, idx, len(self.pairs))

        if not self.all_x:
            raise ValueError("No samples loaded for preload dataset.")
        self.parcels, self.chromos, self.times = self.all_x[0].shape
        LOGGER.info("Loaded %d pairs into memory.", len(self.all_x))

    def __len__(self) -> int:
        return len(self.all_x)

    def __getitem__(self, idx: int):
        return self.all_x[idx], self.all_y[idx]

    def normalize_inplace(self, shared_scale: float):
        if shared_scale <= 0:
            raise ValueError(f"shared_scale must be > 0, got {shared_scale}")
        for i in range(len(self.all_x)):
            self.all_x[i] = self.all_x[i] / shared_scale
            self.all_y[i] = self.all_y[i] / shared_scale


def compute_shared_max_abs_from_preloaded(train_ds: ParcelPairPreloadDataset) -> float:
    shared_scale = 0.0
    for x in train_ds.all_x:
        shared_scale = max(shared_scale, float(torch.max(torch.abs(x)).item()))
    for y in train_ds.all_y:
        shared_scale = max(shared_scale, float(torch.max(torch.abs(y)).item()))
    if shared_scale < 1e-8:
        raise ValueError(f"scale too small (shared_scale={shared_scale:.3e})")
    return shared_scale


class Conv2D_VAE(nn.Module):
    """
    Conditional Variational Autoencoder.
    2D convolutions over (parcel, time) dimensions.
    Chromophores are treated as channels.
    3x3 kernels with padding.
    """
    def __init__(self, parcels: int, chromos: int, times: int, latent_dim: int, hidden_channels: int):
        super().__init__()
        self.parcels = parcels
        self.chromos = chromos
        self.times = times
        self.hidden_channels = hidden_channels

        # Encoder: 2D convs over (parcel, time) with chromo as channels.
        self.encoder = nn.Sequential(
            nn.Conv2d(chromos, hidden_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
        )
        
        # Flattened size of the conv feature map for the linear latent heads.
        flat_dim = hidden_channels * parcels * times
        
        # Latent mean/logvar for reparameterization.
        self.mu = nn.Linear(flat_dim, latent_dim)
        self.logvar = nn.Linear(flat_dim, latent_dim)
        
        # Decoder: project z to feature map, condition on X, then conv to chromos.
        self.z_to_feat = nn.Linear(latent_dim, flat_dim)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels + chromos, hidden_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, chromos, kernel_size=(3, 3), padding=1),
        )

    def encode(self, x_c: torch.Tensor):
        h = self.encoder(x_c)
        # Flatten conv features to feed the linear latent heads (mu/logvar).
        h_flat = h.view(h.size(0), -1)
        return self.mu(h_flat), self.logvar(h_flat)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, x_c: torch.Tensor):
        feat = self.z_to_feat(z).view(-1, self.hidden_channels, self.parcels, self.times)
        feat = torch.cat([feat, x_c], dim=1)
        return self.decoder(feat)

    def forward(self, x: torch.Tensor):
        # x: [batch, parcel, chromo, time] -> [batch, chromo, parcel, time]
        x_c = x.permute(0, 2, 1, 3)
        mu, logvar = self.encode(x_c)
        z = self.reparameterize(mu, logvar)
        y_c = self.decode(z, x_c)
        
        # Return to [batch, parcel, chromo, time] to match targets.
        y = y_c.permute(0, 2, 1, 3)
        return y, mu, logvar


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float,
):
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl, recon_loss, kl


def update_and_save_plot(fig, ax, train_line, test_line, train_curve, test_curve, epoch, out_path):
    train_line.set_data(range(1, epoch + 1), train_curve)
    if test_curve:
        test_line.set_data(range(1, epoch + 1), test_curve)
    ax.relim()
    ax.autoscale_view()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def create_grid_plot(subjects, train_label, test_label, ylabel):
    n_sub = len(subjects)
    n_cols = min(3, n_sub)
    n_rows = math.ceil(n_sub / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False, sharex=True, sharey=False
    )
    axes_flat = axes.ravel()

    plot_items = {}
    for idx, subject in enumerate(subjects):
        ax = axes_flat[idx]
        train_line, = ax.plot([], [], label=train_label)
        test_line, = ax.plot([], [], label=test_label)
        ax.set_title(subject)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelbottom=True)
        ax.legend()
        plot_items[subject] = (ax, train_line, test_line)

    for idx in range(n_sub, len(axes_flat)):
        axes_flat[idx].axis("off")

    return fig, plot_items


def train_one_epoch(model, loader, optimizer, device, kl_weight):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_steps = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        recon, mu, logvar = model(x)
        loss, recon_loss, kl = vae_loss(recon, y, mu, logvar, kl_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl.item()
        total_steps += 1
    denom = max(total_steps, 1)
    return total_loss / denom, total_recon / denom, total_kl / denom


@torch.no_grad()
def eval_one_epoch(model, loader, device, kl_weight):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_steps = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        recon, mu, logvar = model(x)
        loss, recon_loss, kl = vae_loss(recon, y, mu, logvar, kl_weight)
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl.item()
        total_steps += 1
    denom = max(total_steps, 1)
    return total_loss / denom, total_recon / denom, total_kl / denom


def baseline_mean_mse(train_ds, test_ds):
    y_sum = None
    y_count = 0
    for y in train_ds.all_y:
        if y_sum is None:
            y_sum = y.clone()
        else:
            y_sum += y
        y_count += 1
    mean_y = y_sum / max(y_count, 1)

    mse_sum = 0.0
    for y in test_ds.all_y:
        mse_sum += F.mse_loss(mean_y, y, reduction="mean").item()
    return mse_sum / max(len(test_ds), 1)


def run_loso(
    pairs,
    epochs,
    batch_size,
    latent_dim,
    hidden_dim,
    lr,
    save_dir,
    kl_weight,
    mse_fig_name,
    loss_fig_name,
):
    subjects = sorted({k.subject for _, _, k in pairs})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    model_dir = os.path.join(save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    loss_dir = os.path.join(save_dir, "losses")
    os.makedirs(loss_dir, exist_ok=True)

    mse_fig, mse_plot_items = create_grid_plot(subjects, "train mse", "test mse", "MSE")
    loss_fig, loss_plot_items = create_grid_plot(subjects, "train loss", "test loss", "Loss")
    mse_plot_path = os.path.join(save_dir, mse_fig_name)
    loss_plot_path = os.path.join(save_dir, loss_fig_name)
    baseline_rows = []

    for test_subject in subjects:
        train_pairs = [p for p in pairs if p[2].subject != test_subject]
        test_pairs = [p for p in pairs if p[2].subject == test_subject]
        
        # Preload raw data, then compute a shared scale and normalize in memory.
        train_ds = ParcelPairPreloadDataset(train_pairs, stats=None, label="train")
        test_ds = ParcelPairPreloadDataset(test_pairs, stats=None, label="test")
        shared_scale = compute_shared_max_abs_from_preloaded(train_ds)
        train_ds.normalize_inplace(shared_scale)
        test_ds.normalize_inplace(shared_scale)
        stats = {"shared_scale": shared_scale}
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        baseline_mse = baseline_mean_mse(train_ds, test_ds)
        LOGGER.info("[%s] baseline_mean_mse=%.6f", test_subject, baseline_mse)
        baseline_rows.append({"subject": test_subject, "baseline_mse": baseline_mse})

        model = Conv2D_VAE(
            train_ds.parcels,
            train_ds.chromos,
            train_ds.times,
            latent_dim,
            hidden_dim,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        mse_ax, mse_train_line, mse_test_line = mse_plot_items[test_subject]
        loss_ax, loss_train_line, loss_test_line = loss_plot_items[test_subject]

        train_curve = []
        test_curve = []
        train_loss_curve = []
        test_loss_curve = []
        for epoch in range(1, epochs + 1):
            train_loss, train_mse, train_kl = train_one_epoch(model, train_loader, optimizer, device, kl_weight)
            test_loss, test_mse, test_kl = eval_one_epoch(model, test_loader, device, kl_weight)
            train_curve.append(train_mse)
            test_curve.append(test_mse)
            train_loss_curve.append(train_loss)
            test_loss_curve.append(test_loss)

            if epoch % 1 == 0 or epoch == epochs:
                LOGGER.info(
                    "[%s] epoch %03d train_mse=%.6f test_mse=%.6f train_kl=%.6f test_kl=%.6f",
                    test_subject,
                    epoch,
                    train_mse,
                    test_mse,
                    train_kl,
                    test_kl,
                )
            if epoch % 5 == 0 or epoch == epochs:
                update_and_save_plot(
                    mse_fig,
                    mse_ax,
                    mse_train_line,
                    mse_test_line,
                    train_curve,
                    test_curve,
                    epoch,
                    mse_plot_path,
                )
                update_and_save_plot(
                    loss_fig,
                    loss_ax,
                    loss_train_line,
                    loss_test_line,
                    train_loss_curve,
                    test_loss_curve,
                    epoch,
                    loss_plot_path,
                )

        torch.save(model.state_dict(), os.path.join(model_dir, f"{test_subject}_vae.pth"))
        np.save(os.path.join(loss_dir, f"{test_subject}_train_mse.npy"), np.array(train_curve))
        np.save(os.path.join(loss_dir, f"{test_subject}_test_mse.npy"), np.array(test_curve))
        np.save(os.path.join(loss_dir, f"{test_subject}_train_loss.npy"), np.array(train_loss_curve))
        np.save(os.path.join(loss_dir, f"{test_subject}_test_loss.npy"), np.array(test_loss_curve))

    plt.close(mse_fig)
    plt.close(loss_fig)
    if baseline_rows:
        baseline_path = os.path.join(save_dir, "baseline_mse.csv")
        with open(baseline_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["subject", "baseline_mse"])
            writer.writeheader()
            writer.writerows(baseline_rows)


def run_train_all(
    pairs,
    epochs,
    batch_size,
    latent_dim,
    hidden_dim,
    lr,
    save_dir,
    kl_weight,
    mse_fig_name,
    loss_fig_name,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    model_dir = os.path.join(save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    loss_dir = os.path.join(save_dir, "losses")
    os.makedirs(loss_dir, exist_ok=True)

    # Preload raw data, then compute a shared scale and normalize in memory.
    train_ds = ParcelPairPreloadDataset(pairs, stats=None, label="train_all")
    shared_scale = compute_shared_max_abs_from_preloaded(train_ds)
    train_ds.normalize_inplace(shared_scale)
    stats_path = os.path.join(save_dir, "train_all_stats.npz")
    np.savez(stats_path, shared_scale=shared_scale)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = Conv2D_VAE(
        train_ds.parcels,
        train_ds.chromos,
        train_ds.times,
        latent_dim,
        hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mse_fig, mse_plot_items = create_grid_plot(["train_all"], "train mse", "test mse", "MSE")
    loss_fig, loss_plot_items = create_grid_plot(["train_all"], "train loss", "test loss", "Loss")
    mse_plot_path = os.path.join(save_dir, mse_fig_name)
    loss_plot_path = os.path.join(save_dir, loss_fig_name)
    mse_ax, mse_train_line, mse_test_line = mse_plot_items["train_all"]
    loss_ax, loss_train_line, loss_test_line = loss_plot_items["train_all"]

    train_curve = []
    train_loss_curve = []
    for epoch in range(1, epochs + 1):
        train_loss, train_mse, train_kl = train_one_epoch(model, train_loader, optimizer, device, kl_weight)
        train_curve.append(train_mse)
        train_loss_curve.append(train_loss)
        if epoch % 1 == 0 or epoch == epochs:
            LOGGER.info(
                "[train_all] epoch %03d train_mse=%.6f train_kl=%.6f",
                epoch,
                train_mse,
                train_kl,
            )
        if epoch % 5 == 0 or epoch == epochs:
            update_and_save_plot(
                mse_fig,
                mse_ax,
                mse_train_line,
                mse_test_line,
                train_curve,
                [],
                epoch,
                mse_plot_path,
            )
            update_and_save_plot(
                loss_fig,
                loss_ax,
                loss_train_line,
                loss_test_line,
                train_loss_curve,
                [],
                epoch,
                loss_plot_path,
            )

    torch.save(model.state_dict(), os.path.join(model_dir, "train_all_vae.pth"))
    np.save(os.path.join(loss_dir, "train_all_train_mse.npy"), np.array(train_curve))
    np.save(os.path.join(loss_dir, "train_all_train_loss.npy"), np.array(train_loss_curve))
    plt.close(mse_fig)
    plt.close(loss_fig)


def main():
    subset_type = "subset_2"
    sparse_root = "datasets/full_processed/BallSqueezingHD_modified"
    dense_root = f"datasets/{subset_type}_processed/BallSqueezingHD_modified"
    
    # Set to False to run LOSO, or
    # set to True to train on all data (for final model -> inference)
    train_all = True
    
    epochs = 20
    batch_size = 16
    latent_dim = 32
    hidden_dim = 16
    lr = 3e-4
    kl_weight = 1e-3
    

    if train_all:
        save_dir = f"src/subset/vae_results/{subset_type}_train_all_Conv2D_VAE"
        train_all_mse_fig = f"{subset_type}_train_all_mse.png"
        train_all_loss_fig = f"{subset_type}_train_all_loss.png"
    else:
        save_dir = f"src/subset/vae_results/{subset_type}_loso_Conv2D_VAE"
        loso_mse_fig = f"{subset_type}_loso_mse.png"
        loso_loss_fig = f"{subset_type}_loso_loss.png"

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    pairs = list_pairs(sparse_root, dense_root)
    if train_all:
        run_train_all(
            pairs,
            epochs=epochs,
            batch_size=batch_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            save_dir=save_dir,
            kl_weight=kl_weight,
            mse_fig_name=train_all_mse_fig,
            loss_fig_name=train_all_loss_fig,
        )
    else:
        run_loso(
            pairs,
            epochs=epochs,
            batch_size=batch_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            save_dir=save_dir,
            kl_weight=kl_weight,
            mse_fig_name=loso_mse_fig,
            loss_fig_name=loso_loss_fig,
        )


if __name__ == "__main__":
    main()
