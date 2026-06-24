import numpy as np
import torch


ONLINE_AUGMENTATION_GROUPS = {
    "time_domain": [
        "gaussian_noise", # 
        "smooth_time_mask", # 
        "time_reverse", # 
        "sign_flip", # 
    ],
    "frequency_domain": [
        "ft_surrogate", #
        "frequency_shift", #
        "bandstop_filter", # 
    ],
    "spatial_domain": [
        "space_symmetry", # 
        "space_dropout", # 
        "space_shuffle", # 
    ],
}

ONLINE_AUGMENTATIONS = [
    "none",
    *ONLINE_AUGMENTATION_GROUPS["time_domain"],
    *ONLINE_AUGMENTATION_GROUPS["frequency_domain"],
    *ONLINE_AUGMENTATION_GROUPS["spatial_domain"],
]


def _resolve_mask_width(time_len, aug_params):
    if "mask_width" in aug_params:
        return max(1, min(int(aug_params["mask_width"]), time_len))
    mask_fraction = aug_params.get("mask_fraction", 0.1)
    return max(1, min(int(round(time_len * mask_fraction)), time_len))


def _space_dropout(x, drop_prob, rng):
    keep_mask = rng.random(x.shape[0]) > drop_prob
    if not np.any(keep_mask):
        keep_mask[rng.integers(0, x.shape[0])] = True
    x = x.clone()
    x[~torch.as_tensor(keep_mask, dtype=torch.bool), ...] = 0
    return x


def _space_shuffle(x, rng):
    permutation = torch.as_tensor(rng.permutation(x.shape[0]), dtype=torch.long)
    return x.index_select(0, permutation)


def _space_symmetry(x, aug_params):
    symmetry_pairs = aug_params.get("symmetry_pairs")
    x = x.clone()

    if symmetry_pairs:
        for left_idx, right_idx in symmetry_pairs:
            tmp = x[left_idx].clone()
            x[left_idx] = x[right_idx]
            x[right_idx] = tmp
        return x

    half = x.shape[0] // 2
    if half == 0:
        return x
    mirrored = x.clone()
    mirrored[:half] = x[-half:]
    mirrored[-half:] = x[:half]
    return mirrored


def _random_phase_rfft(x, phase_scale, rng):
    spectrum = torch.fft.rfft(x, dim=-1)
    if spectrum.shape[-1] <= 2:
        return x

    random_phase = torch.as_tensor(
        rng.uniform(
            -phase_scale,
            phase_scale,
            size=spectrum.shape[:-1] + (spectrum.shape[-1] - 2,),
        ),
        dtype=x.dtype,
        device=x.device,
    )
    phase_factor = torch.polar(torch.ones_like(random_phase), random_phase)
    spectrum[..., 1:-1] = spectrum[..., 1:-1] * phase_factor
    return torch.fft.irfft(spectrum, n=x.shape[-1], dim=-1)


def _frequency_shift(x, shift_bins):
    spectrum = torch.fft.fft(x, dim=-1)
    shifted = torch.roll(spectrum, shifts=int(shift_bins), dims=-1)
    return torch.fft.ifft(shifted, dim=-1).real


def _bandstop_filter(x, start_bin, stop_bin):
    spectrum = torch.fft.rfft(x, dim=-1)
    spectrum[..., start_bin:stop_bin] = 0
    return torch.fft.irfft(spectrum, n=x.shape[-1], dim=-1)


def apply_augmentation(x, aug_name, aug_params, rng):
    aug_params = aug_params or {}

    if aug_name not in ONLINE_AUGMENTATIONS:
        raise ValueError(f"Unknown augmentation {aug_name!r}")

    if aug_name == "none":
        return x

    aug_prob = aug_params.get("aug_prob", 0.5)
    if rng.random() >= aug_prob:
        return x

    if aug_name == "gaussian_noise":
        std = aug_params.get("std", 0.01)
        return x + torch.randn_like(x) * std

    if aug_name == "smooth_time_mask":
        time_len = x.shape[-1]
        mask_width = _resolve_mask_width(time_len, aug_params)
        start = int(rng.integers(0, time_len - mask_width + 1))
        x = x.clone()
        x[..., start:start + mask_width] = 0
        return x

    if aug_name == "time_reverse":
        return torch.flip(x, dims=(-1,))

    if aug_name == "sign_flip":
        return -x

    if aug_name == "ft_surrogate":
        phase_scale = aug_params.get("phase_scale", np.pi)
        return _random_phase_rfft(x, phase_scale, rng)

    if aug_name == "frequency_shift":
        shift_bins = aug_params.get("shift_bins")
        if shift_bins is None:
            shift_fraction = aug_params.get("shift_fraction", 0.02)
            shift_bins = max(1, int(round(x.shape[-1] * shift_fraction)))
        return _frequency_shift(x, shift_bins)

    if aug_name == "bandstop_filter":
        n_freq = x.shape[-1] // 2 + 1
        band_width = aug_params.get("band_width")
        if band_width is None:
            band_fraction = aug_params.get("band_fraction", 0.1)
            band_width = max(1, int(round(n_freq * band_fraction)))
        max_start = max(1, n_freq - band_width)
        start_bin = int(rng.integers(1, max_start + 1))
        stop_bin = min(start_bin + band_width, n_freq)
        return _bandstop_filter(x, start_bin, stop_bin)

    if aug_name == "space_symmetry":
        return _space_symmetry(x, aug_params)

    if aug_name == "space_dropout":
        drop_prob = aug_params.get("drop_prob", 0.1)
        return _space_dropout(x, drop_prob, rng)

    if aug_name == "space_shuffle":
        return _space_shuffle(x, rng)

    raise ValueError(f"Unknown augmentation {aug_name!r}")
