import numpy as np
import torch


ONLINE_AUGMENTATIONS = {
    "none": "no augmentation",
    "gaussian_noise": "add random Gaussian noise",
    "smooth_time_mask": "mask a contiguous time window",
    "time_reverse": "reverse the time axis",
    "sign_flip": "multiply the signal by -1",
    "ft_surrogate": "randomize Fourier phase",
    "frequency_shift": "shift frequency bins",
    "bandstop_filter": "remove a random frequency band",
    "space_symmetry": "swap spatial halves or configured pairs",
    "space_dropout": "zero random channels/parcels",
    "space_shuffle": "shuffle channels/parcels",
}

def _resolve_mask_width(time_len, aug_params):
    """Return the time-mask width from either an absolute width or fraction."""
    if "mask_width" in aug_params:
        return max(1, min(int(aug_params["mask_width"]), time_len))
    if "mask_fraction" not in aug_params:
        raise ValueError("smooth_time_mask requires 'mask_width' or 'mask_fraction'")
    mask_fraction = aug_params["mask_fraction"]
    return max(1, min(int(round(time_len * mask_fraction)), time_len))


def _space_dropout(x, drop_prob, rng):
    """Set a random subset of channels/parcels to zero while keeping at least one."""
    keep_mask = rng.random(x.shape[0]) > drop_prob
    if not np.any(keep_mask):
        keep_mask[rng.integers(0, x.shape[0])] = True
    x = x.clone()
    x[~torch.as_tensor(keep_mask, dtype=torch.bool), ...] = 0
    return x


def _space_shuffle(x, rng):
    """Randomly permute the channel/parcel axis."""
    permutation = torch.as_tensor(rng.permutation(x.shape[0]), dtype=torch.long)
    return x.index_select(0, permutation)


def _space_symmetry(x, aug_params):
    """Swap configured spatial pairs, or half-swap the channel/parcel axis."""
    if "symmetry_pairs" not in aug_params:
        raise ValueError("space_symmetry requires 'symmetry_pairs'; use None for half-swap")
    symmetry_pairs = aug_params["symmetry_pairs"]
    x = x.clone()

    if symmetry_pairs:
        for left_idx, right_idx in symmetry_pairs:
            tmp = x[left_idx].clone()
            x[left_idx] = x[right_idx]
            x[right_idx] = tmp
        return x

    half = x.shape[0] // 2
    if half == 0:
        raise ValueError("space_symmetry requires at least two channels/parcels")
    mirrored = x.clone()
    mirrored[:half] = x[-half:]
    mirrored[-half:] = x[:half]
    return mirrored


def _random_phase_rfft(x, phase_scale, rng):
    """Randomize Fourier phase while preserving the original spectral magnitude."""
    spectrum = torch.fft.rfft(x, dim=-1)
    if spectrum.shape[-1] <= 2:
        raise ValueError("ft_surrogate requires at least three frequency bins")

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
    """Shift the full complex spectrum by a fixed number of frequency bins."""
    spectrum = torch.fft.fft(x, dim=-1)
    shifted = torch.roll(spectrum, shifts=int(shift_bins), dims=-1)
    return torch.fft.ifft(shifted, dim=-1).real


def _bandstop_filter(x, start_bin, stop_bin):
    """Zero a contiguous frequency-bin interval in the real FFT spectrum."""
    spectrum = torch.fft.rfft(x, dim=-1)
    spectrum[..., start_bin:stop_bin] = 0
    return torch.fft.irfft(spectrum, n=x.shape[-1], dim=-1)


def apply_augmentation(x, aug_name, aug_params, rng):
    """Apply one selected online augmentation to a loaded fNIRS trial tensor."""
    if aug_name not in ONLINE_AUGMENTATIONS:
        raise ValueError(f"Unknown augmentation {aug_name!r}")

    if aug_name == "none":
        return x

    if aug_params is None:
        raise ValueError(f"aug_params must be provided for augmentation {aug_name!r}")

    if "aug_prob" not in aug_params:
        raise ValueError(f"aug_params for {aug_name!r} must include 'aug_prob'")
    aug_prob = aug_params["aug_prob"]
    if rng.random() >= aug_prob:
        return x

    if aug_name == "gaussian_noise":
        if "std" not in aug_params:
            raise ValueError("gaussian_noise requires 'std'")
        std = aug_params["std"]
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
        if "phase_scale" not in aug_params:
            raise ValueError("ft_surrogate requires 'phase_scale'")
        phase_scale = aug_params["phase_scale"]
        return _random_phase_rfft(x, phase_scale, rng)

    if aug_name == "frequency_shift":
        if "shift_bins" not in aug_params:
            raise ValueError("frequency_shift requires 'shift_bins'")
        shift_bins = aug_params["shift_bins"]
        return _frequency_shift(x, shift_bins)

    if aug_name == "bandstop_filter":
        n_freq = x.shape[-1] // 2 + 1
        if "band_width" not in aug_params:
            raise ValueError("bandstop_filter requires 'band_width'")
        band_width = aug_params["band_width"]
        max_start = max(1, n_freq - band_width)
        start_bin = int(rng.integers(1, max_start + 1))
        stop_bin = min(start_bin + band_width, n_freq)
        return _bandstop_filter(x, start_bin, stop_bin)

    if aug_name == "space_symmetry":
        return _space_symmetry(x, aug_params)

    if aug_name == "space_dropout":
        if "drop_prob" not in aug_params:
            raise ValueError("space_dropout requires 'drop_prob'")
        drop_prob = aug_params["drop_prob"]
        return _space_dropout(x, drop_prob, rng)

    if aug_name == "space_shuffle":
        return _space_shuffle(x, rng)

    raise ValueError(f"Unknown augmentation {aug_name!r}")
