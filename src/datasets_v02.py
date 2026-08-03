import torch
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset
from online_augmentations import apply_augmentation


class PreprocessedNIRSDataset(Dataset):
    def __init__(self, data_csv_path, mode="train"):
        self.data_csv = pd.read_csv(data_csv_path)
        self.mode = mode

    def __len__(self):
        return len(self.data_csv)  # Total number of trials

    def __getitem__(self, idx):
        """
        Loads a single trial on demand (efficient memory usage).
        """
        data_row = self.data_csv.loc[idx]

        record = xr.open_dataarray(data_row["snirf_file"])
        record.time.attrs['units'] = 'seconds'
        duration = 7

        if self.mode == "train":
            random_delta = np.random.choice(np.linspace(-1, 1, 5))
            start_time = data_row["onset"] + random_delta  # in seconds
        else:
            start_time = data_row["onset"]
        end_time = data_row["onset"] + duration  + 5   # in seconds

        baseline = record.sel(time=slice(data_row["onset"] - 2.5 , data_row["onset"])).mean("time")

        # Then, trimming is easy with `.sel()`:
        x = x.sel(time=slice(start_time, end_time)) - baseline
        x = x.isel(time=slice(0, 61))


        x = x + np.abs(x.min()) + 10
        hbr = 1/x
        hbr_baseline = hbr.mean("time")
        x = ((hbr - hbr_baseline)/hbr_baseline)

        # Normalize
        x = x / np.abs(x).max()
        x = x.fillna(0)
        
        y = data_row["trial_type"]

        # Convert to tensor
        trial_tensor = torch.tensor(x.values, dtype=torch.float32).unsqueeze(1)
        label_tensor = torch.tensor(int(y), dtype=torch.long)

        return trial_tensor, label_tensor
    
class PreprocessedNIRSDatasetV2(Dataset):
    def __init__(self, data_csv_path, mode="train"):
        self.data_csv = pd.read_csv(data_csv_path)
        self.mode = mode

    def __len__(self):
        return len(self.data_csv)  # Total number of trials

    def __getitem__(self, idx):
        """
        Loads a single trial on demand (efficient memory usage).
        """
        data_row = self.data_csv.loc[idx]

        record = xr.open_dataarray(data_row["snirf_file"])
        record.time.attrs['units'] = 'seconds'

        record = record / np.abs(record).max()
        record = record.fillna(0)

        duration = 7

        if self.mode == "train":
            random_delta = np.random.choice(np.linspace(-1, 1, 5))
            start_time = data_row["onset"] + random_delta  # in seconds
        else:
            start_time = data_row["onset"]
        end_time = data_row["onset"] + duration + 5   # in seconds

        baseline = record.sel(time=slice(data_row["onset"] - 2.5 , data_row["onset"])).mean("time")

        # Then, trimming is easy with `.sel()`:
        x = record.sel(time=slice(start_time, end_time)) - baseline
        x = x.isel(time=slice(0, 61))

        # Normalize        
        y = data_row["trial_type"]

        # Convert to tensor
        trial_tensor = torch.tensor(x.values, dtype=torch.float32)
        label_tensor = torch.tensor(int(y), dtype=torch.long)

        return trial_tensor, label_tensor
    
class PreprocessedNIRSDatasetV3(Dataset):
    def __init__(self, data_csv_path, mode="train"):
        self.data_csv = pd.read_csv(data_csv_path)
        self.mode = mode

    def __len__(self):
        return len(self.data_csv)  # Total number of trials

    def __getitem__(self, idx):
        """
        Loads a single trial on demand (efficient memory usage).
        """
        data_row = self.data_csv.loc[idx]

        record = xr.open_dataarray(data_row["snirf_file"])
        record.time.attrs['units'] = 'seconds'
        duration = 7

        if self.mode == "train":
            random_delta = np.random.choice(np.linspace(-1, 1, 5))
            start_time = data_row["onset"] + random_delta  # in seconds
        else:
            start_time = data_row["onset"]
        end_time = data_row["onset"] + duration + 5   # in seconds

        baseline = record.sel(time=slice(data_row["onset"] - 2.5 , data_row["onset"])).mean("time")

        # Then, trimming is easy with `.sel()`:
        x = record.sel(time=slice(start_time, end_time)) - baseline
        x = x.isel(time=slice(0, 61))

        # Normalize
        x = x / np.abs(x).max()
        x = x.fillna(0)
        
        y = data_row["trial_type"]

        # Convert to tensor
        trial_tensor = torch.tensor(x.values, dtype=torch.float32).unsqueeze(1)
        label_tensor = torch.tensor(int(y), dtype=torch.long)

        return trial_tensor, label_tensor

class fNIRSChannelSpaceLoad(Dataset):
    def __init__(self, data_csv_path, mode="train", chromo="HbO"):
        self.data_csv = pd.read_csv(data_csv_path)
        self.mode = mode
        self.chromo = chromo

    def __len__(self):
        return len(self.data_csv)  # Total number of trials

    def __getitem__(self, idx):
        """
        Loads a single trial on demand (efficient memory usage).
        """
        data_row = self.data_csv.loc[idx]
        if self.chromo == "both":
            record = xr.open_dataarray(data_row["snirf_file"])
        else:
            record = xr.open_dataarray(data_row["snirf_file"]).sel(chromo=self.chromo)
        record.time.attrs['units'] = 'seconds'

        duration = 10

        if self.mode == "train":
            random_delta = np.random.choice(np.linspace(-2.5, 2.5, 9))
            start_time = data_row["onset"] + random_delta  # in seconds
        else:
            start_time = data_row["onset"]
        end_time = data_row["onset"] + duration + 5   # in seconds

        baseline = record.sel(time=slice(data_row["onset"] - 2.5 , data_row["onset"])).mean("time")

        # Then, trimming is easy with `.sel()`:
        x = record.sel(time=slice(start_time, end_time)) - baseline
        x = x.isel(time=slice(0, 87))

        # Normalize        
        y = data_row["trial_type"]

        # Convert to tensor
        if self.chromo == 'both':
            trial_tensor = torch.tensor(x.values, dtype=torch.float32)
        else:
            trial_tensor = torch.tensor(x.values, dtype=torch.float32).unsqueeze(1)
        label_tensor = torch.tensor(int(y), dtype=torch.long)

        return trial_tensor, label_tensor

class fNIRSChannelSpaceSegmentLoad(Dataset):
    def __init__(self, data_csv_path, mode="train", chromo="HbO"):
        self.data_csv = pd.read_csv(data_csv_path)
        self.mode = mode
        self.chromo = chromo

    def __len__(self):
        return len(self.data_csv)  # Total number of trials

    def __getitem__(self, idx):
        """
        Loads a single trial on demand (efficient memory usage).
        """
        data_row = self.data_csv.loc[idx]
        if self.chromo == "both":
            record = xr.open_dataarray(data_row["snirf_file"])
        else:
            record = xr.open_dataarray(data_row["snirf_file"]).sel(chromo=self.chromo)
        record.time.attrs['units'] = 'seconds'

        x = record
        # Normalize        
        y = data_row["trial_type"]

        # Convert to tensor
        if self.chromo == 'both':
            trial_tensor = torch.tensor(x.values, dtype=torch.float32)
        else:
            trial_tensor = torch.tensor(x.values, dtype=torch.float32).unsqueeze(1)
        label_tensor = torch.tensor(int(y), dtype=torch.long)

        return trial_tensor, label_tensor



class fNIRSPreloadDataset(Dataset):
    def __init__(
        self,
        data_csv_path,
        mode="train",
        chromo="HbO",
        aug_name="none",
        aug_params=None,
        input_normalization="none",
        seed=42,
    ):
        self.data_csv = (
            data_csv_path.reset_index(drop=True).copy()
            if isinstance(data_csv_path, pd.DataFrame)
            else pd.read_csv(data_csv_path)
        )
        self.mode = mode
        self.chromo = chromo
        self.aug_name = aug_name
        self.aug_params = aug_params
        self.input_normalization = input_normalization
        self.rng = np.random.default_rng(seed)

        if input_normalization not in {"none", "per_window"}:
            raise ValueError(
                f"Unknown input_normalization: {input_normalization}. "
                "Choose 'none' or 'per_window'."
            )

        # === Pre-load all trials into RAM ===
        self.all_trials = []
        self.all_labels = []

        print(f"Preloading {len(self.data_csv)} trials into memory...")

        expected_shape = None
        for i, row in self.data_csv.iterrows():
            if chromo == "both":
                record = xr.open_dataarray(row["snirf_file"])
                trial_tensor = torch.tensor(record.values, dtype=torch.float32)
            else:
                record = xr.open_dataarray(row["snirf_file"]).sel(chromo=chromo)
                trial_tensor = torch.tensor(record.values, dtype=torch.float32).unsqueeze(1)
            if input_normalization == "per_window":
                scale = trial_tensor.std(
                    dim=(0, 2),
                    keepdim=True,
                    unbiased=False,
                ).clamp_min(1e-6)
                trial_tensor = trial_tensor / scale
            if expected_shape is None:
                expected_shape = tuple(trial_tensor.shape)
            elif tuple(trial_tensor.shape) != expected_shape:
                raise ValueError(
                    f"Trial {row['snirf_file']} has shape {tuple(trial_tensor.shape)}; "
                    f"expected {expected_shape}"
                )
            label_tensor = torch.tensor(int(row["trial_type"]), dtype=torch.long)

            self.all_trials.append(trial_tensor)
            self.all_labels.append(label_tensor)

        print(f"Loaded {len(self.all_trials)} trials into memory.")

    def __len__(self):
        return len(self.all_trials)

    def __getitem__(self, idx):
        trial = self.all_trials[idx]
        if self.mode == "train":
            trial = apply_augmentation(
                trial,
                self.aug_name,
                self.aug_params,
                self.rng,
            )
        return trial, self.all_labels[idx]
