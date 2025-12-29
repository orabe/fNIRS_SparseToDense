import torch
import glob
import pickle
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from pathlib import PureWindowsPath
from torch import nn
import torch.nn as nn
import torch
import os
from torch import optim
import os    

class Dataset(torch.utils.data.Dataset):
    def __init__(self, files):
            self.files = files
            self.labels = {'Right':1, 'Left':2}

    def __len__(self):
            return len(self.files)

    def normalize(self, x):
        x = x.to_numpy()
        return (x - np.min(x))/ np.ptp(x)
    
    def normalize_chromo(self, x):
        data = []
        for chan in range(x.shape[1]):
            chromo = []
            for chr in range(x.shape[0]):
                chromo.append(self.normalize(x[chr, chan, :]))
            data.append(chromo)
        return np.array(data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        segmentfile = self.files[index]

        # read
        segment = None
        with open(segmentfile, 'rb') as handle:
            segment = pickle.load(handle)

        #normalize
        xt = torch.from_numpy(segment['xt'].to_numpy()).float() #[chromo, chan, time])
        # xt = torch.from_numpy(segment['xt_conc_pcr'].to_numpy()).float() #[chromo, chan, time])
        y = segment['class']-1

        return xt, y

class PreloadedDataset(torch.utils.data.Dataset):
    """Eagerly load all segments into RAM to minimize per-batch I/O overhead."""

    def __init__(self, files, dtype=torch.float32, pin_memory=False):
        self.samples = []
        self.labels = []
        self.dtype = dtype
        self.pin_memory = pin_memory
        files = list(files)
        if len(files) == 0:
            raise ValueError("PreloadedDataset received an empty file list.")

        print(f"Preloading {len(files)} samples into memory...")
        for idx, segmentfile in enumerate(files, start=1):
            with open(segmentfile, 'rb') as handle:
                segment = pickle.load(handle)

            xt = torch.from_numpy(segment['xt'].to_numpy()).to(dtype).contiguous()
            y = torch.tensor(segment['class'] - 1, dtype=torch.long)

            if pin_memory:
                xt = xt.pin_memory()
                y = y.pin_memory()

            self.samples.append(xt)
            self.labels.append(y)

            if idx % 100 == 0 or idx == len(files):
                print(f"  Loaded {idx}/{len(files)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

class CNN2D_BaselineV2(nn.Module):
    def __init__(self) -> None:
            super().__init__()
            self.model = nn.Sequential(
            
            # nn.Conv2d(109, 64, kernel_size=(1, 3)), # parcel space
            # nn.Conv2d(100, 64, kernel_size=(1, 3)), # channel space - BSQ
            # nn.Conv2d(68, 64, kernel_size=(1, 3)), # channel space - freshmotor
            
            nn.Conv2d(110, 64, kernel_size=(1, 3)), # parcel space
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(64),
            nn.MaxPool2d(kernel_size=(1, 3)),
            
            nn.Conv2d(64, 32, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 3)),
            
            nn.Conv2d(32, 16, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(16),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.5),
            
            nn.Linear(16*4, 2),
            # nn.Softmax(dim=-1)
        )
    def forward(self, x):
        # print(x.shape) # (Batch, Chromo, Channels, Time) -> [32, 2, 109, 90]
        x = x.permute(0, 2, 1, 3)
        # print(x.shape) # (Batch, Channels, Chromo, Time) -> [32, 109, 2, 90]
        return self.model(x)

class CNN2DModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN2DModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=(3, 1), padding=1),
            nn.LeakyReLU(0.2),  
            nn.BatchNorm2d(8),
            nn.Dropout(0.3),
            nn.Conv2d(8, 12, kernel_size=(3, 1), padding=1),
            nn.LeakyReLU(0.2),  
            nn.BatchNorm2d(12),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=(3, 1)),
            nn.Conv2d(12, 24, kernel_size=(3, 1), padding=1),
            nn.LeakyReLU(0.2),  
            nn.BatchNorm2d(24),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(0.3),
            nn.Conv2d(24, 24, kernel_size=(3, 1), padding=1),
            nn.LeakyReLU(0.2),  
            nn.BatchNorm2d(24),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.Dropout(0.3),
            nn.Conv2d(24, 32, kernel_size=(3, 1), padding=1),
            nn.LeakyReLU(0.2),  
            nn.BatchNorm2d(32),
            nn.Dropout(0.3),
            nn.Conv2d(32, 32, kernel_size=(3, 1), padding=1),
            nn.LeakyReLU(0.3),  
            nn.BatchNorm2d(32),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.2),  
            nn.BatchNorm2d(32),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.Conv2d(32, 32, kernel_size=(1, 1), padding=1),
            nn.LeakyReLU(0.2),  
            nn.BatchNorm2d(32),
            nn.Dropout(0.3),
            nn.Conv2d(32, 48, kernel_size=(1, 1), padding=1),
            nn.LeakyReLU(0.2),  
            nn.BatchNorm2d(48),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.Flatten(start_dim=1),
            # nn.Linear(48*4, 32),
            nn.Linear(48, 32),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.conv(x.permute(0, 2, 1, 3))  # [B, C, N, T]
        return out

