import torch
import glob
import pickle
import numpy as np
from torch import nn

import matplotlib.pyplot as plt
plt.style.use('ggplot')
from pathlib import PureWindowsPath
import torch
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np


class CNN2D_BaselineV2(nn.Module):
    def __init__(self) -> None:
            super().__init__()
            self.model = nn.Sequential(
            nn.Conv2d(100, 64, kernel_size=(1, 3)),
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
            nn.Linear(16*1*2, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        # x = x.permute(0, 2, 1, 3)
        return self.model(x)

class CNN2D_BaselineV21(nn.Module):
    def __init__(self) -> None:
            super().__init__()
            self.model = nn.Sequential(
            nn.Conv2d(104, 64, kernel_size=(1, 3)),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.InstanceNorm2d(64),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Conv2d(64, 32, kernel_size=(1, 3)),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.InstanceNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Conv2d(32, 16, kernel_size=(1, 3)),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.InstanceNorm2d(16),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.4),
            nn.Linear(16*1*2, 2)
        )
    def forward(self, x):
        return self.model(x)

class CNN2D_BaselineV3(nn.Module):
    def __init__(self) -> None:
            super().__init__()
            self.model = nn.Sequential(
            nn.Conv2d(100, 64, kernel_size=(1, 3)),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(64),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Conv2d(64, 32, kernel_size=(1, 3)),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Conv2d(32, 16, kernel_size=(1, 3)),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(16),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.5),
            nn.Linear(16*1*1, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.model(x)
    
class CNN2D_Baseline_Image(nn.Module):
    def __init__(self) -> None:
            super().__init__()
            self.model = nn.Sequential(
            nn.Conv2d(104, 64, kernel_size=(1, 3)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.MaxPool2d(kernel_size=(1, 3)),

            nn.Conv2d(64, 32, kernel_size=(1, 3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.MaxPool2d(kernel_size=(1, 3)),

            nn.Conv2d(32, 16, kernel_size=(1, 3)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Dropout(0.6),

            # nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Flatten(start_dim=1),
            # nn.Dropout(0.5),
            nn.Linear(16*1*3, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.model(x)

class CNN2D_Baseline_Image_both_chromos(nn.Module):
    def __init__(self) -> None:
            super().__init__()
            self.model = nn.Sequential(
            nn.Conv2d(104, 64, kernel_size=(1, 3)),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(64),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Conv2d(64, 32, kernel_size=(1, 3)),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Conv2d(32, 16, kernel_size=(1, 3)),
            nn.LeakyReLU(),
            # nn.Dropout(0.6),
            # nn.InstanceNorm2d(16),
            # nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.6),
            nn.Linear(16*2*3, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.model(x)
    
class CNN2DChannel(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(100, 64, kernel_size=(1, 3)),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(64),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Conv2d(64, 32, kernel_size=(1, 3)),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Conv2d(32, 16, kernel_size=(1, 3)),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(16),
            nn.MaxPool2d(kernel_size=(1, 3)),
        )

        # Don't define Linear layers yet
        self.classifier = None

    def forward(self, x):
        x = self.feature_extractor(x)

        if self.classifier is None:
            # First pass — define classifier based on actual input
            flattened_size = x.view(x.size(0), -1).size(1)
            self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Dropout(0.5),
                nn.Linear(flattened_size, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 2)
            )
            # Move to same device as input
            self.classifier.to(x.device)

        x = self.classifier(x)
        return x

class CNN2DChannelV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(100, 64, kernel_size=(1, 3)),
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
        )

        # Don't define Linear layers yet
        self.classifier = None

    def forward(self, x):
        x = self.feature_extractor(x)

        if self.classifier is None:
            # First pass — define classifier based on actual input
            flattened_size = x.view(x.size(0), -1).size(1)
            self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Dropout(0.5),
                nn.Linear(flattened_size, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 2)
            )
            # Move to same device as input
            self.classifier.to(x.device)

        x = self.classifier(x)
        return x


class CNN2DImage(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(104, 64, kernel_size=(1, 3)),
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
        )

        # Don't define Linear layers yet
        self.classifier = None

    def forward(self, x):
        x = self.feature_extractor(x)

        if self.classifier is None:
            # First pass — define classifier based on actual input
            flattened_size = x.view(x.size(0), -1).size(1)
            self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Dropout(0.5),
                nn.Linear(flattened_size, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 2)
            )
            # Move to same device as input
            self.classifier.to(x.device)

        x = self.classifier(x)
        return x

class CNN2DImageWUSTL(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(371, 64, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(64),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Conv2d(64, 32, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.InstanceNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 3)),
            # nn.Conv2d(32, 16, kernel_size=(1, 3)),
            # nn.ReLU(),
            # nn.Dropout(0.6),
            # nn.InstanceNorm2d(16),
            # nn.MaxPool2d(kernel_size=(1, 3)),
        )

        # Don't define Linear layers yet
        self.classifier = None

    def forward(self, x):
        x = self.feature_extractor(x)

        if self.classifier is None:
            # First pass — define classifier based on actual input
            flattened_size = x.view(x.size(0), -1).size(1)
            self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Dropout(0.5),
                nn.Linear(flattened_size, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 2)
            )
            # Move to same device as input
            self.classifier.to(x.device)

        x = self.classifier(x)
        return x

class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x):
        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs

class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)

class MSTCN_WRAP(nn.Module):
    def __init__(self, num_stages=4, num_layers=8, num_f_maps=16, dim=200, num_classes=2):
        super(MSTCN_WRAP, self).__init__()
        self.mstn_encode = MultiStageModel(num_stages, num_layers, num_f_maps, dim, num_classes)
        self.cnn = nn.Sequential(
            nn.Conv1d(num_stages*num_classes, 16, kernel_size=(3)),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.4),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(16, 32, kernel_size=(3)),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.4),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(32, 32, kernel_size=(3)),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.4),
            nn.MaxPool1d(kernel_size=3),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.4),
            nn.Linear(32*2, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        x = x.flatten(start_dim=1, end_dim=2)
        x = self.mstn_encode(x)
        x = x.permute(1, 0, 2, 3).flatten(start_dim=1, end_dim=2)
        return self.cnn(x)

class Transformer(nn.Module):
    def __init__(self, embedding_dim=128, num_layers=6) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.project = nn.Linear(200, embedding_dim)
        self.classification = nn.Sequential(
            nn.Linear(embedding_dim, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        x = x.flatten(start_dim=1, end_dim=2).permute(0, 2, 1)
        x = self.project(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.classification(x)


class ImprovedTransformer(nn.Module):
    def __init__(self, embedding_dim=64, num_layers=4, num_classes=21, time_steps=87, dropout=0.1):
        super().__init__()


        self.embedding_dim = embedding_dim
        self.time_steps = time_steps

        # Linear projection from parcels to embedding dim
        self.project = nn.Linear(371, embedding_dim)

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, time_steps, embedding_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=dropout, batch_first=True, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classification = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # Input shape: (B, 1, 371, 87)
        x = x.flatten(start_dim=1, end_dim=2).permute(0, 2, 1)  # → (B, 87, 371)
        x = self.project(x)  # → (B, 87, 128)
        x = x + self.positional_encoding  # add position
        x = self.transformer(x)  # → (B, 87, 128)
        x = x.mean(dim=1)  # average pooling over time
        x = self.dropout(x)
        return self.classification(x)


class BoldT(nn.Module):
    def __init__(self, embedding_dim=64, num_layers=4, num_classes=21,
                 time_steps=87, dropout=0.1, pooling="cls"):
        super().__init__()

        assert pooling in ["cls", "mean"], "pooling must be 'cls' or 'mean'"
        self.pooling = pooling
        self.embedding_dim = embedding_dim
        self.time_steps = time_steps

        # Linear projection from parcels to embedding dim
        self.project = nn.Linear(371, embedding_dim)

        # CLS token (if used)
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
            pe_len = time_steps + 1
        else:
            self.cls_token = None
            pe_len = time_steps

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, pe_len, embedding_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=256
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classification = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # Input shape: (B, 1, 371, 87)
        x = x.flatten(start_dim=1, end_dim=2).permute(0, 2, 1)  # → (B, 87, 371)
        x = self.project(x)                                    # → (B, 87, embedding_dim)

        B = x.size(0)

        if self.pooling == "cls":
            # Expand CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)      # (B, 1, embedding_dim)
            # Concatenate CLS at beginning
            x = torch.cat((cls_tokens, x), dim=1)              # (B, 88, embedding_dim)

        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Transformer
        x = self.transformer(x)

        # Pooling
        if self.pooling == "cls":
            x = x[:, 0]                                        # CLS token output
        else:
            x = x.mean(dim=1)                                  # mean pooling

        # Classification
        x = self.dropout(x)
        return self.classification(x)

