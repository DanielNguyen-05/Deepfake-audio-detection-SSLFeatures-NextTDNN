# -*- coding: utf-8 -*-
"""
Created on Fri May 23 15:06:22 2025

@author: ADMIN
"""

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from typing import Union, List

from NeXt_TDNN_ASV.models.utils import LayerNorm


class ECALayer(nn.Module):
    """
    Efficient Channel Attention (ECA) module.
    """
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T)
        y = self.avg_pool(x)             # (N, C, 1)
        y = y.transpose(-1, -2)          # (N, 1, C)
        y = self.conv(y)                 # (N, 1, C)
        y = self.sigmoid(y).transpose(-1, -2)  # (N, C, 1)
        return x * y                      # channel-wise scaling


class NeXtTDNN(nn.Module):
    """ NeXt-TDNN with integrated ECA attention. """
    def __init__(
        self,
        in_chans=80,
        depths=[1, 1, 1],
        dims=[256, 256, 256],
        drop_path_rate=0.,
        kernel_size: Union[int, List[int]] = 7,
        block: str = "TSConvNeXt",
        eca_k: int = 3,
    ):
        super().__init__()
        self.depths = depths
        # Stem layer
        self.stem = nn.ModuleList()
        stem_conv = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.stem.append(stem_conv)
        
        # Import block type
        block_mod = importlib.import_module(f"NeXt_TDNN_ASV.models.{block}")
        Block = getattr(block_mod, block)

        # Build stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i, num in enumerate(self.depths):
            stage = nn.Sequential(*[
                Block(dim=dims[i], drop_path=dp_rates[cur + j], kernel_size=kernel_size)
                for j in range(num)
            ])
            self.stages.append(stage)
            cur += num

        # MFA layer: merge features from all stages
        self.MFA = nn.Sequential(
            nn.Conv1d(3 * dims[-1], 3 * dims[-1], kernel_size=1),
            LayerNorm(3 * dims[-1], eps=1e-6, data_format="channels_first")
        )

        # ECA attention
        self.eca = ECALayer(channels=3 * dims[-1], k_size=eca_k)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem[0](x)
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        x = torch.cat(features, dim=1)
        x = self.MFA(x)
        x = self.eca(x)  # apply ECA after MFA
        return x


def MainModel(**kwargs):
    return NeXtTDNN(**kwargs)
