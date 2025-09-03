# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from typing import List


from NeXt_TDNN_ASV.models.utils import LayerNorm, GRN

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from typing import Union, List



class TSConvNeXt(nn.Module):
    """TSConvNeXt Block.

    Args:
        dim (int): Kanal sayısı.
        drop_path (float): Stokastik derinlik oranı.
        kernel_size (int | list[int]): Tek bir kernel veya çok-ölçekli liste.
    """
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        kernel_size: Union[int, List[int]] = [7,65],
    ):
        super().__init__()

        # kernel_size'i daima liste hâline getir
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]

        # Her kernel tek sayı olmalı
        for k in kernel_size:
            assert (k - 1) % 2 == 0, "`kernel_size` tek sayı olmalıdır"

        self.num_scale = len(kernel_size)
        assert (
            dim % self.num_scale == 0
        ), f"`dim` ({dim}) ölçek sayısına ({self.num_scale}) tam bölünmelidir"

        # 1×1 projeksiyon
        self.projection_linear = nn.Conv1d(dim, dim, kernel_size=1)

        # Çok-ölçekli depth-wise konvolüsyonlar
        self.mscconv = nn.ModuleList(
            [
                nn.Conv1d(
                    dim // self.num_scale,
                    dim // self.num_scale,
                    kernel_size=k,
                    padding=(k - 1) // 2,
                    groups=dim // self.num_scale,
                )
                for k in kernel_size
            ]
        )

        # Ölçekleri birleştirmek için point-wise (1×1) lineer katman
        self.pwconv_1stage = nn.Linear(dim, dim)

        # FFN bölümü
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # Stokastik derinlik
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T)
        Returns:
            (N, C, T)
        """
        # ----- Çok-ölçekli konvolüsyon -----
        residual = x
        x = self.projection_linear(x)               # (N, C, T)
        chunks = x.chunk(self.num_scale, dim=1)     # Ölçeklere ayır

        x = torch.cat(
            [conv(chunk) for conv, chunk in zip(self.mscconv, chunks)], dim=1
        )
        x = self.act(x)

        # Ölçekleri 1×1 ile kaynaştır
        x = x.permute(0, 2, 1)                      # (N, T, C)
        x = self.pwconv_1stage(x)
        x = x.permute(0, 2, 1)                      # (N, C, T)
        x = x + residual                            # İlk artık bağlantı

        # ----- FFN -----
        residual = x
        x = x.permute(0, 2, 1)                      # (N, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)                      # (N, C, T)

        x = residual + self.drop_path(x)            # İkinci artık bağlantı
        return x