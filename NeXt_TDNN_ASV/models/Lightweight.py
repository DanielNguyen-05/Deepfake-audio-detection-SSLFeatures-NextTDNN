# lightweight_cnn_conformer.py
"""Lightweight CNN‑Conformer speaker‑embedding extractor

Implements the architecture introduced in
*H. Wang, X. Lin, J. Zhang, “A Lightweight CNN‑Conformer Model for Automatic
Speaker Verification,” IEEE SPL 31, 2024*.

The network processes frame‑level features shaped **(B, F, T)** such as
HuBERT/WavLM vectors. Internally, inputs are reshaped to **(B, 1, T, F)** so
that the lightweight CNN front‑end can perform 2‑D time/frequency operations.
The model outputs an L2‑normalised 192‑D speaker embedding suitable for metric
learning or downstream anti‑spoofing tasks.
"""
from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
#   Helper layers
# -----------------------------------------------------------------------------
class StdPooling(nn.Module):
    """Standard‑deviation pooling along a dimension (unbiased=False)."""
    def __init__(self, dim: int = -1, keepdim: bool = True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        var = torch.var(x, dim=self.dim, keepdim=self.keepdim, unbiased=False)
        return torch.sqrt(var + 1e-8)


class ChannelFrequencyAttention(nn.Module):
    """Channel‑Frequency Attention (CFA) — Eq.(1) in the paper."""
    def __init__(self, in_ch: int, k: int = 3, hidden_ch: int = 8):
        super().__init__()
        pad = k // 2
        self.std_pool = StdPooling(dim=2, keepdim=True)
        self.conv1 = nn.Conv2d(1, hidden_ch, k, padding=pad)
        self.bn = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, 1, k, padding=pad)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,T,F)
        s = self.std_pool(x).permute(0, 2, 3, 1)        # (B,1,F,C)
        a = self.sigmoid(self.conv2(F.relu(self.bn(self.conv1(s)))))
        a = a.permute(0, 3, 1, 2)                       # (B,C,1,F)
        return x * a                                    # broadcast over T


class MBConvBlock(nn.Module):
    """MobileNetV2 inverted residual block (expansion default ×2)."""
    def __init__(self, in_ch: int, out_ch: int, expansion: int = 2, stride: int = 1):
        super().__init__()
        hid = in_ch * expansion
        self.use_res = stride == 1 and in_ch == out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hid, 1, bias=False),
            nn.BatchNorm2d(hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, hid, 3, stride=stride, padding=1, groups=hid, bias=False),
            nn.BatchNorm2d(hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        y = self.block(x)
        return y + x if self.use_res else y


class LightFFN(nn.Module):
    """Lightweight FFN with depth‑wise convolution (no 4× expansion)."""
    def __init__(self, d_model: int, k: int = 3, p: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.dw = nn.Conv1d(d_model, d_model, k, padding=k // 2, groups=d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(p)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,C)
        y = self.drop(self.act(self.fc1(x)))
        y = self.dw(y.transpose(1, 2)).transpose(1, 2)
        y = self.drop(self.act(y))
        y = self.drop(self.fc2(y))
        return y


class ConformerConvModule(nn.Module):
    """Conformer depth‑wise separable conv module."""
    def __init__(self, d_model: int, k: int = 15, p: float = 0.1):
        super().__init__()
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.glu = nn.GLU(dim=1)
        self.dw = nn.Conv1d(d_model, d_model, k, padding=k // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv1d(d_model, d_model, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,C)
        y = x.transpose(1, 2)
        y = self.glu(self.pw1(y))
        y = self.act(self.bn(self.dw(y)))
        y = self.drop(self.pw2(y))
        return y.transpose(1, 2)


class LightConformerBlock(nn.Module):
    """Conformer block with Macaron‑style LightFFN and CFA front‑end."""
    def __init__(self, d_model: int = 256, n_heads: int = 4,
                 conv_k: int = 15, ffn_k: int = 3, p: float = 0.1):
        super().__init__()
        self.ffn1 = LightFFN(d_model, ffn_k, p)
        self.ffn2 = LightFFN(d_model, ffn_k, p)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=p, batch_first=True)
        self.conv = ConformerConvModule(d_model, conv_k, p)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + 0.5 * self.drop(self.ffn1(self.norm1(x)))
        attn, _ = self.mha(self.norm2(x), self.norm2(x), self.norm2(x), key_padding_mask=mask)
        x = x + self.drop(attn)
        x = x + self.drop(self.conv(self.norm3(x)))
        x = x + 0.5 * self.drop(self.ffn2(self.norm4(x)))
        return x


class AttentiveStatsPooling(nn.Module):
    """Attentive statistics pooling (mean + std)."""
    def __init__(self, d_model: int, bottleneck: int = 128):
        super().__init__()
        self.lin = nn.Linear(d_model, bottleneck)
        self.tanh = nn.Tanh()
        self.attn = nn.Linear(bottleneck, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # (B,T,C)
        h = self.tanh(self.lin(x))
        e = self.attn(h).squeeze(-1)
        if mask is not None:
            e = e.masked_fill(mask, -1e9)
        w = F.softmax(e, dim=1).unsqueeze(-1)
        mean = torch.sum(w * x, dim=1)
        std = torch.sqrt(torch.sum(w * (x - mean.unsqueeze(1)) ** 2, dim=1) + 1e-8)
        return torch.cat([mean, std], dim=1)

# -----------------------------------------------------------------------------
#   CNN front‑end
# -----------------------------------------------------------------------------
class CNNFrontEnd(nn.Module):
    """Two‑stage MBConv front‑end with subsampling and CFA."""
    def __init__(self, chans: List[int] = [64, 256], k: int = 3):
        super().__init__()
        pad = k // 2
        self.subsample = nn.Sequential(
            nn.Conv2d(1, chans[0], k, stride=2, padding=pad, bias=False),
            nn.BatchNorm2d(chans[0]),
            nn.ReLU(inplace=True),
        )
        self.mb1 = MBConvBlock(chans[0], chans[0])
        self.mb2 = MBConvBlock(chans[0], chans[1])
        self.cfa = ChannelFrequencyAttention(chans[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,1,T,F)
        x = self.subsample(x)      # (B,C1,T/2,F/2)
        x = self.mb1(x)
        x = self.mb2(x)
        x = self.cfa(x)            # (B,C2,T/2,F/2)
        x = x.mean(dim=3)          # avg over F  -> (B,C2,T')
        return x.transpose(1, 2)   # (B,T',C2)

# -----------------------------------------------------------------------------
#   Full model
# -----------------------------------------------------------------------------
class CNNConformerModel(nn.Module):
    """Lightweight CNN‑Conformer speaker embedder."""
    def __init__(self, *, d_model: int = 192, num_blocks: int = 3,
                 emb_dim: int = 192, dropout: float = 0.1):
        super().__init__()
        # front‑end outputs (B,T',d_model)
        self.front = CNNFrontEnd([64, d_model])
        # encoder
        self.encoder = nn.ModuleList(
            [LightConformerBlock(d_model, 4, 15, 3, dropout) for _ in range(num_blocks)]
        )
        # pooling + projection
        self.pool = AttentiveStatsPooling(d_model)
        self.proj = nn.Linear(d_model * 2, emb_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x should be shaped (B,1,T,F). Returns L2‑normalised embedding."""
        x = self.front(x)
        for blk in self.encoder:
            x = blk(x, mask)
        x = self.pool(x, mask)
        x = F.normalize(self.proj(x), dim=1)
        return x
