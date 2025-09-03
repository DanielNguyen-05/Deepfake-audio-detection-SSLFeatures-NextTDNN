
from __future__ import annotations
import argparse
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import eval_metrics as em
from loss import AMSoftmax, OCSoftmax, setup_seed
from NeXt_TDNN_ASV.models import NeXt_TDNN_ECA_ilk_ilk_Normal # backbone

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# ATTENTIONAL MULTI‑FEATURE FUSION (1‑D) --------------------------------------
# -----------------------------------------------------------------------------
class CoordAtt1D(nn.Module):
    """Coordinate Attention adapted to 1‑D (C × T) feature maps."""
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mid = max(8, channels // reduction)
        self.pool_t = nn.AdaptiveAvgPool1d(1)                 # GAPₜ
        self.conv1  = nn.Conv1d(channels, mid, 1, bias=False)
        self.bn1    = nn.BatchNorm1d(mid)
        self.act    = nn.SiLU()
        self.conv2  = nn.Conv1d(mid, channels, 1, bias=False)
        self.sig    = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:       # x: (B, C, T)
        y = self.pool_t(x)                                    # (B, C, 1)
        y = self.conv2(self.act(self.bn1(self.conv1(y))))     # (B, C, 1)
        return x * self.sig(y)                                # broadcast


class AMF_PSWF(nn.Module):
    """Parallel–Supplementary Weighted Fusion for two (B,C,T) tensors."""
    def __init__(self, channels: int):
        super().__init__()
        self.ca1 = CoordAtt1D(channels)
        self.ca2 = CoordAtt1D(channels)
        self.ca3 = CoordAtt1D(channels)  # supplementary

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        # Apply coordinate attention separately
        s1 = self.ca1(h1)
        s2 = self.ca2(h2)
        h3 = self.ca3(h1 + h2)
        return s1 * h3 + s2 * (1.0 - h3)  # (B, C, T)


# -----------------------------------------------------------------------------
# DATASET ---------------------------------------------------------------------
# -----------------------------------------------------------------------------
class MultiSSLFeatureDataset(Dataset):
    """Loads *paired* WavLM & HuBERT features (.pt) for ASVspoof‑2019."""

    def __init__(
        self,
        access_type: str,
        feat_roots: List[str | Path],  # [root_hubert, root_wavlm] (any order)
        protocol_dir: str | Path,
        part: str,
        feat_len: int = 750,
        padding: str = "repeat",
    ) -> None:
        super().__init__()
        assert len(feat_roots) == 2, "Expect two feature roots (HuBERT & WavLM)."
        self.feat_roots = [Path(r) for r in feat_roots]
        self.access_type = access_type
        self.part = part
        self.feat_len = feat_len
        self.padding = padding

        proto_fp = Path(protocol_dir) / f"ASVspoof2019.{access_type}.cm.{part}.trl.txt"
        with proto_fp.open("r", encoding="utf-8") as fh:
            rows = [ln.strip().split() for ln in fh]
        # items: (utt_id, label)
        self.items = [(r[1], 0 if r[4] == "bonafide" else 1) for r in rows]

        # infer channel dim (C) from first feature file
        sample = torch.load(self._feat_path(self.items[0][0], 0), map_location="cpu")
        self.C = sample.shape[0]

    # ---------- helpers ----------
    def _feat_path(self, utt_id: str, branch: int) -> Path:
        """branch=0 for feat_root[0], branch=1 for feat_root[1]."""
        root = self.feat_roots[branch]
        return root / self.access_type / self.part / f"{utt_id}.pt"

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        if T == self.feat_len:
            return x
        if T > self.feat_len:
            return x[:, : self.feat_len]
        # padding
        if self.padding == "zero":
            pad = torch.zeros(x.shape[0], self.feat_len - T)
        else:  # repeat
            pad = x.repeat(1, (self.feat_len + T - 1) // T)[:, : self.feat_len - T]
        return torch.cat([x, pad], dim=1)

    # ---------- mandatory ----------
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        utt_id, label = self.items[idx]
        feat1 = self._pad(torch.load(self._feat_path(utt_id, 0), map_location="cpu"))  # (C,T)
        feat2 = self._pad(torch.load(self._feat_path(utt_id, 1), map_location="cpu"))
        return feat1, feat2, utt_id, label

    def collate_fn(self, batch):
        f1, f2, ids, labs = zip(*batch)
        f1 = torch.stack(f1)
        f2 = torch.stack(f2)
        labs = torch.tensor(labs)
        return f1, f2, ids, labs


# -----------------------------------------------------------------------------
# MODEL -----------------------------------------------------------------------
# -----------------------------------------------------------------------------
class AMF_ECAPA_Model(nn.Module):
    """AMF fusion + NeXtTDNN‑ECA‑SE backbone + linear classifier."""

    def __init__(self, channels: int, emb_dim: int = 192, num_classes: int = 2, feat_len: int = 750):
        super().__init__()
        self.fusion = AMF_PSWF(channels)
        self.backbone = NeXt_TDNN_ECA_ilk_ilk_Normal.NeXtTDNN(in_chans=channels)
        # Infer backbone output dim once for pooling / fc
        with torch.no_grad():
            dummy = torch.randn(1, channels, feat_len)
            out = self.backbone(dummy)
            c_out = out.shape[1] if out.ndim == 3 else out.shape[-1]
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(c_out, emb_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(emb_dim, num_classes)

    # ----- API: embedding & logits -----
    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h*: (B,C,T)
        fused = self.fusion(h1, h2)
        x = self.backbone(fused)          # (B,C',T') or (B,C')
        if x.ndim == 3:
            x = self.pool(x).squeeze(-1)  # (B,C')
        emb = F.normalize(self.relu(self.fc1(x)), dim=1)
        logits = self.fc2(emb)
        return emb, logits


# -----------------------------------------------------------------------------
# TRAINING UTILITIES -----------------------------------------------------------
# -----------------------------------------------------------------------------
@torch.no_grad()
def forward_emb_logits(model: AMF_ECAPA_Model, f1: torch.Tensor, f2: torch.Tensor):
    return model(f1, f2)


def save_checkpoint(model: nn.Module, aux: Optional[nn.Module], path: Path) -> None:
    ckpt = {"model_state": model.state_dict()}
    if aux:
        ckpt["aux_state"] = aux.state_dict()
    torch.save(ckpt, path)


def adjust_lr(opt, base: float, decay: float, interval: int, epoch: int) -> None:
    lr = base * (decay ** (epoch // interval))
    for g in opt.param_groups:
        g["lr"] = lr


# -----------------------------------------------------------------------------
# TRAINING LOOP ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def train(args) -> None:
    # Resolve two roots (comma‑separated CLI arg → List[Path])
    feat_roots = [Path(p.strip()) for p in str(args.path_to_features).split(",")]

    train_ds = MultiSSLFeatureDataset(
        args.access_type, feat_roots, args.path_to_protocol, "train", args.feat_len, args.padding
    )
    dev_ds = MultiSSLFeatureDataset(
        args.access_type, feat_roots, args.path_to_protocol, "dev", args.feat_len, args.padding
    )

    train_loader = DataLoader(
        train_ds, args.batch_size, True,
        num_workers=args.num_workers, collate_fn=train_ds.collate_fn, pin_memory=True
    )
    dev_loader = DataLoader(
        dev_ds, args.batch_size, False,
        num_workers=args.num_workers, collate_fn=dev_ds.collate_fn, pin_memory=True
    )

    model = AMF_ECAPA_Model(train_ds.C, args.emb_dim, num_classes=2, feat_len=args.feat_len)
    model.to(args.device)

    opt_model = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    aux: Optional[nn.Module] = None
    opt_aux = None
    if args.add_loss == "amsoftmax":
        aux = AMSoftmax(2, args.emb_dim, s=args.alpha, m=args.r_real).to(args.device)
        opt_aux = torch.optim.SGD(aux.parameters(), lr=0.01)
    elif args.add_loss == "ocsoftmax":
        aux = OCSoftmax(
            args.emb_dim, r_real=args.r_real, r_fake=args.r_fake, alpha=args.alpha
        ).to(args.device)
        opt_aux = torch.optim.SGD(aux.parameters(), lr=args.lr)

    best_eer, early = float("inf"), 0
    ckpt_dir = args.out_fold / "checkpoint"

    for ep in range(args.num_epochs):
        # ─── TRAIN ────────────────────────────────────────────────────────────
        model.train()
        adjust_lr(opt_model, args.lr, args.lr_decay, args.interval, ep)
        if opt_aux:
            adjust_lr(opt_aux, args.lr, args.lr_decay, args.interval, ep)

        for f1, f2, _, y in tqdm(train_loader, desc=f"Train {ep+1}"):
            f1, f2, y = f1.to(args.device), f2.to(args.device), y.to(args.device)
            opt_model.zero_grad();  opt_aux and opt_aux.zero_grad()

            emb, logits = model(f1, f2)
            loss = ce(logits, y)
            if aux:
                if args.add_loss == "ocsoftmax":
                    l, logits = aux(emb, y)
                    loss = l * args.weight_loss
                else:  # amsoftmax
                    o, m = aux(emb, y)
                    loss = ce(m, y) * args.weight_loss
                    logits = o
            loss.backward()
            opt_model.step();  opt_aux and opt_aux.step()

        # ─── VALIDATE ─────────────────────────────────────────────────────────
        model.eval();  scores, labs = [], []
        with torch.no_grad():
            for f1, f2, _, y in tqdm(dev_loader, desc="Dev"):
                f1, f2, y = f1.to(args.device), f2.to(args.device), y.to(args.device)
                emb, logits = model(f1, f2)
                if aux:
                    if args.add_loss == "ocsoftmax":
                        _, logits = aux(emb, y)
                    else:
                        logits, _ = aux(emb, y)
                prob = (F.softmax(logits, dim=1)[:, 0] if logits.dim() > 1 else logits)
                scores.append(prob.cpu().numpy());  labs.append(y.cpu().numpy())
        scores = np.concatenate(scores);  labs = np.concatenate(labs)
        eer = em.compute_eer(scores[labs == 0], scores[labs == 1])[0]

        # log & checkpoints
        with (args.out_fold / "eer.log").open("a", encoding="utf-8") as fp:
            fp.write(f"{ep+1}\t{eer:.6f}\n")
        print(f"Epoch {ep+1}: EER = {eer:.4f}")

        save_checkpoint(model, aux, ckpt_dir / f"epoch_{ep+1}.pt")
        if eer < best_eer:
            best_eer, early = eer, 0
            save_checkpoint(model, aux, args.out_fold / "anti-spoofing_model.pt")
            if aux:
                torch.save(aux.state_dict(), args.out_fold / "anti-spoofing_loss_model.pt")
        else:
            early += 1
        if early >= 100:
            print("Early stop — no improvement for 100 epochs");  break

# -----------------------------------------------------------------------------
# ENTRY POINT -----------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("ECAPA‑NeXtTDNN + AMF (WavLM & HuBERT) trainer")
    parser.add_argument("--access_type", default="LA")
    parser.add_argument("--path_to_features", required=True,
                        help="Comma‑separated roots of HuBERT & WavLM feature folders")
    parser.add_argument("--path_to_protocol", required=True)
    parser.add_argument("--out_fold", required=True)
    parser.add_argument("--feat_len", type=int, default=750)
    parser.add_argument("--padding", choices=["zero", "repeat"], default="repeat")
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=598)
    parser.add_argument("--add_loss", choices=["softmax", "amsoftmax", "ocsoftmax"], default="ocsoftmax")
    parser.add_argument("--weight_loss", type=float, default=1.0)
    parser.add_argument("--r_real", type=float, default=0.9)
    parser.add_argument("--r_fake", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--continue_training", action="store_true")

    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        # Interactive defaults ------------------------------------------------
        args = argparse.Namespace(
            access_type="LA",
            path_to_features=("E:/akademikcalismalar/POST/DeepFakeAudio/DATASETLER/ASV2019Features/"
                               "HUBERT_LARGE_L8,"
                               "E:/akademikcalismalar/POST/DeepFakeAudio/DATASETLER/ASV2019Features/"
                               "WAVLM_LARGE_L8"),
            path_to_protocol=r"E:/akademikcalismalar/POST/DeepFakeAudio/DATASETLER/ASV2019/LA/ASVspoof2019_LA_cm_protocols",
            out_fold=Path("./models/amf_hubert_wavlm_nextdnn_eca_L8_Normal"),
            feat_len=750,
            padding="repeat",
            emb_dim=256,
            num_epochs=100,
            batch_size=32,
            lr=1e-3,
            lr_decay=0.5,
            interval=30,
            gpu="0",
            num_workers=0,
            seed=598,
            add_loss="ocsoftmax",
            weight_loss=1.0,
            r_real=0.9,
            r_fake=0.2,
            alpha=20.0,
            continue_training=False,
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)

    args.out_fold = Path(args.out_fold)
    args.path_to_protocol = Path(args.path_to_protocol)
    if not args.continue_training:
        if args.out_fold.exists():
            shutil.rmtree(args.out_fold)
        (args.out_fold / "checkpoint").mkdir(parents=True, exist_ok=True)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(args)
