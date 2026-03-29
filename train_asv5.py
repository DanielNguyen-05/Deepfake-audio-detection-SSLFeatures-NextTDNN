from __future__ import annotations
import argparse
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import csv
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import eval_metrics as em
from loss import AMSoftmax, OCSoftmax, setup_seed
from NeXt_TDNN_ASV.models import NeXt_TDNN_ECA_ilk_ilk_Light  # backbone

warnings.filterwarnings("ignore")

# =============================================================================
# ATTENTIONAL MULTI-FEATURE FUSION (1-D)
# =============================================================================
class CoordGate1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mid = max(8, channels // reduction)
        self.pool_t = nn.AdaptiveAvgPool1d(1)
        self.conv1  = nn.Conv1d(channels, mid, 1, bias=False)
        self.bn1    = nn.BatchNorm1d(mid)
        self.act    = nn.SiLU()
        self.conv2  = nn.Conv1d(mid, channels, 1, bias=False)
        self.sig    = nn.Sigmoid()
    def forward(self, x):
        y = self.pool_t(x)
        y = self.conv2(self.act(self.bn1(self.conv1(y))))
        return self.sig(y)  # sadece [0,1] kapı
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
    def __init__(self, channels: int):
        super().__init__()
        self.ca1  = CoordAtt1D(channels)   # h1’i iyileştir
        self.ca2  = CoordAtt1D(channels)   # h2’yi iyileştir
        self.gate = CoordGate1D(channels)  # kapı üret
    def forward(self, h1, h2):
        s1 = self.ca1(h1)
        s2 = self.ca2(h2)
        g  = self.gate(h1 + h2)            # (B,C,1) veya (B,C,1) broadcast
        return s1 * g + s2 * (1.0 - g)


# =============================================================================
# ASVspoof5 Track-1 DATASET (HuBERT & WavLM .pt eşlemeli)
# =============================================================================
class ASVspoof5Track1FeatureDataset(Dataset):
    """
    - feat_roots: [root_hubert, root_wavlm] (sırası fark etmez)
    - klasör yapısı: <root>/<split>/<utt_id>.pt  (split: train/dev/eval)
    - protokol dosyası: csv/tsv/whitespace ve başlık var/yok esnek okuma
      en az (utt_id, label) içermeli; label {bonafide, spoof} veya {0,1}
    """
    def __init__(
        self,
        feat_roots: List[str | Path],
        protocol_file: str | Path,
        split: str,                      # "train" | "dev" | "eval"
        feat_len: int = 750,
        padding: str = "repeat",
    ) -> None:
        super().__init__()
        assert len(feat_roots) == 2, "HuBERT ve WavLM köklerini (iki adet) ver."
        self.feat_roots = [Path(r) for r in feat_roots]
        self.split = split
        self.feat_len = int(feat_len)
        self.padding = padding

        proto_fp = Path(protocol_file)
        if not proto_fp.exists():
            raise FileNotFoundError(f"Protokol bulunamadı: {proto_fp}")

        self.items = self._read_protocol(proto_fp)

        sample = torch.load(self._feat_path(self.items[0][0], branch=0), map_location="cpu")
        if sample.ndim != 2:
            raise ValueError(f".pt tensörü (C,T) olmalı, gelen shape: {tuple(sample.shape)}")
        self.C = sample.shape[0]

    # ---------------------- protokol okuma ----------------------
    def _read_protocol(self, path: Path):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if "\t" in text and ("," not in text.splitlines()[0] or text.count("\t") >= text.count(",")):
            delim = "\t"
        elif "," in text.splitlines()[0]:
            delim = ","
        else:
            delim = None

        rows = []
        headers = None

        if delim in (",", "\t"):
            sniffer = csv.Sniffer()
            lines = text.splitlines()
            has_header = False
            try:
                sample = "\n".join(lines[:10])
                has_header = sniffer.has_header(sample)
            except Exception:
                pass
            reader = csv.reader(lines, delimiter=delim)
            if has_header:
                headers = next(reader, None)
            for r in reader:
                if any(tok.strip() for tok in r):
                    rows.append([tok.strip() for tok in r])
        else:
            for ln in text.splitlines():
                if ln.strip():
                    rows.append(re.split(r"\s+", ln.strip()))

        uid_idx = self._guess_uid_index(rows, headers)
        lab_idx = self._guess_label_index(rows, headers)

        items = []
        for r in rows:
            uid = r[uid_idx]
            lab_tok = r[lab_idx].lower()
            if lab_tok in ("bonafide", "bona-fide", "genuine", "real", "target"):
                lab = 0
            elif lab_tok in ("spoof", "attack", "non-target", "fake"):
                lab = 1
            else:
                try:
                    v = int(float(lab_tok))
                    lab = 0 if v == 0 else 1
                except Exception:
                    joined = " ".join(r).lower()
                    if any(k in joined for k in ("bonafide", "genuine", "target")):
                        lab = 0
                    elif any(k in joined for k in ("spoof", "attack", "non-target", "fake")):
                        lab = 1
                    else:
                        raise ValueError(f"Etiket çözümlenemedi: {r}")
            items.append((uid, lab))
        if not items:
            raise ValueError(f"Protokolden öğe çıkmadı: {path}")
        return items

    # def _guess_uid_index(self, rows, headers):
    #     if headers:
    #         hmap = {h.lower(): i for i, h in enumerate(headers)}
    #         for key in ("utt", "utt_id", "id", "utterance", "file", "filename", "key"):
    #             if key in hmap:
    #                 return hmap[key]
    #     return 0
    

    # def _guess_label_index(self, rows, headers):
    #     if headers:
    #         hmap = {h.lower(): i for i, h in enumerate(headers)}
    #         for key in ("label", "bonafide_or_spoof", "class", "target", "spoof", "is_spoof"):
    #             if key in hmap:
    #                 return hmap[key]
    #     max_cols = max(len(r) for r in rows)
    #     for j in range(min(6, max_cols)):
    #         col = " ".join(r[j].lower() for r in rows[:100] if len(r) > j)
    #         if any(k in col for k in ("bonafide", "spoof", "target", "non-target", "fake", "genuine")):
    #             return j
    #     return -1
    # ---------------------- protokol sütun sezgileri ----------------------
    def _guess_uid_index(self, rows, headers):
        """
        Utterance ID sütununu bulur.
        Başlık varsa 'utt', 'utt_id' veya 'flac_file_name' benzeri adları seçer.
        Yoksa T_/D_/E_ + 10 rakam kalıbını en çok sağlayan kolonu alır.
        """
        if headers:
            hmap = {h.lower(): i for i, h in enumerate(headers)}
            for key in ("utt", "utt_id", "flac_file_name", "flac", "file_name", "filename", "wav", "fname", "file"):
                if key in hmap:
                    return hmap[key]
    
        # Başlık yoksa: T_/D_/E_ + 10 rakam (örn. T_0000000000) kalıbını ara
        import re
        pat = re.compile(r"^[TDE]_\d{10}$")
        max_cols = max(len(r) for r in rows)
        best_j, best_score = 0, -1
        sample_rows = rows[:200]  # hız için
        for j in range(max_cols):
            score = 0
            for r in sample_rows:
                if len(r) > j and pat.match(r[j]):
                    score += 1
            if score > best_score:
                best_j, best_score = j, score
        return best_j


    def _guess_label_index(self, rows, headers):
        """
        Etiket (bonafide/spoof) sütununu bulur.
        'KEY' başlığını da tanır. Başlık yoksa tüm sütunları içerik sezgisiyle tarar.
        """
        if headers:
            hmap = {h.lower(): i for i, h in enumerate(headers)}
            for key in ("label", "bonafide_or_spoof", "class", "target", "spoof", "is_spoof", "key"):
                if key in hmap:
                    return hmap[key]
    
        max_cols = max(len(r) for r in rows)
        # İlk 100 satırın her sütununda anahtar kelime izleri var mı diye bak
        for j in range(max_cols):
            col_text = " ".join(r[j].lower() for r in rows[:100] if len(r) > j)
            if any(k in col_text for k in ("bonafide", "bona-fide", "genuine", "target", "non-target", "spoof", "fake")):
                return j
        # bulunamazsa son çare: son sütun
        return max_cols - 1 if max_cols > 0 else -1

    # ---------------------- yardımcılar ----------------------
    def _feat_path(self, utt_id: str, branch: int) -> Path:
        root = self.feat_roots[branch]
        p = root / self.split / f"{utt_id}.pt"
        if not p.exists():
            alt = list(root.glob(f"**/{self.split}/{utt_id}.pt"))
            if alt:
                return alt[0]
        return p

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        if T == self.feat_len:
            return x
        if T > self.feat_len:
            return x[:, : self.feat_len]
        if self.padding == "zero":
            pad = torch.zeros(x.shape[0], self.feat_len - T, dtype=x.dtype)
        else:
            pad = x.repeat(1, (self.feat_len + T - 1) // T)[:, : self.feat_len - T]
        return torch.cat([x, pad], dim=1)

    # ---------------------- Dataset API ----------------------
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        utt_id, label = self.items[idx]
        try:
            f1 = torch.load(self._feat_path(utt_id, 0), map_location="cpu")  # (C,T)
            f2 = torch.load(self._feat_path(utt_id, 1), map_location="cpu")  # (C,T)
        except Exception as e:
            return None  # collate_fn bunu ayıklayacak

        f1 = self._pad(f1);  f2 = self._pad(f2)
        return f1, f2, utt_id, int(label)

    def collate_fn(self, batch):
        # None dönen örnekleri at
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None

        f1s, f2s, uids, labs = zip(*batch)
        f1s = torch.stack(f1s, dim=0)  # (B,C,T)
        f2s = torch.stack(f2s, dim=0)  # (B,C,T)
        labs = torch.as_tensor(labs, dtype=torch.long)
        return f1s, f2s, list(uids), labs

# =============================================================================
# MODEL
# =============================================================================
class AMF_ECAPA_Model(nn.Module):
    """AMF fusion + NeXtTDNN-ECA backbone + linear classifier."""
    def __init__(self, channels: int, emb_dim: int = 192, num_classes: int = 2, feat_len: int = 750):
        super().__init__()
        self.fusion = AMF_PSWF(channels)
        self.backbone = NeXt_TDNN_ECA_ilk_ilk_Light.NeXtTDNN(in_chans=channels)
        with torch.no_grad():
            dummy = torch.randn(1, channels, feat_len)
            out = self.backbone(dummy)
            c_out = out.shape[1] if out.ndim == 3 else out.shape[-1]
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(c_out, emb_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(emb_dim, num_classes)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fused = self.fusion(h1, h2)            # (B,C,T)
        x = self.backbone(fused)               # (B,C',T') veya (B,C')
        if x.ndim == 3:
            x = self.pool(x).squeeze(-1)       # (B,C')
        emb = F.normalize(self.relu(self.fc1(x)), dim=1)
        logits = self.fc2(emb)     
        return emb, logits


# =============================================================================
# UTILITIES
# =============================================================================
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


# =============================================================================
# TRAIN LOOP
# =============================================================================
def train(args) -> None:
    feat_roots = [Path(p.strip()) for p in str(args.path_to_features).split(",")]
    for r in feat_roots:
        if not r.exists():
            raise FileNotFoundError(f"Feature root bulunamadı: {r}")

    # Datasets & loaders
    train_ds = ASVspoof5Track1FeatureDataset(
        feat_roots=feat_roots,
        protocol_file=args.proto_train,
        split="train",
        feat_len=args.feat_len,
        padding=args.padding,
    )
    dev_ds = ASVspoof5Track1FeatureDataset(
        feat_roots=feat_roots,
        protocol_file=args.proto_dev,
        split="dev",
        feat_len=args.feat_len,
        padding=args.padding,
    )

    train_loader = DataLoader(
        train_ds, args.batch_size, True,
        num_workers=args.num_workers, collate_fn=train_ds.collate_fn, pin_memory=True
    )
    dev_loader = DataLoader(
        dev_ds, args.batch_size, False,
        num_workers=args.num_workers, collate_fn=dev_ds.collate_fn, pin_memory=True
    )

    # Model + losses
    model = AMF_ECAPA_Model(train_ds.C, args.emb_dim, num_classes=2, feat_len=args.feat_len).to(args.device)
    opt_model = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best_eer, early = float("inf"), 0
    ckpt_dir = args.out_fold / "checkpoint"

    for ep in range(args.num_epochs):
        # -------------------- TRAIN --------------------
        model.train()
        adjust_lr(opt_model, args.lr, args.lr_decay, args.interval, ep)

        for f1, f2, _, y in tqdm(train_loader, desc=f"Train {ep+1}"):
            f1, f2, y = f1.to(args.device), f2.to(args.device), y.to(args.device)
            opt_model.zero_grad()

            emb, logits = model(f1, f2)
            loss = ce(logits, y)
            loss.backward()
            opt_model.step()

        # -------------------- VALID --------------------
        model.eval();  scores, labs = [], []
        with torch.no_grad():
            for f1, f2, _, y in tqdm(dev_loader, desc="Dev"):
                f1, f2, y = f1.to(args.device), f2.to(args.device), y.to(args.device)
                emb, logits = model(f1, f2)
                
                # prob = (F.softmax(logits, dim=1)[:, 0] if logits.dim() > 1 else logits)
                # scores.append(prob.detach().cpu().numpy());  labs.append(y.detach().cpu().numpy())

                # Dùng raw fake logit (cao = spoof) để tính EER
                fake_scores = logits[:, 1].cpu().numpy()
                scores.append(fake_scores)
                labs.append(y.cpu().numpy())

        scores = np.concatenate(scores)
        labs = np.concatenate(labs)
        
        # EER với fake logit (flip sign để bona có score cao hơn)
        bona  = -scores[labs == 0]
        spoof = -scores[labs == 1]
        eer = em.compute_eer(bona, spoof)[0]

        # log & checkpoints
        print(f"Epoch {ep+1}: EER = {eer:.4f}")

        with (args.out_fold / "eer.log").open("a", encoding="utf-8") as fp:
            fp.write(f"{ep+1}\t{eer:.6f}\n")

        save_checkpoint(model, ckpt_dir / f"epoch_{ep+1}.pt")

        if eer < best_eer:
            best_eer, early = eer, 0
            save_checkpoint(model, args.out_fold / "anti-spoofing_model.pt")
        else:
            early += 1
        if early >= args.patience:
            print(f"Early stop sau {args.patience} epoch")
            break


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("ECAPA-NeXtTDNN + AMF (HuBERT & WavLM) trainer for ASVspoof5 Track-1")
    parser.add_argument("--path_to_features", required=True)
    parser.add_argument("--proto_train", required=True)
    parser.add_argument("--proto_dev", required=True)
    parser.add_argument("--proto_eval")
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
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=598)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--continue_training", action="store_true")

    if len(sys.argv) > 1:
        args = parser.parse_args()
        args.out_fold = Path(args.out_fold)
    else:
        args = argparse.Namespace(
            path_to_features="/Users/dangnguyen/Desktop/Deepfake-audio-detection-SSLFeatures-NextTDNN/RADAR2026-dev/features/HUBERT_LARGE_L8,/Users/dangnguyen/Desktop/Deepfake-audio-detection-SSLFeatures-NextTDNN/RADAR2026-dev/features/WAVLM_LARGE_L8",
            proto_train="/Users/dangnguyen/Desktop/Deepfake-audio-detection-SSLFeatures-NextTDNN/RADAR2026-dev/ASVspoof5.train.tsv",
            proto_dev="/Users/dangnguyen/Desktop/Deepfake-audio-detection-SSLFeatures-NextTDNN/RADAR2026-dev/ASVspoof5.dev.track_1.tsv",
            out_fold=Path("./models/amf_hubert_wavlm_nextdnn_eca_L8_Light_ASVSpoof5_PLAIN"),
            feat_len=750,
            padding="repeat",
            emb_dim=256,
            num_epochs=100,
            batch_size=32,
            lr=1e-3,
            lr_decay=0.5,
            interval=30,
            gpu="0",
            num_workers=4,
            seed=598,
            patience=100,
            continue_training=False,
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)

    args.out_fold = Path(args.out_fold)
    if not args.continue_training and args.out_fold.exists():
        shutil.rmtree(args.out_fold)
    (args.out_fold / "checkpoint").mkdir(parents=True, exist_ok=True)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(args)