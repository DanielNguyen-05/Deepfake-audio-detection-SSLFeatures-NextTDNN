
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import logging
import warnings
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf

# ───────────────────────────────── CONFIG ────────────────────────────────
ACCESS_TYPE = "LA"  # "LA" or "PA"
PATH_TO_FEATURES = (
    "E:/akademikcalismalar/POST/DeepFakeAudio/DATASETLER/ASV2019Features/"
    "HUBERT_LARGE_L8,"
    "E:/akademikcalismalar/POST/DeepFakeAudio/DATASETLER/ASV2019Features/"
    "WAVLM_LARGE_L8"
)
PATH_TO_PROTOCOL_DIR = (
    "E:/akademikcalismalar/POST/DeepFakeAudio/DATASETLER/ASV2019/LA/"
    "ASVspoof2019_LA_cm_protocols"
)
EVAL_PROTOCOL_FILE = (
    Path(PATH_TO_PROTOCOL_DIR) / "ASVspoof2019.LA.cm.eval.trl.txt"
).as_posix()
OUT_FOLD = "./models/amf_hubert_wavlm_nextdnn_eca_L8_Normal"  # training output folder
FEAT_LEN = 750
PADDING = "repeat"  # "zero" | "repeat"
BATCH_SIZE = 32
NUM_WORKERS = 0
ADD_LOSS = "ocsoftmax"  # "softmax" | "amsoftmax" | "ocsoftmax"
DEVICE = "cuda"  # fallback to CPU if GPU unavailable

# Protocol column indices (for LA eval)
UTT_COL_INDEX = 1
TAG_COL_INDEX = 3

warnings.filterwarnings("ignore")
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

# ────────────────────────── IMPORT TRAINING MODULE ───────────────────────
# `trainAMF.py` must be discoverable in PYTHONPATH or same directory.
from train import (
    MultiSSLFeatureDataset,
    forward_emb_logits,  # convenience wrapper → (emb, logits)
)

# If OC‑Softmax / AM‑Softmax was used during training, we need the loss layer
# class definitions – they reside in `loss.py`.
from loss import OCSoftmax, AMSoftmax  # noqa: F401  (for torch.load)

# ──────────────────────── UTILITY: TAG LOOKUP ────────────────────────────

def _load_tag_lookup(protocol_file: str | Path) -> dict[str, str]:
    """Return a lookup mapping utterance‑id → attack_tag (Axx or "-")."""
    file_path = Path(protocol_file)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    lookup: dict[str, str] = {}
    with file_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) <= max(UTT_COL_INDEX, TAG_COL_INDEX):
                continue
            utt_raw = parts[UTT_COL_INDEX]
            utt_stem = Path(utt_raw).stem
            tag_val = parts[TAG_COL_INDEX]
            for variant in {
                utt_raw,
                utt_stem,
                f"{utt_stem}.wav",
                f"{utt_stem}.flac",
                f"{utt_stem}.npy",
            }:
                lookup[variant] = tag_val
    return lookup

_TAG_LOOKUP = _load_tag_lookup(EVAL_PROTOCOL_FILE)

# ───────────────────────────── EVALUATION ────────────────────────────────
@torch.no_grad()
def evaluate(model: torch.nn.Module, aux_loss_fn: Optional[torch.nn.Module] = None):
    feat_roots: List[Path] = [Path(p.strip()) for p in PATH_TO_FEATURES.split(",")]
    eval_ds = MultiSSLFeatureDataset(
        ACCESS_TYPE,
        feat_roots,
        PATH_TO_PROTOCOL_DIR,
        part="eval",
        feat_len=FEAT_LEN,
        padding=PADDING,
    )
    loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=eval_ds.collate_fn,
        pin_memory=True,
    )

    model.eval();  aux_loss_fn and aux_loss_fn.eval()

    scores, labels, utt_ids, utt_tags = [], [], [], []
    warned_missing = False

    for f1, f2, uids, labs in tqdm(loader, desc="Eval"):
        f1 = f1.to(DEVICE);  f2 = f2.to(DEVICE);  labs = labs.to(DEVICE)

        emb, logits = forward_emb_logits(model, f1, f2)

        if aux_loss_fn is not None:
            if ADD_LOSS == "amsoftmax":
                logits, _ = aux_loss_fn(emb, labs)
            else:  # ocsoftmax
                _, logits = aux_loss_fn(emb, labs)

        prob_bona = logits if logits.dim() == 1 else F.softmax(logits, 1)[:, 0]
        prob_bona = prob_bona.cpu().numpy();  labs_np = labs.cpu().numpy()

        for p, l, uid in zip(prob_bona, labs_np, uids):
            key = Path(uid).stem
            tag = _TAG_LOOKUP.get(key)
            if tag is None:
                if not warned_missing:
                    logging.warning("%s not in protocol → tag=A00", key)
                    warned_missing = True
                tag = "A00"
            scores.append(float(p));  labels.append(int(l))
            utt_ids.append(uid);  utt_tags.append(tag)

    scores_arr = np.asarray(scores);  labels_arr = np.asarray(labels)

    out_dir = Path(OUT_FOLD);  out_dir.mkdir(parents=True, exist_ok=True)
    score_fp = out_dir / "eval_scores_with_tags_deneme.txt"
    with score_fp.open("w", encoding="utf-8") as fp:
        for uid, tag, lab, s in zip(utt_ids, utt_tags, labels_arr, scores_arr):
            lab_str = "spoof" if lab == 1 else "bonafide"
            fp.write(f"{uid} {tag} {lab_str} {s:.6f}\n")

    try:
        eer_cm, min_tDCF = compute_eer_and_tdcf(score_fp)
        print(f"[Eval] EER = {eer_cm:.4f}\tmin tDCF = {min_tDCF:.4f}")
    except Exception:
        print("[Eval] Labels not found — skipping EER/tDCF.")

# ─────────────────────────────── MAIN ────────────────────────────────────

def main():
    """Load checkpoint dict, rebuild model, then run evaluation."""
    global DEVICE
    DEVICE = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    model_ckpt = Path(OUT_FOLD) / "anti-spoofing_model.pt"
    if not model_ckpt.is_file():
        raise FileNotFoundError(model_ckpt)
    ckpt: dict = torch.load(model_ckpt, map_location="cpu")  # dict w/ 'model_state'

    # ── Rebuild AMF_ECAPA model -------------------------------------------------
    from trainAMF import AMF_ECAPA_Model  # class definition is in training module

    feat_roots = [Path(p.strip()) for p in PATH_TO_FEATURES.split(",")]
    # use a tiny dataset instance to infer channel dim (C)
    tmp_ds = MultiSSLFeatureDataset(
        ACCESS_TYPE,
        feat_roots,
        PATH_TO_PROTOCOL_DIR,
        part="eval",  # any split works just to read one sample
        feat_len=FEAT_LEN,
        padding=PADDING,
    )
    channels = tmp_ds.C

    # Preserve embedding dim if stored; else fallback to 256
    emb_dim = ckpt.get("emb_dim", 256)

    model = AMF_ECAPA_Model(channels, emb_dim=emb_dim, num_classes=2, feat_len=FEAT_LEN)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(DEVICE).eval()

    # ── Aux loss layer (optional) ----------------------------------------------
    loss_model = None
    if ADD_LOSS in {"amsoftmax", "ocsoftmax"} and "aux_state" in ckpt:
        if ADD_LOSS == "amsoftmax":
            loss_model = AMSoftmax(2, emb_dim, s=20.0, m=0.9)
        else:  # ocsoftmax
            loss_model = OCSoftmax(emb_dim)
        loss_model.load_state_dict(ckpt["aux_state"], strict=False)
        loss_model.to(DEVICE).eval()

    evaluate(model, loss_model)


if __name__ == "__main__":
    main()
