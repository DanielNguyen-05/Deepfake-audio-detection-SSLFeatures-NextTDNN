from __future__ import annotations
import logging
import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ========================== CONFIG ==========================
PATH_TO_FEATURES = (
    "/Users/dangnguyen/Desktop/RADAR26/Deepfake-audio-detection-SSLFeatures-NextTDNN/RADAR2026-dev/features/HUBERT_LARGE_L8,"
    "/Users/dangnguyen/Desktop/RADAR26/Deepfake-audio-detection-SSLFeatures-NextTDNN/RADAR2026-dev/features/WAVLM_LARGE_L8"
)

EVAL_PROTOCOL_FILE = "/Users/dangnguyen/Desktop/RADAR26/Deepfake-audio-detection-SSLFeatures-NextTDNN/RADAR2026-dev/ASVspoof5.eval.track_1.tsv"
OUT_FOLD = "/Users/dangnguyen/Desktop/RADAR26/Deepfake-audio-detection-SSLFeatures-NextTDNN/models/amf_hubert_wavlm_nextdnn_eca_L8_Light_ASVSpoof5"

FEAT_LEN = 750
PADDING = "repeat"
BATCH_SIZE = 32
NUM_WORKERS = 0
DEVICE = "cpu"
ADD_LOSS = "ocsoftmax"

warnings.filterwarnings("ignore")
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

from train_asv5 import ASVspoof5Track1FeatureDataset, AMF_ECAPA_Model, forward_emb_logits


from loss import OCSoftmax, AMSoftmax 


# ----------------------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: torch.nn.Module, aux_loss_fn: Optional[torch.nn.Module] = None) -> Tuple[float | None, float | None]:
 

    feat_roots: List[Path] = [Path(p.strip()) for p in PATH_TO_FEATURES.split(",")]
    eval_ds = ASVspoof5Track1FeatureDataset(
        feat_roots=feat_roots,
        protocol_file=EVAL_PROTOCOL_FILE,
        split="eval",
        feat_len=FEAT_LEN,
        padding=PADDING,
    )
    loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=eval_ds.collate_fn,  # None batch'leri filtreler
        pin_memory=True,
    )

    model.eval()
    if aux_loss_fn is not None:
        aux_loss_fn.eval()

    out_dir = Path(OUT_FOLD)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_fp = out_dir / "eval_scores.txt"

   
    processed = set()
    if score_fp.is_file():
        with score_fp.open("r", encoding="utf-8") as fp:
            for line in fp:
                parts = line.strip().split()
                if parts:
                    processed.add(parts[0])

    # (b) Append modunda aç: her batch’te yaz
    fp_out = score_fp.open("a", encoding="utf-8")

    try:
        for batch in tqdm(loader, desc="Eval"):
            if batch is None:
                continue  

            f1, f2, uids, labs = batch

            keep_idx = [i for i, u in enumerate(uids) if u not in processed]
            if len(keep_idx) == 0:
                continue

            f1 = f1[keep_idx].to(DEVICE)
            f2 = f2[keep_idx].to(DEVICE)
            labs = labs[keep_idx].to(DEVICE)
            uids = [uids[i] for i in keep_idx]

            emb, logits = forward_emb_logits(model, f1, f2)

            if aux_loss_fn is not None:
                if ADD_LOSS == "amsoftmax":
                    logits, _ = aux_loss_fn(emb, labs)
                else:
                    _, logits = aux_loss_fn(emb, labs)

            prob_bona = logits if logits.dim() == 1 else F.softmax(logits, dim=1)[:, 0]
            probs = prob_bona.detach().cpu().numpy().tolist()
            labs_cpu = labs.detach().cpu().numpy().tolist()

            # (d) ANINDA dosyaya yaz (append)
            for uid, lab, s in zip(uids, labs_cpu, probs):
                lab_str = "bonafide" if int(lab) == 0 else "spoof"
                fp_out.write(f"{uid} {lab_str} {s:.6f}\n")
                processed.add(uid)
    finally:
        fp_out.close()

    eer = None
    try:
        import eval_metrics as em 
        all_scores, all_labels = [], []
        with score_fp.open("r", encoding="utf-8") as fp:
            for line in fp:
                _, lab_str, s = line.strip().split()
                all_scores.append(float(s))
                all_labels.append(0 if lab_str == "bonafide" else 1)
        scores_np = np.asarray(all_scores)
        labels_np = np.asarray(all_labels)
        eer = em.compute_eer(scores_np[labels_np == 0], scores_np[labels_np == 1])[0]
        logging.info(f"[Eval] EER = {eer:.4f}")
    except Exception as e:
        logging.warning(f"EER hesaplanamadı (etiket/protokol eksik olabilir): {e}")

    return eer, None  



def infer_emb_dim_from_state_dict(state: dict) -> int:
    """Checkpoint içindeki fc1.weight'tan emb_dim'i sez."""
    for k in ("fc1.weight", "module.fc1.weight"):
        if k in state:
            return int(state[k].shape[0])
    return 256  # emniyetli varsayılan

def main():
    global DEVICE
    DEVICE = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(OUT_FOLD) / "anti-spoofing_model.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint yok: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")  # {'model_state', opsiyonel 'aux_state'}

    # Kanal sayısını (C) dataset'ten öğren
    feat_roots = [Path(p.strip()) for p in PATH_TO_FEATURES.split(",")]
    tmp_ds = ASVspoof5Track1FeatureDataset(
        feat_roots=feat_roots,
        protocol_file=EVAL_PROTOCOL_FILE,
        split="eval",
        feat_len=FEAT_LEN,
        padding=PADDING,
    )
    channels = tmp_ds.C

    model_state = ckpt.get("model_state", ckpt)  # emniyet için
    emb_dim = infer_emb_dim_from_state_dict(model_state)

    model = AMF_ECAPA_Model(channels, emb_dim=emb_dim, num_classes=2, feat_len=FEAT_LEN)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        logging.warning(f"StateDict uyuşmazlıkları — missing: {missing}, unexpected: {unexpected}")
    model.to(DEVICE).eval()

    aux_model = None
    aux_state = ckpt.get("aux_state")
    if aux_state is not None and ADD_LOSS in {"amsoftmax", "ocsoftmax"}:
        if ADD_LOSS == "amsoftmax":
            aux_model = AMSoftmax(2, emb_dim, s=20.0, m=0.9)
        else:
            aux_model = OCSoftmax(emb_dim)
        aux_missing, aux_unexp = aux_model.load_state_dict(aux_state, strict=False)
        if aux_missing or aux_unexp:
            logging.warning(f"AUX StateDict uyuşmazlıkları — missing: {aux_missing}, unexpected: {aux_unexp}")
        aux_model.to(DEVICE).eval()

    evaluate(model, aux_model)


if __name__ == "__main__":
    main()