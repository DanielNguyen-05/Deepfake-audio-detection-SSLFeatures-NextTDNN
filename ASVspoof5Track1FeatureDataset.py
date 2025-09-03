# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 12:03:16 2025

@author: ADMIN
"""

# -----------------------------------------------------------------------------
# ASVSPOOF5 TRACK-1 DATASET (Çift Özellik .pt) --------------------------------
# -----------------------------------------------------------------------------
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

class ASVspoof5Track1FeatureDataset(Dataset):
    """
    ASVspoof5 Track-1 için *eşlenik* (HuBERT & WavLM) .pt özellik yükleyici.
    Protokol dosyası CSV/TSV/whitespace ayracını ve başlık var/yok durumlarını esnek okur.

    Beklenti:
      - feat_roots: [root_hubert, root_wavlm] (sırası fark etmez)
      - Her kökün altında split klasörü (train/dev/eval) ve içinde {utt_id}.pt dosyası.
        Örn: <root>/train/<utt_id>.pt
      - Protokol satırlarında *en azından* (utt_id, label) bilgisi bulunmalı.
        Label {bonafide, spoof} ya da {0,1} olabilir.
    """
    def __init__(
        self,
        feat_roots: List[str | Path],            # [root_hubert, root_wavlm]
        protocol_file: str | Path,               # tek bir dosya: train/dev/eval için ayrı ayrı çağır
        split: str,                               # "train" | "dev" | "eval"
        feat_len: int = 750,
        padding: str = "repeat",
    ) -> None:
        super().__init__()
        assert len(feat_roots) == 2, "HuBERT ve WavLM köklerini ver (iki adet)."
        self.feat_roots = [Path(r) for r in feat_roots]
        self.split = split
        self.feat_len = int(feat_len)
        self.padding = padding

        proto_fp = Path(protocol_file)
        if not proto_fp.exists():
            raise FileNotFoundError(f"Protokol bulunamadı: {proto_fp}")

        # Protokolden (utt_id, label_int) çıkar
        self.items = self._read_protocol(proto_fp)

        # Kanal sayısını ilk örnekten öğren
        sample = torch.load(self._feat_path(self.items[0][0], branch=0), map_location="cpu")
        if sample.ndim != 2:
            raise ValueError(f".pt tensörü (C,T) olmalı, gelen shape: {tuple(sample.shape)}")
        self.C = sample.shape[0]

    # ---------------------- protokol okuma ----------------------
    def _read_protocol(self, path: Path) -> List[Tuple[str, int]]:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        # Ayraç tespiti (csv/tsv/space)
        if "\t" in text and ("," not in text.splitlines()[0] or text.count("\t") >= text.count(",")):
            delim = "\t"
        elif "," in text.splitlines()[0]:
            delim = ","
        else:
            delim = None  # whitespace

        rows: List[List[str]] = []
        headers: Optional[List[str]] = None

        if delim in (",", "\t"):
            sniffer = csv.Sniffer()
            lines = text.splitlines()
            # Başlık var mı?
            has_header = False
            try:
                sample = "\n".join(lines[:10])
                has_header = sniffer.has_header(sample)
            except Exception:
                pass

            reader = csv.reader(lines, delimiter=delim)
            if has_header:
                headers = next(reader)
            for r in reader:
                if any(tok.strip() for tok in r):
                    rows.append([tok.strip() for tok in r])
        else:
            for ln in text.splitlines():
                if ln.strip():
                    parts = re.split(r"\s+", ln.strip())
                    rows.append(parts)

        # Kolon indekslerini bul (esnek)
        uid_idx = self._guess_uid_index(rows, headers)
        lab_idx = self._guess_label_index(rows, headers)

        items: List[Tuple[str,int]] = []
        for r in rows:
            uid = r[uid_idx]
            lab_tok = r[lab_idx].lower()
            if lab_tok in ("bonafide", "bona-fide", "genuine", "real", "target"):
                lab = 0
            elif lab_tok in ("spoof", "attack", "non-target", "fake"):
                lab = 1
            else:
                # sayısal mı?
                try:
                    v = int(float(lab_tok))
                    # 0 -> bona, 1 -> spoof varsayımı
                    lab = 0 if v == 0 else 1
                except Exception:
                    # fallback: satırda 'bonafide' / 'spoof' arama
                    joined = " ".join(r).lower()
                    if "bonafide" in joined or "genuine" in joined or "target" in joined:
                        lab = 0
                    elif "spoof" in joined or "attack" in joined or "non-target" in joined:
                        lab = 1
                    else:
                        raise ValueError(f"Etiket çözümlenemedi: {r}")
            items.append((uid, lab))
        if not items:
            raise ValueError(f"Protokolden öğe çıkmadı: {path}")
        return items

        # Not: İsterseniz burada protokolde wav yolu varsa, Path stem'ini utt_id olarak da çekebilirsiniz.

    def _guess_uid_index(self, rows: List[List[str]], headers: Optional[List[str]]) -> int:
        # Başlık varsa, ortak alan adlarını ara
        if headers:
            hmap = {h.lower(): i for i, h in enumerate(headers)}
            for key in ("utt", "utt_id", "id", "utterance", "file", "filename", "key"):
                if key in hmap:
                    return hmap[key]
        # Başlık yoksa: çoğu protokolde ilk kolon utt_id olur
        return 0

    def _guess_label_index(self, rows: List[List[str]], headers: Optional[List[str]]) -> int:
        if headers:
            hmap = {h.lower(): i for i, h in enumerate(headers)}
            for key in ("label", "bonafide_or_spoof", "class", "target", "spoof", "is_spoof"):
                if key in hmap:
                    return hmap[key]
        # Heuristik: satırda 'bonafide/spoof' geçen kolon
        for j in range(min(6, max(len(r) for r in rows))):
            col = " ".join(r[j].lower() for r in rows[:100] if len(r) > j)
            if ("bonafide" in col or "spoof" in col or "target" in col or "non-target" in col):
                return j
        # Son çare: son kolon
        return -1

    # ---------------------- yardımcılar ----------------------
    def _feat_path(self, utt_id: str, branch: int) -> Path:
        root = self.feat_roots[branch]
        # Klasör yapısı: <root>/<split>/<utt_id>.pt
        p = root / self.split / f"{utt_id}.pt"
        if not p.exists():
            # Alternatif: kök/<track1>/<split>/<utt_id>.pt vb. için ufak arama
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
            rep = x.repeat(1, (self.feat_len + T - 1) // T)[:, : self.feat_len - T]
            pad = rep
        return torch.cat([x, pad], dim=1)

    # ---------------------- Dataset API ----------------------
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        utt_id, label = self.items[idx]
        f1 = torch.load(self._feat_path(utt_id, 0), map_location="cpu")  # (C,T)
        f2 = torch.load(self._feat_path(utt_id, 1), map_location="cpu")  # (C,T)
        f1 = self._pad(f1);  f2 = self._pad(f2)
        return f1, f2, utt_id, label

    def collate_fn(self, batch):
        f1, f2, ids, labs = zip(*batch)
        return torch.stack(f1), torch.stack(f2), list(ids), torch.tensor(labs, dtype=torch.long)
