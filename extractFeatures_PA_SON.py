# -*- coding: utf-8 -*-
"""
ASVspoof2019 PA/LA toplu özellik (HuBERT / Wav2Vec2 / WavLM) çıkarıcı
Güncelleme: 12 Haziran 2025 – NumPy’sız, soundfile + haricî FFmpeg
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, List
import subprocess, io, torch, torchaudio, soundfile as sf
from tqdm import tqdm

# ------------------------------------------------------------------------- #
# 1) Harici FFmpeg yürütülebilirinin TAM yolunu yazın
# ------------------------------------------------------------------------- #
FFMPEG_EXE = r"C:\tools\ffmpeg\bin\ffmpeg.exe"   # ← kendi yolunuzu yazın

torchaudio.set_audio_backend("soundfile")  # yalnız soundfile kullanıyoruz

# ------------------------------------------------------------------------- #
# 2) Sağlam yükleyici: soundfile → haricî FFmpeg pipe → soundfile
# ------------------------------------------------------------------------- #
def robust_load(path: Path, target_sr: int) -> tuple[torch.Tensor, int]:
    """FLAC tensöre dönüştür (NumPy gerektirmez)."""
    try:
        data, sr = sf.read(str(path), always_2d=True)
    except Exception:
        cmd = [
            FFMPEG_EXE, "-v", "error", "-nostdin",
            "-i", str(path), "-err_detect", "ignore_err",
            "-ac", "1", "-ar", str(target_sr),
            "-f", "wav", "pipe:1",
        ]
        raw = subprocess.check_output(cmd)
        data, sr = sf.read(io.BytesIO(raw), always_2d=True)
    wav = torch.tensor(data.T.tolist(), dtype=torch.float32)  # (C, T)
    return wav, sr

# ------------------------------------------------------------------------- #
# 3) Ana işlev
# ------------------------------------------------------------------------- #
def extract_partition(
    *,
    access_type: str,
    part: str,
    protocol_dir: str | Path,
    audio_root: str | Path,
    output_dir: str | Path,
    bundle_name: str = "WAVLM_LARGE",
    layer: int = 8,
    downsample: Optional[int] = None,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    protocol_dir, audio_root, output_dir = map(Path, (protocol_dir, audio_root, output_dir))

    proto_fp = protocol_dir / f"ASVspoof2019.{access_type}.cm.{part}.trl.txt"
    audio_dir = audio_root / access_type / f"ASVspoof2019_{access_type}_{part}" / "flac"
    out_dir  = output_dir / access_type / part
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = getattr(torchaudio.pipelines, bundle_name)
    model  = bundle.get_model().to(device).eval()
    SR     = bundle.sample_rate
    for p in model.parameters():
        p.requires_grad_(False)

    def get_repr(wav: torch.Tensor) -> torch.Tensor:
        hlist, _ = model.extract_features(wav)
        h = hlist[layer].squeeze(0).transpose(0, 1)
        if downsample and downsample > 1:
            T = h.shape[1] // downsample * downsample
            h = h[:, :T].view(h.shape[0], -1, downsample).mean(-1)
        return h.cpu()

    with proto_fp.open("r", encoding="utf8") as f:
        utt_ids = [ln.split()[1] for ln in f]

    for utt_id in tqdm(utt_ids, desc=f"{access_type}-{part}", ncols=80):
        out_fp = out_dir / f"{utt_id}.pt"
        if out_fp.is_file():
            continue
        wav_fp = audio_dir / f"{utt_id}.flac"
        wav, sr = robust_load(wav_fp, SR)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.to(device)
        with torch.inference_mode():
            feat = get_repr(wav)
        torch.save(feat, out_fp)

# ------------------------------------------------------------------------- #
# 4) CLI ve Jupyter/Spyder girişi
# ------------------------------------------------------------------------- #

def _cli():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--access_type", choices=["LA", "PA"], required=True)
    p.add_argument("--part", choices=["train", "dev", "eval"])
    p.add_argument("--all", action="store_true")
    p.add_argument("--protocol_dir", required=True)
    p.add_argument("--audio_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--bundle", default="WAVLM_LARGE")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--downsample", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    parts = ["train", "dev", "eval"] if args.all else [args.part]
    for part in parts:
        extract_partition(
            access_type=args.access_type,
            part=part,
            protocol_dir=args.protocol_dir,
            audio_root=args.audio_root,
            output_dir=args.output_dir,
            bundle_name=args.bundle,
            layer=args.layer,
            downsample=args.downsample,
            device=args.device,
        )

if __name__ == "__main__":
    # PyCharm / Terminal → CLI
    if "PYCHARM_HOSTED" not in os.environ and not any(k.endswith("JPY_PARENT_PID") for k in os.environ):
        _cli()
    # Jupyter / Spyder → Otomatik döngü
    else:
        PARTS = ["train","dev","eval"]
        PARAMS = {
            "access_type": "PA",
            "protocol_dir": r"E:\akademikcalismalar\POST\DeepFakeAudio\DATASETLER\ASV2019\PA\ASVspoof2019_PA_cm_protocols",
            "audio_root": r"E:\akademikcalismalar\POST\DeepFakeAudio\DATASETLER\ASV2019",
            "output_dir": r"E:\akademikcalismalar\POST\DeepFakeAudio\DATASETLER\ASV2019Features\HUBERT_LARGE_L8",
            "bundle_name": "HUBERT_LARGE",
            "layer": 8,
            "downsample": None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        for part in PARTS:
            print(f"\n>>> İşleniyor: {part}\n")
            extract_partition(part=part, **PARAMS)
