

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

import torch
import torchaudio
from tqdm import tqdm


def extract_partition(
    *,
    access_type: str,
    part: str,
    protocol_dir: str | Path,
    audio_root: str | Path,
    output_dir: str | Path,
    bundle_name: str = "HUBERT_LARGE",  # <-- varsayılan artık Wav2Vec 2.0
    layer: int = 8,
    downsample: Optional[int] = None,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    
    protocol_dir = Path(protocol_dir)
    audio_root = Path(audio_root)
    output_dir = Path(output_dir)

    proto_fp = protocol_dir / f"ASVspoof2019.{access_type}.cm.{part}.trl.txt"
    if not proto_fp.is_file():
        raise FileNotFoundError(proto_fp)

    audio_dir = (
        audio_root
        / access_type
        / f"ASVspoof2019_{access_type}_{part}"
        / "flac"
    )
    if not audio_dir.is_dir():
        raise FileNotFoundError(audio_dir)

    out_dir = output_dir / access_type / part
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- model -------------------------------------------------------------
    try:
        bundle = getattr(torchaudio.pipelines, bundle_name)
    except AttributeError as exc:
        raise ValueError(
            f"{bundle_name} not found in torchaudio.pipelines.* — geçerli isim için torchaudio.pipelines.list_pretrained() kullanın"
        ) from exc

    model = bundle.get_model().to(device).eval()
    sr_bundle = bundle.sample_rate
    for p in model.parameters():
        p.requires_grad_(False)

    def get_repr(wav: torch.Tensor) -> torch.Tensor:
        hlist, _ = model.extract_features(wav)
        h = hlist[layer].squeeze(0).transpose(0, 1)  # (C, T)
        if downsample and downsample > 1:
            T = h.shape[1] // downsample * downsample
            h = h[:, :T].view(h.shape[0], -1, downsample).mean(-1)
        return h.cpu()

    # ---- iterate -----------------------------------------------------------
    with proto_fp.open("r", encoding="utf8") as f:
        utt_ids = [ln.split()[1] for ln in f]

    for utt_id in tqdm(utt_ids, desc=f"{access_type}-{part}", ncols=80):
        out_fp = out_dir / f"{utt_id}.pt"
        if out_fp.is_file():
            continue  # skip cached
        wav_fp = audio_dir / f"{utt_id}.flac"
        if not wav_fp.is_file():
            tqdm.write(f"Missing: {wav_fp}")
            continue
        wav, sr = torchaudio.load(str(wav_fp))
        if sr != sr_bundle:
            wav = torchaudio.functional.resample(wav, sr, sr_bundle)
        if wav.shape[0] > 1:  # stereo → mono
            wav = wav.mean(0, keepdim=True)
        wav = wav.to(device)
        with torch.inference_mode():
            h = get_repr(wav)
        torch.save(h, out_fp)


def _cli():
    """Komut satırı arayüzü. Tek bir part veya tüm partlar işlendirilebilir."""
    import argparse

    p = argparse.ArgumentParser(
        "Wav2Vec / HuBERT / WavLM vb. özellik çıkarıcı (batch mode)")
    p.add_argument("--access_type", choices=["LA", "PA"], required=True)

    # İki farklı kullanım: --part <train|dev|eval> veya --all
    p.add_argument("--part", choices=["train", "dev", "eval"], default=None,
                   help="İşlenecek part (train/dev/eval). --all kullanırsanız gerekmez.")
    p.add_argument("--all", action="store_true",
                   help="train, dev ve eval partlarının hepsini sırayla işle")

    p.add_argument("--protocol_dir", required=True)
    p.add_argument("--audio_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--bundle", default="WAV2VEC2_BASE")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--downsample", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # -- hangi part(lar) işlenecek? -----------------------------------------
    if not args.all and args.part is None:
        p.error("Either --part or --all must be specified")

    parts: List[str] = ["train", "dev", "eval"] if args.all else [args.part]

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



# -------------------------------------------------------------------------
PARAMS_COMMON = {
    "access_type": "LA",  # "LA" veya "PA"
    "protocol_dir": r"E:\akademikcalismalar\POST\DeepFakeAudio\DATASETLER\ASV2019\LA\LA\ASVspoof2019_LA_cm_protocols",
    "audio_root":   r"E:\akademikcalismalar\POST\DeepFakeAudio\DATASETLER\ASV2019",
    "output_dir":   r"E:\akademikcalismalar\POST\DeepFakeAudio\DATASETLER\ASV2019Features\HUBERT_LARGE_L8",
    "bundle_name": "HUBERT_LARGE",  # WAV2VEC2_BASE, WAV2VEC2_LARGE, WAV2VEC2_XLSR53, HUBERT_BASE, WAVLM_BASE
    "layer": 8,
    "downsample": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

PARTS_TO_RUN: List[str] = ["train", "dev", "eval"]  


if __name__ == "__main__":

    # Terminal kullanımıysa doğrudan _cli'ye delege et
    if "PYCHARM_HOSTED" not in os.environ and not any(
        key.endswith("JPY_PARENT_PID") for key in os.environ):
        _cli()

    # Aksi hâlde (Jupyter/Spyder): tanımlı listeden sırayla işle
    else:
        for _part in PARTS_TO_RUN:
            print(f"\n>>> İşleniyor: {_part}\n")
            extract_partition(part=_part, **PARAMS_COMMON)
