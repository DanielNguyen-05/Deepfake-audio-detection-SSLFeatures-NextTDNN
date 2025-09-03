
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

import torch
import torchaudio
from tqdm import tqdm


def extract_partition(
    *,
    part: str,
    protocol_dir: str | Path,
    audio_root: str | Path,
    output_dir: str | Path,
    bundle_name: str = "WAVLM_LARGE",  # e.g. HUBERT_BASE, WAV2VEC2_BASE
    layer: int = 8,
    downsample: Optional[int] = None,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Extract features for one partition (train/dev/eval).

    Parameters
    ----------
    part : {{"train", "dev", "eval"}}
        Dataset partition to process.
    protocol_dir : Pathlike
        Directory containing ``ASVspoof5.*.tsv`` protocol files.
    audio_root : Pathlike
        Root directory with unpacked ``flac_T``, ``flac_D`` and ``flac_E_eval``
        sub‑directories.
    output_dir : Pathlike
        Where ``*.pt`` feature files will be written (mirrors *part* hierarchy).
    bundle_name : str, optional
        Name of the torchaudio pre‑trained bundle.
    layer : int, optional
        Which hidden layer to take (HuBERT has 12 for BASE, 24 for LARGE/XLARGE).
    downsample : int | None, optional
        Temporal mean‑pooling factor.  ``None`` keeps native frame rate.
    device : str | torch.device, optional
        Compute device ("cuda" / "cpu").
    """

    protocol_dir = Path(protocol_dir)
    audio_root = Path(audio_root)
    output_dir = Path(output_dir)

    # ---------------------------------------------------------------------
    # 1. Map part → protocol filename & audio sub‑folder
    # ---------------------------------------------------------------------
    proto_map = {
        "train": "ASVspoof5.train.tsv",
        "dev": "ASVspoof5.dev.track_1.tsv",
        "eval": "ASVspoof5.eval.track_1.tsv",
    }
    audio_map = {
        # "train": "flac_T",
        # "dev": "flac_D",
        "eval": "flac_E_eval",
    }
    if part not in proto_map:
        raise ValueError(f"Unknown part '{part}'. Choose from {list(proto_map)}")

    proto_fp = protocol_dir / proto_map[part]
    if not proto_fp.is_file():
        raise FileNotFoundError(proto_fp)

    audio_dir = audio_root / audio_map[part]
    if not audio_dir.is_dir():
        raise FileNotFoundError(audio_dir)

    out_dir = output_dir / part
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 2. Load pre‑trained model bundle (HuBERT / Wav2Vec 2.0 / WavLM …)
    # ---------------------------------------------------------------------
    try:
        bundle = getattr(torchaudio.pipelines, bundle_name)
    except AttributeError as exc:
        raise ValueError(
            f"'{bundle_name}' bulunamadı.  Geçerli isimler için\n"
            "  >>> import torchaudio, pprint; pprint.pp(torchaudio.pipelines.list_pretrained())"
        ) from exc

    model = bundle.get_model().to(device).eval()
    sr_bundle = bundle.sample_rate

    for p in model.parameters():
        p.requires_grad_(False)

    def get_repr(wav: torch.Tensor) -> torch.Tensor:
        """Forward pass → hidden layer representation."""
        # extract_features returns list[List[Tensor]] where outer len==num_layers+1
        hlist, _ = model.extract_features(wav)
        h = hlist[layer].squeeze(0).transpose(0, 1)  # (C, T)
        if downsample and downsample > 1:
            T = h.shape[1] // downsample * downsample
            h = h[:, :T].view(h.shape[0], -1, downsample).mean(-1)
        return h.cpu()

    # ---------------------------------------------------------------------
    # 3. Iterate through utterances listed in protocol file
    # ---------------------------------------------------------------------
    with proto_fp.open("r", encoding="utf8") as f:
        # Column‑2 = FLAC_FILE_NAME
        utt_ids: List[str] = [ln.split()[1] for ln in f]

    for utt_id in tqdm(utt_ids, desc=part, ncols=80):
        out_fp = out_dir / f"{utt_id}.pt"
        if out_fp.is_file():  # cached
            continue

        wav_fp = audio_dir / f"{utt_id}.flac"
        if not wav_fp.is_file():
            tqdm.write(f"★ Missing: {wav_fp}")
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


# -------------------------------------------------------------------------
# CLI helper
# -------------------------------------------------------------------------

def _cli():
    """Command‑line entry point."""
    import argparse

    p = argparse.ArgumentParser(
        "ASVspoof5 feature extractor (HuBERT / Wav2Vec2 / WavLM)")
    p.add_argument("--part", choices=["train", "dev", "eval"], required=True,
                   help="Which partition to process")
    p.add_argument("--protocol_dir", required=True,
                   help="Directory with ASVspoof5.*.tsv protocol files")
    p.add_argument("--audio_root", required=True,
                   help="Directory with flac_T / flac_D / flac_E")
    p.add_argument("--output_dir", required=True,
                   help="Where extracted .pt files will be stored")
    # p.add_argument("--bundle", default="HUBERT_LARGE",
    #                help="torchaudio bundle name, e.g. HUBERT_BASE, WAVLM_BASE…")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--downsample", type=int, default=None,
                   help="Temporal mean‑pool factor (e.g. 3 → fps/3)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    extract_partition(
        part=args.part,
        protocol_dir=args.protocol_dir,
        audio_root=args.audio_root,
        output_dir=args.output_dir,
        bundle_name=args.bundle,
        layer=args.layer,
        downsample=args.downsample,
        device=args.device,
    )


# -------------------------------------------------------------------------
# Example notebook usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # When launched from terminal execute CLI; inside IDE/Jupyter run loop below
    if "PYCHARM_HOSTED" not in os.environ and not any(k.endswith("JPY_PARENT_PID") for k in os.environ):
        _cli()
    else:
        PARAMS_COMMON = {
            "protocol_dir": "E:/akademikcalismalar/POST/DeepFakeAudio/DATASETLER/ASVSPOOF5/ASVspoof5_protocols",
            "audio_root": "E:/akademikcalismalar/POST/DeepFakeAudio/DATASETLER/ASVSPOOF5",
            "output_dir": "E:/akademikcalismalar/POST/DeepFakeAudio/DATASETLER/ASVSPOOF5/features/WAVLM_LARGE_L8",
            "bundle_name": "WAVLM_LARGE",
            "layer": 8,
            "downsample": None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        for _part in ["eval"]:
            print(f"\n>>> Processing {_part}…")
            extract_partition(part=_part, **PARAMS_COMMON)
