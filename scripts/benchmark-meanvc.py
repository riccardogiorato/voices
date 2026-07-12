#!/usr/bin/env python3
"""Run the official MeanVC checkpoint across the shared target-voice matrix."""

import argparse
import json
import os
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "public/audio/short-unseen-tests/english-samantha.wav"
REFERENCE_DIR = ROOT / "public/audio/model-comparison/references"
OUTPUT_DIR = ROOT / "public/audio/model-comparison/outputs"
TARGETS = {
    "lj-speech": REFERENCE_DIR / "lj-speech-linda-johnson.wav",
    "cmu-rms": REFERENCE_DIR / "cmu-arctic-rms.wav",
    "llvc-target": REFERENCE_DIR / "llvc-fixed-target.wav",
}


def wav_stats(path):
    with wave.open(str(path), "rb") as audio:
        rate, frames = audio.getframerate(), audio.getnframes()
        values = np.frombuffer(audio.readframes(frames), dtype="<i2").astype(np.float32) / 32768
    return {
        "durationSeconds": frames / rate,
        "sampleRate": rate,
        "peak": float(np.max(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values**2))),
        "finite": bool(np.isfinite(values).all()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meanvc-root", default=os.environ.get("MEANVC_ROOT", "/tmp/voices-model-bench/MeanVC"))
    parser.add_argument("--output", default="reports/meanvc-benchmark.json")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=20)
    args = parser.parse_args()
    meanvc_root = Path(args.meanvc_root).resolve()
    sys.path.insert(0, str(meanvc_root))
    sys.path.insert(0, str(meanvc_root / "src/infer"))

    from src.infer.dit_kvcache import DiT
    from src.infer.infer_ref import MelSpectrogramFeatures, extract_features_from_audio, inference, setup_seed
    from src.model.utils import load_checkpoint
    from src.runtime.speaker_verification.verification import init_model as init_sv_model

    device = "cpu"
    with (meanvc_root / "src/config/config_200ms.json").open() as handle:
        config = json.load(handle)
    load_started = time.perf_counter()
    model = DiT(**config["model"]).to(device)
    model = load_checkpoint(model, str(meanvc_root / "src/ckpt/model_200ms.safetensors"), device=device, use_ema=False).float().eval()
    vocos = torch.jit.load(meanvc_root / "src/ckpt/vocos.pt").to(device)
    asr_model = torch.jit.load(meanvc_root / "src/ckpt/fastu2++.pt").to(device)
    sv_model = init_sv_model("wavlm_large", meanvc_root / "src/runtime/speaker_verification/ckpt/wavlm_large_finetune.pth").to(device).eval()
    mel_extractor = MelSpectrogramFeatures(
        sample_rate=16000, n_fft=1024, win_size=640, hop_length=160,
        n_mels=80, fmin=0, fmax=8000, center=True,
    ).to(device)
    load_seconds = time.perf_counter() - load_started

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for target, reference in TARGETS.items():
        setup_seed(42)
        started = time.perf_counter()
        bn, embedding, prompt = extract_features_from_audio(
            str(SOURCE), str(reference), asr_model, sv_model, mel_extractor, device
        )
        feature_seconds = time.perf_counter() - started
        _, output_audio, model_seconds = inference(
            model, vocos, bn, embedding, prompt, args.chunk_size, args.steps, device
        )
        output = OUTPUT_DIR / f"meanvc-{target}.wav"
        torchaudio.save(str(output), output_audio.cpu(), 16000, encoding="PCM_S", bits_per_sample=16)
        total_seconds = time.perf_counter() - started
        stats = wav_stats(output)
        run = {
            "target": target,
            "reference": str(reference.relative_to(ROOT)),
            "output": str(output.relative_to(ROOT)),
            "featureSeconds": feature_seconds,
            "modelAndVocoderSeconds": model_seconds,
            "totalSeconds": total_seconds,
            "modelRtf": model_seconds / stats["durationSeconds"],
            "endToEndRtf": total_seconds / stats["durationSeconds"],
            "audio": stats,
        }
        runs.append(run)
        print(
            f"meanvc {target:11} features={feature_seconds:6.2f}s model={model_seconds:6.2f}s "
            f"total={total_seconds:6.2f}s end-to-end RTF={run['endToEndRtf']:.3f}",
            flush=True,
        )

    model_files = [
        meanvc_root / "src/ckpt/model_200ms.safetensors",
        meanvc_root / "src/ckpt/meanvc_200ms.pt",
        meanvc_root / "src/ckpt/fastu2++.pt",
        meanvc_root / "src/ckpt/vocos.pt",
        meanvc_root / "src/runtime/speaker_verification/ckpt/wavlm_large_finetune.pth",
        Path.home() / ".cache/s3prl/download/f2d5200177fd6a33b278b7b76b454f25cd8ee866d55c122e69fccf6c7467d37d.wavlm_large.pt",
    ]
    report = {
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(SOURCE.relative_to(ROOT)),
        "precision": "official FP32 PyTorch checkpoint",
        "loadSeconds": load_seconds,
        "advertisedCheckpointBytes": sum(path.stat().st_size for path in model_files[:4]),
        "actualRuntimeArtifactBytes": sum(
            path.stat().st_size for index, path in enumerate(model_files) if path.exists() and index != 1
        ),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "steps": args.steps,
        "chunkSize": args.chunk_size,
        "runs": runs,
        "compatibilityPatch": "Removed unused Trainer imports from temporary src/model/__init__.py to avoid training-only dependencies.",
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
