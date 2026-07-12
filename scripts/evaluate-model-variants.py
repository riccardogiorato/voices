#!/usr/bin/env python3
"""Compare no-training LLVC precision variants with the original FP32 teacher."""

import argparse
import json
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "public/models"
SHORT_DIR = ROOT / "public/audio"
LONG_DIR = SHORT_DIR / "long-tests"
MODELS = {
    "fp32": MODEL_DIR / "common-voice-llvc.onnx",
    "fp16": MODEL_DIR / "common-voice-llvc-fp16.onnx",
    "q8-high": MODEL_DIR / "common-voice-llvc-q8-high.onnx",
    "q8-balanced": MODEL_DIR / "common-voice-llvc-q8-balanced.onnx",
    "q8-max": MODEL_DIR / "common-voice-llvc-q8-max.onnx",
}
SAMPLES = [
    *(('bundled', path) for path in sorted(SHORT_DIR.glob("*.wav"))),
    *(('long-unseen', path) for path in sorted(LONG_DIR.glob("*.wav"))),
]


def session(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def initial_states():
    return {
        "enc_state": np.zeros((1, 512, 510), np.float32),
        "dec_state": np.zeros((1, 2, 13, 256), np.float32),
        "out_state": np.zeros((1, 512, 4), np.float32),
        "conv_state": np.zeros((1, 1, 24), np.float32),
    }


def decode(path, output):
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(path),
        "-ar", "16000", "-ac", "1", "-f", "f32le", str(output),
    ], check=True)
    return np.fromfile(output, dtype=np.float32)


def convert(model, audio):
    current = initial_states()
    context = np.zeros(32, np.float32)
    outputs = []
    times = []
    names = [output.name for output in model.get_outputs()]
    for offset in range(0, len(audio) - 208, 208):
        samples = audio[offset:offset + 208]
        model_input = np.concatenate([context, samples]).reshape(1, 1, 240).astype(np.float32)
        started = time.perf_counter()
        values = model.run(None, {"audio": model_input, **current})
        times.append((time.perf_counter() - started) * 1000)
        result = dict(zip(names, values))
        outputs.append(result["converted"].reshape(-1))
        current = {
            "enc_state": result["enc_state_next"],
            "dec_state": result["dec_state_next"],
            "out_state": result["out_state_next"],
            "conv_state": result["conv_state_next"],
        }
        context = samples[-32:]
    return np.concatenate(outputs), np.asarray(times)


def spectrum(audio, size=256, hop=128):
    frames = np.lib.stride_tricks.sliding_window_view(audio, size)[::hop] * np.hanning(size)
    return np.abs(np.fft.rfft(frames, axis=1))


def metrics(reference, candidate):
    reference_spectrum, candidate_spectrum = spectrum(reference), spectrum(candidate)
    delta = reference - candidate
    reference_energy = np.sum(reference ** 2) + 1e-12
    noise_energy = np.sum(delta ** 2) + 1e-12
    return {
        "mae": float(np.mean(np.abs(delta))),
        "mse": float(np.mean(delta ** 2)),
        "correlation": float(np.corrcoef(reference, candidate)[0, 1]),
        "snrDb": float(10 * np.log10(reference_energy / noise_energy)),
        "spectralError": float(np.linalg.norm(reference_spectrum - candidate_spectrum) / (np.linalg.norm(reference_spectrum) + 1e-12)),
        "logMagnitudeError": float(np.mean(np.abs(np.log1p(20 * reference_spectrum) - np.log1p(20 * candidate_spectrum)))),
        "peak": float(np.max(np.abs(candidate))),
        "rms": float(np.sqrt(np.mean(candidate ** 2))),
        "finite": bool(np.isfinite(candidate).all()),
    }


def timing(times):
    return {
        "averageMs": float(np.mean(times)),
        "p99Ms": float(np.percentile(times, 99)),
        "maxMs": float(np.max(times)),
        "chunks": int(len(times)),
    }


def model_size(name, path):
    total = path.stat().st_size
    external = Path(f"{path}.data")
    if external.exists():
        total += external.stat().st_size
    return {"bytes": total, "megabytes": total / 1_000_000}


def aggregate(samples, category):
    selected = [value for value in samples if category == "all" or value["category"] == category]
    keys = ("mae", "mse", "correlation", "snrDb", "spectralError", "logMagnitudeError")
    return {key: float(np.mean([value["quality"][key] for value in selected])) for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/quantization-evaluation.json")
    parser.add_argument("--sample-dir", help="Evaluate only WAV files in this directory as short-unseen")
    parser.add_argument("--sample-pattern", default="*.wav", help="Glob used with --sample-dir")
    parser.add_argument("--models", default=",".join(MODELS), help="Comma-separated model names; fp32 must be included")
    args = parser.parse_args()
    model_names = args.models.split(",")
    if "fp32" not in model_names or any(name not in MODELS for name in model_names):
        raise SystemExit(f"Models must include fp32 and use: {', '.join(MODELS)}")
    selected_models = {name: MODELS[name] for name in model_names}
    samples = SAMPLES if not args.sample_dir else [
        ("short-unseen", path) for path in sorted((ROOT / args.sample_dir).glob(args.sample_pattern))
    ]
    if not samples:
        raise SystemExit("No WAV samples found")
    loaded = {name: session(path) for name, path in selected_models.items()}
    results = {name: [] for name in selected_models}
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for sample_index, (category, sample) in enumerate(samples, 1):
            audio = decode(sample, temporary / f"{sample_index}.f32")
            print(f"[{sample_index}/{len(samples)}] {category}/{sample.name}: {len(audio) / 16000:.2f}s", flush=True)
            outputs = {}
            for name, model in loaded.items():
                converted, times = convert(model, audio)
                outputs[name] = converted
                quality = metrics(converted, converted) if name == "fp32" else metrics(outputs["fp32"], converted)
                results[name].append({
                    "sample": sample.stem,
                    "category": category,
                    "durationSeconds": len(audio) / 16000,
                    "quality": quality,
                    "nativeTiming": timing(times),
                })
                print(f"  {name:12} avg={np.mean(times):6.3f}ms corr={quality['correlation']:.6f} snr={quality['snrDb']:7.2f}dB", flush=True)

    reference_bytes = model_size("fp32", selected_models["fp32"])["bytes"]
    report = {
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": {
            "training": False,
            "reference": "Original FP32 teacher, stateful 208-sample chunks at 16 kHz",
            "sampleCount": len(samples),
            "categories": {category: sum(value == category for value, _ in samples) for category in sorted({value for value, _ in samples})},
            "qualityReference": "Each variant compared sample-for-sample with FP32 output",
        },
        "models": {},
    }
    for name, path in selected_models.items():
        size = model_size(name, path)
        report["models"][name] = {
            "path": str(path.relative_to(ROOT)),
            "size": {**size, "fractionOfFp32": size["bytes"] / reference_bytes},
            "samples": results[name],
            "aggregate": {
                category: aggregate(results[name], category)
                for category in (*sorted({value for value, _ in samples}), "all")
            },
        }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
