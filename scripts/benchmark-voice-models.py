#!/usr/bin/env python3
"""Benchmark zero-shot ONNX voice converters and the fixed-target LLVC baseline."""

import argparse
import json
import os
import resource
import subprocess
import tempfile
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from voiceclonnx import VoiceCloner


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
        channels = audio.getnchannels()
        rate = audio.getframerate()
        width = audio.getsampwidth()
        frames = audio.getnframes()
        raw = audio.readframes(frames)
    if width != 2:
        raise ValueError(f"Expected PCM16 WAV, got {width * 8}-bit: {path}")
    values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    return {
        "durationSeconds": frames / rate,
        "sampleRate": rate,
        "channels": channels,
        "peak": float(np.max(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values**2))),
        "finite": bool(np.isfinite(values).all()),
    }


def decode(path, output):
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-ar", "16000", "-ac", "1", "-f", "f32le", str(output)],
        check=True,
    )
    return np.fromfile(output, dtype=np.float32)


def initial_llvc_states():
    return {
        "enc_state": np.zeros((1, 512, 510), np.float32),
        "dec_state": np.zeros((1, 2, 13, 256), np.float32),
        "out_state": np.zeros((1, 512, 4), np.float32),
        "conv_state": np.zeros((1, 1, 24), np.float32),
    }


def run_llvc(source, output):
    model_path = ROOT / "public/models/common-voice-llvc-q8-high.onnx"
    started = time.perf_counter()
    model = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    session_seconds = time.perf_counter() - started
    with tempfile.TemporaryDirectory() as temporary:
        samples = decode(source, Path(temporary) / "source.f32")
    current = initial_llvc_states()
    context = np.zeros(32, np.float32)
    converted = []
    chunk_times = []
    names = [item.name for item in model.get_outputs()]
    started = time.perf_counter()
    for offset in range(0, len(samples) - 208, 208):
        chunk = samples[offset : offset + 208]
        inputs = np.concatenate([context, chunk]).reshape(1, 1, 240).astype(np.float32)
        chunk_started = time.perf_counter()
        values = model.run(None, {"audio": inputs, **current})
        chunk_times.append((time.perf_counter() - chunk_started) * 1000)
        result = dict(zip(names, values))
        converted.append(result["converted"].reshape(-1))
        current = {
            "enc_state": result["enc_state_next"],
            "dec_state": result["dec_state_next"],
            "out_state": result["out_state_next"],
            "conv_state": result["conv_state_next"],
        }
        context = chunk[-32:]
    inference_seconds = time.perf_counter() - started
    values = np.clip(np.concatenate(converted), -1, 1)
    with wave.open(str(output), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes((values * 32767).astype("<i2").tobytes())
    model_bytes = model_path.stat().st_size
    return {
        "target": "llvc-target",
        "output": str(output.relative_to(ROOT)),
        "sessionSeconds": session_seconds,
        "inferenceSeconds": inference_seconds,
        "rtf": inference_seconds / (len(values) / 16000),
        "chunkTimingMs": {
            "average": float(np.mean(chunk_times)),
            "p99": float(np.percentile(chunk_times, 99)),
            "maximum": float(np.max(chunk_times)),
        },
        "audio": wav_stats(output),
        "modelBytes": model_bytes,
    }


def cached_model_bytes(repo_fragment):
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    repos = list(cache.glob(f"models--*--*{repo_fragment}*"))
    return sum(path.stat().st_size for repo in repos for path in (repo / "blobs").glob("*") if path.is_file())


def run_zero_shot(engine):
    constructor_started = time.perf_counter()
    cloner = VoiceCloner(engine=engine, quantized=True)
    constructor_seconds = time.perf_counter() - constructor_started
    runs = []
    for index, (target, reference) in enumerate(TARGETS.items()):
        output = OUTPUT_DIR / f"{engine}-q8-{target}.wav"
        started = time.perf_counter()
        cloner.clone_voice(str(SOURCE), str(reference), str(output))
        elapsed = time.perf_counter() - started
        audio = wav_stats(output)
        runs.append({
            "target": target,
            "reference": str(reference.relative_to(ROOT)),
            "output": str(output.relative_to(ROOT)),
            "mode": "cold-session" if index == 0 else "warm-session",
            "wallSeconds": elapsed,
            "rtf": elapsed / audio["durationSeconds"],
            "audio": audio,
        })
        print(f"{engine:9} {target:11} {elapsed:7.2f}s RTF={runs[-1]['rtf']:.3f}", flush=True)
    return {
        "precision": "INT8/Q8 ONNX",
        "constructorSeconds": constructor_seconds,
        "cachedModelBytes": cached_model_bytes("openvoice-v2" if engine == "openvoice" else "facodec"),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/voice-model-benchmark.json")
    parser.add_argument("--engines", default="openvoice,facodec,llvc")
    args = parser.parse_args()
    engines = [item.strip() for item in args.engines.split(",")]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(SOURCE.relative_to(ROOT)),
        "sourceAudio": wav_stats(SOURCE),
        "targets": {name: str(path.relative_to(ROOT)) for name, path in TARGETS.items()},
        "method": {
            "networkDownloadsExcluded": True,
            "zeroShotOrder": list(TARGETS),
            "note": "First zero-shot conversion includes lazy ONNX session creation; following conversions reuse sessions.",
        },
        "models": {},
    }
    for engine in engines:
        if engine in ("openvoice", "facodec"):
            report["models"][f"{engine}-q8"] = run_zero_shot(engine)
        elif engine == "llvc":
            output = OUTPUT_DIR / "llvc-q8-high-llvc-target.wav"
            run = run_llvc(SOURCE, output)
            report["models"]["llvc-q8-high"] = {
                "precision": "static INT8 weights/activations with retained sensitive FP32 nodes",
                "targetMode": "fixed any-to-one; other targets require training",
                "runs": [run],
            }
            print(f"llvc      llvc-target {run['inferenceSeconds']:7.2f}s RTF={run['rtf']:.3f}", flush=True)
        else:
            raise SystemExit(f"Unknown engine: {engine}")
    report["peakProcessRssBytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname != "Darwin":
        report["peakProcessRssBytes"] *= 1024
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
