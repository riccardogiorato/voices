#!/usr/bin/env python3
"""Evaluate the distilled browser model against the full teacher on long clips."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "public/audio/long-tests"
LANGUAGES = ("english", "italian", "spanish", "french")


def session(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def states(enc_dim, dec_dim):
    return {
        "enc_state": np.zeros((1, enc_dim, 510), np.float32),
        "dec_state": np.zeros((1, 2, 13, dec_dim), np.float32),
        "out_state": np.zeros((1, enc_dim, 4), np.float32),
        "conv_state": np.zeros((1, 1, 24), np.float32),
    }


def convert(model, audio, enc_dim, dec_dim):
    current = states(enc_dim, dec_dim)
    context = np.zeros(32, np.float32)
    outputs = []
    names = [output.name for output in model.get_outputs()]
    for offset in range(0, len(audio) - 208, 208):
        samples = audio[offset:offset + 208]
        model_input = np.concatenate([context, samples]).reshape(1, 1, 240).astype(np.float32)
        values = model.run(None, {"audio": model_input, **current})
        result = dict(zip(names, values))
        outputs.append(result["converted"].reshape(-1))
        current = {
            "enc_state": result["enc_state_next"],
            "dec_state": result["dec_state_next"],
            "out_state": result["out_state_next"],
            "conv_state": result["conv_state_next"],
        }
        context = samples[-32:]
    return np.concatenate(outputs)


def spectrum(audio, size=256, hop=64):
    frames = np.lib.stride_tricks.sliding_window_view(audio, size)[::hop] * np.hanning(size)
    return np.abs(np.fft.rfft(frames, axis=1))


def metrics(teacher, student):
    teacher_spectrum, student_spectrum = spectrum(teacher), spectrum(student)
    return {
        "mae": float(np.mean(np.abs(teacher - student))),
        "mse": float(np.mean((teacher - student) ** 2)),
        "correlation": float(np.corrcoef(teacher, student)[0, 1]),
        "spectralError": float(np.linalg.norm(teacher_spectrum - student_spectrum) / (np.linalg.norm(teacher_spectrum) + 1e-12)),
        "logMagnitudeError": float(np.mean(np.abs(np.log1p(20 * teacher_spectrum) - np.log1p(20 * student_spectrum)))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-audio", action="store_true")
    args = parser.parse_args()
    teacher_model = session(ROOT / "public/models/common-voice-llvc.onnx")
    student_model = session(ROOT / "public/models/common-voice-llvc-student128.onnx")
    results = {}
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for language in LANGUAGES:
            raw = temporary / f"{language}.f32"
            subprocess.run([
                "ffmpeg", "-loglevel", "error", "-y", "-i", str(SAMPLE_DIR / f"{language}.wav"),
                "-ar", "16000", "-ac", "1", "-f", "f32le", str(raw),
            ], check=True)
            audio = np.fromfile(raw, dtype=np.float32)
            teacher = convert(teacher_model, audio, 512, 256)
            student = convert(student_model, audio, 128, 64)
            results[language] = metrics(teacher, student)
            if args.write_audio:
                student_raw = temporary / f"{language}-student.f32"
                student.astype(np.float32).tofile(student_raw)
                subprocess.run([
                    "ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ar", "16000", "-ac", "1",
                    "-i", str(student_raw), "-ar", "24000", "-c:a", "pcm_s16le",
                    str(SAMPLE_DIR / f"{language}-student.wav"),
                ], check=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
