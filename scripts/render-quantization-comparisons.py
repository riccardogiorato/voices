#!/usr/bin/env python3
"""Render the original teacher and the two selected PTQ variants for listening."""

import argparse
import importlib.util
import tempfile
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
module_spec = importlib.util.spec_from_file_location("variant_evaluation", ROOT / "scripts/evaluate-model-variants.py")
evaluation = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(evaluation)
OUTPUT = ROOT / "public/audio/quantization-comparison"
SELECTED = ("fp32", "q8-high", "q8-max")


def write_wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", default="public/audio/long-tests")
    parser.add_argument("--output-dir", default=str(OUTPUT.relative_to(ROOT)))
    args = parser.parse_args()
    sample_dir = ROOT / args.sample_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    models = {name: evaluation.session(evaluation.MODELS[name]) for name in SELECTED}
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for sample in sorted(sample_dir.glob("*.wav")):
            audio = evaluation.decode(sample, temporary / f"{sample.stem}.f32")
            for name, model in models.items():
                converted, _ = evaluation.convert(model, audio)
                path = output_dir / f"{sample.stem}-{name}.wav"
                write_wav(path, converted)
                print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
