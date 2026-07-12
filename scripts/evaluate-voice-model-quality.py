#!/usr/bin/env python3
"""Score model-comparison clips for speaker similarity and intelligibility."""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from jiwer import wer
from resemblyzer import VoiceEncoder, preprocess_wav


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "public/audio/short-unseen-tests/english-samantha.wav"
OUTPUT_DIR = ROOT / "public/audio/model-comparison/outputs"
REFERENCE_DIR = ROOT / "public/audio/model-comparison/references"
EXPECTED = (
    "On Thursday morning, the research team carried a small recorder across the quiet harbor. "
    "They compared each signal carefully, noted the changing wind, and repeated the final "
    "sentence twice so the experiment would remain clear and useful."
)
TARGETS = {
    "lj-speech": REFERENCE_DIR / "lj-speech-linda-johnson.wav",
    "cmu-rms": REFERENCE_DIR / "cmu-arctic-rms.wav",
    "llvc-target": REFERENCE_DIR / "llvc-fixed-target.wav",
}


def normalize(text):
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def target_for(path):
    for target in TARGETS:
        if path.stem.endswith(target):
            return target
    raise ValueError(f"Cannot infer target from {path.name}")


def model_for(path):
    for suffix in TARGETS:
        marker = f"-{suffix}"
        if path.stem.endswith(marker):
            return path.stem[: -len(marker)]
    raise ValueError(path)


def transcribe(model, path):
    segments, info = model.transcribe(str(path), language="en", beam_size=5, vad_filter=False)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text, info.duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/voice-model-quality.json")
    parser.add_argument("--whisper-model", default="base.en")
    args = parser.parse_args()
    files = sorted(OUTPUT_DIR.glob("*.wav"))
    if not files:
        raise SystemExit("No model-comparison outputs found")

    print("Loading independent speaker encoder...", flush=True)
    speaker = VoiceEncoder(device="cpu")
    reference_embeddings = {
        target: speaker.embed_utterance(preprocess_wav(path)) for target, path in TARGETS.items()
    }
    print(f"Loading Whisper {args.whisper_model}...", flush=True)
    whisper = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")
    source_transcript, _ = transcribe(whisper, SOURCE)
    normalized_expected = normalize(EXPECTED)
    results = []
    for index, path in enumerate(files, 1):
        target = target_for(path)
        embedding = speaker.embed_utterance(preprocess_wav(path))
        similarity = float(np.dot(embedding, reference_embeddings[target]))
        transcript, duration = transcribe(whisper, path)
        score = {
            "model": model_for(path),
            "target": target,
            "output": str(path.relative_to(ROOT)),
            "speakerCosineSimilarity": similarity,
            "transcript": transcript,
            "wordErrorRate": float(wer(normalized_expected, normalize(transcript))),
            "durationSeconds": duration,
        }
        results.append(score)
        print(
            f"[{index}/{len(files)}] {score['model']:18} {target:11} "
            f"speaker={similarity:.3f} WER={score['wordErrorRate']:.3f}",
            flush=True,
        )

    aggregates = {}
    for model_name in sorted({item["model"] for item in results}):
        selected = [item for item in results if item["model"] == model_name]
        aggregates[model_name] = {
            "voices": len(selected),
            "meanSpeakerCosineSimilarity": float(np.mean([item["speakerCosineSimilarity"] for item in selected])),
            "meanWordErrorRate": float(np.mean([item["wordErrorRate"] for item in selected])),
        }
    report = {
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": {
            "speakerEvaluator": "Resemblyzer 0.1.4, independent GE2E speaker encoder",
            "asrEvaluator": f"faster-whisper {args.whisper_model}, CPU INT8, beam size 5",
            "expectedTranscript": EXPECTED,
            "sourceAsrTranscript": source_transcript,
            "sourceAsrWordErrorRate": float(wer(normalized_expected, normalize(source_transcript))),
            "warning": "Automatic metrics are screening signals, not a substitute for human MOS/listening tests.",
        },
        "results": results,
        "aggregates": aggregates,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
