#!/usr/bin/env python3
"""End-to-end custom-speaker LLVC fine-tune, export, and Q8 optimization pipeline."""

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import wave
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".aiff", ".aif"}


def voice_slug(value):
    normalized = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    if not slug:
        raise argparse.ArgumentTypeError(
            "voice name must contain at least one ASCII letter or number"
        )
    return slug


def audio_files(paths):
    found = []
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_dir():
            found.extend(
                sorted(
                    item
                    for item in path.rglob("*")
                    if item.suffix.lower() in AUDIO_EXTENSIONS
                )
            )
        elif path.suffix.lower() in AUDIO_EXTENSIONS and path.exists():
            found.append(path)
        else:
            raise FileNotFoundError(f"Audio input not found or unsupported: {path}")
    return list(dict.fromkeys(found))


def decode(path, output, sample_rate=16000):
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-f",
            "f32le",
            str(output),
        ],
        check=True,
    )
    return np.fromfile(output, dtype=np.float32)


def write_pcm16(path, samples, sample_rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.clip(samples, -1, 1)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes((values * 32767).astype("<i2").tobytes())


def make_reference(target_files, output, max_seconds=30):
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        pieces = [
            decode(path, temporary / f"target-{index}.f32")
            for index, path in enumerate(target_files)
        ]
    joined = np.concatenate(pieces)[: int(max_seconds * 16000)]
    peak = float(np.max(np.abs(joined))) if len(joined) else 0
    if peak < 1e-4:
        raise ValueError("Target reference is silent")
    joined = joined * min(1.0, 0.95 / peak)
    write_pcm16(output, joined)
    return len(joined) / 16000


def split_sources(sources, seed):
    if len(sources) < 3:
        raise ValueError(
            "At least three source recordings are required for train/val/dev isolation"
        )
    shuffled = list(sources)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * 0.1))
    dev_count = max(1, round(len(shuffled) * 0.1))
    train_count = len(shuffled) - val_count - dev_count
    if train_count < 1:
        raise ValueError("Not enough source recordings after split")
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "dev": shuffled[train_count + val_count :],
    }


def prepare_parallel_dataset(
    target_files, sources, dataset_dir, reference, seed, force
):
    if dataset_dir.exists():
        if not force:
            raise FileExistsError(
                f"Dataset already exists: {dataset_dir}; pass --force to replace it"
            )
        shutil.rmtree(dataset_dir)
    reference_seconds = make_reference(target_files, reference)
    from voiceclonnx import VoiceCloner

    cloner = VoiceCloner(engine="openvoice", quantized=True)
    splits = split_sources(sources, seed)
    counts = {name: 0 for name in splits}
    source_manifest = {}
    chunk_samples = 65536
    minimum_samples = 16000
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for split, paths in splits.items():
            split_dir = dataset_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            source_manifest[split] = [str(path) for path in paths]
            for source_index, source in enumerate(paths):
                converted_path = temporary / f"{split}-{source_index}-converted.wav"
                cloner.clone_voice(str(source), str(reference), str(converted_path))
                original = decode(
                    source, temporary / f"{split}-{source_index}-original.f32"
                )
                converted = decode(
                    converted_path, temporary / f"{split}-{source_index}-converted.f32"
                )
                usable = min(len(original), len(converted))
                original, converted = original[:usable], converted[:usable]
                for offset in range(0, usable, chunk_samples):
                    end = min(offset + chunk_samples, usable)
                    if end - offset < minimum_samples:
                        continue
                    name = f"{counts[split]:06d}"
                    write_pcm16(
                        split_dir / f"{name}_original.wav", original[offset:end]
                    )
                    write_pcm16(
                        split_dir / f"{name}_converted.wav", converted[offset:end]
                    )
                    counts[split] += 1
                print(
                    f"prepared {split}: {source.name} ({counts[split]} cumulative pairs)",
                    flush=True,
                )
    if any(count == 0 for count in counts.values()):
        raise RuntimeError(f"Dataset split produced no chunks: {counts}")
    metadata = {
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": "OpenVoice V2 Q8 via voiceclonnx",
        "sampleRate": 16000,
        "chunkSamples": chunk_samples,
        "referenceSeconds": reference_seconds,
        "targetFiles": [str(path) for path in target_files],
        "sources": source_manifest,
        "pairCounts": counts,
        "warning": "Synthetic parallel targets inherit OpenVoice artifacts; inspect pairs before training.",
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def make_config(llvc_root, dataset_dir, output, batch_size, learning_rate, seed):
    template = json.loads((llvc_root / "experiments/llvc_nc/config.json").read_text())
    template["data"]["dir"] = str(dataset_dir.resolve())
    template["optim"]["lr"] = learning_rate
    template["batch_size"] = batch_size
    template["eval_batch_size"] = batch_size
    template["fp16_run"] = False
    template["aux_fairseq"]["c"] = 0
    template["aux_fairseq"]["checkpoint_path"] = ""
    template["seed"] = seed
    template["test_dir"] = str((ROOT / "public/audio/short-unseen-tests").resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(template, indent=2) + "\n")


def run(command):
    print("$ " + " ".join(str(item) for item in command), flush=True)
    subprocess.run([str(item) for item in command], check=True)


def artifact_info(path):
    paths = [path]
    external = Path(f"{path}.data")
    if external.exists():
        paths.append(external)
    digest = hashlib.sha256()
    size = 0
    for item in paths:
        size += item.stat().st_size
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "path": str(path),
        "bytes": size,
        "sha256": digest.hexdigest(),
        "files": [str(item) for item in paths],
    }


def update_state(path, **values):
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n")
    return state


def main():
    parser = argparse.ArgumentParser(
        description="Create a custom any-to-one LLVC model from a consented speaker reference."
    )
    parser.add_argument("--voice-name", required=True)
    parser.add_argument("--target-audio", type=Path, action="append")
    parser.add_argument("--source-audio", type=Path, action="append")
    parser.add_argument(
        "--paired-dataset",
        type=Path,
        help="Existing LLVC train/val/dev pairs; skips synthesis",
    )
    parser.add_argument("--llvc-root", type=Path, default=ROOT / "work/LLVC")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "public/models/custom"
    )
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda")
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "train", "export", "quantize"),
        default="all",
    )
    parser.add_argument(
        "--confirm-rights",
        action="store_true",
        help="Confirm permission to clone the target voice",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    slug = voice_slug(args.voice_name)
    llvc_root = args.llvc_root.resolve()
    workspace = (args.workspace or ROOT / "work/custom-voices" / slug).resolve()
    output_dir = args.output_dir.resolve()
    dataset_dir = (
        args.paired_dataset.resolve() if args.paired_dataset else workspace / "dataset"
    )
    reference = workspace / f"reference-{slug}.wav"
    config = workspace / "config.json"
    checkpoint = workspace / "checkpoints" / f"llvc-finetuned-{slug}.pth"
    base_checkpoint = (
        args.base_checkpoint
        or llvc_root / "llvc_models/models/checkpoints/llvc_nc/G_500000.pth"
    ).resolve()
    base_onnx = output_dir / f"llvc-base-{slug}.onnx"
    optimized_onnx = output_dir / f"llvc-optimized-q8-{slug}.onnx"
    manifest_path = output_dir / f"llvc-{slug}.json"
    state_path = workspace / "pipeline-state.json"
    q8_report = workspace / "q8-report.json"
    training_report = workspace / "training-report.json"
    default_sources = [
        ROOT / "public/audio/short-unseen-tests",
        ROOT / "public/audio/long-tests",
    ]

    plan = {
        "voiceName": args.voice_name,
        "slug": slug,
        "stage": args.stage,
        "workspace": str(workspace),
        "dataset": str(dataset_dir),
        "baseCheckpoint": str(base_checkpoint),
        "checkpoint": str(checkpoint),
        "baseOnnx": str(base_onnx),
        "optimizedQ8Onnx": str(optimized_onnx),
        "manifest": str(manifest_path),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    if args.stage in ("all", "prepare") and not args.target_audio:
        raise SystemExit("--target-audio is required for the prepare stage")
    if args.stage in ("all", "prepare") and not args.confirm_rights:
        raise SystemExit(
            "Pass --confirm-rights to confirm consent and usage rights for the target voice"
        )
    if args.stage == "all" and not args.force:
        existing = [
            path for path in (base_onnx, optimized_onnx, manifest_path) if path.exists()
        ]
        if existing:
            raise SystemExit(
                f"Output already exists: {existing[0]}; pass --force to replace this voice"
            )
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required")
    if not llvc_root.joinpath("model.py").exists():
        raise SystemExit(f"LLVC checkout not found: {llvc_root}")
    target_files = audio_files(args.target_audio) if args.target_audio else []

    if args.stage in ("all", "prepare"):
        if args.paired_dataset:
            for split in ("train", "val", "dev"):
                if not dataset_dir.joinpath(split).is_dir():
                    raise SystemExit(
                        f"Missing paired-dataset split: {dataset_dir / split}"
                    )
            workspace.mkdir(parents=True, exist_ok=True)
            reference_seconds = make_reference(target_files, reference)
            dataset_metadata = {
                "generator": "user-supplied pairs",
                "referenceSeconds": reference_seconds,
                "targetFiles": [str(path) for path in target_files],
            }
        else:
            sources = audio_files(args.source_audio or default_sources)
            dataset_metadata = prepare_parallel_dataset(
                target_files, sources, dataset_dir, reference, args.seed, args.force
            )
        make_config(
            llvc_root,
            dataset_dir,
            config,
            args.batch_size,
            args.learning_rate,
            args.seed,
        )
        update_state(state_path, dataset=dataset_metadata)
        if args.stage == "prepare":
            return

    if not config.exists():
        raise SystemExit(f"Missing {config}; run --stage prepare first")
    if args.stage in ("all", "train"):
        if not base_checkpoint.exists():
            raise SystemExit(f"Base LLVC checkpoint not found: {base_checkpoint}")
        run(
            [
                sys.executable,
                ROOT / "scripts/finetune-llvc.py",
                "--llvc-root",
                llvc_root,
                "--config",
                config,
                "--base-checkpoint",
                base_checkpoint,
                "--output-checkpoint",
                checkpoint,
                "--steps",
                args.steps,
                "--device",
                args.device,
                "--report",
                training_report,
            ],
        )
        update_state(
            state_path,
            training={
                "steps": args.steps,
                "batchSize": args.batch_size,
                "learningRate": args.learning_rate,
                "device": args.device,
                "actualDevice": json.loads(training_report.read_text())["device"],
                "seed": args.seed,
                "baseCheckpoint": str(base_checkpoint),
                "validation": json.loads(training_report.read_text()).get(
                    "validationMeanAbsoluteError"
                ),
            },
        )
        if args.stage == "train":
            return

    if args.stage in ("all", "export"):
        if not checkpoint.exists():
            raise SystemExit(f"Missing fine-tuned checkpoint: {checkpoint}")
        if base_onnx.exists() and not args.force:
            raise SystemExit(f"Output exists: {base_onnx}; pass --force to replace it")
        run(
            [
                sys.executable,
                ROOT / "scripts/export-llvc.py",
                "--source",
                llvc_root,
                "--config",
                config,
                "--checkpoint",
                checkpoint,
                "--output",
                base_onnx,
            ],
        )
        update_state(state_path, baseArtifact=artifact_info(base_onnx))
        if args.stage == "export":
            return

    if args.stage in ("all", "quantize"):
        if not base_onnx.exists():
            raise SystemExit(f"Missing base ONNX model: {base_onnx}")
        if optimized_onnx.exists() and not args.force:
            raise SystemExit(
                f"Output exists: {optimized_onnx}; pass --force to replace it"
            )
        run(
            [
                sys.executable,
                ROOT / "scripts/quantize-llvc-q8.py",
                "--source",
                base_onnx,
                "--output",
                optimized_onnx,
                "--calibration",
                dataset_dir / "train",
                "--rows-per-clip",
                4,
                "--report",
                q8_report,
            ],
        )

    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    training = state.get(
        "training",
        {
            "steps": args.steps,
            "batchSize": args.batch_size,
            "learningRate": args.learning_rate,
            "device": args.device,
            "seed": args.seed,
            "baseCheckpoint": str(base_checkpoint),
        },
    )
    training["parallelTargetGenerator"] = state.get("dataset", {}).get("generator") or (
        "OpenVoice V2 Q8" if not args.paired_dataset else "user-supplied pairs"
    )

    manifest = {
        **plan,
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "training": training,
        "artifacts": {
            "base": artifact_info(base_onnx),
            "optimizedQ8": artifact_info(optimized_onnx),
        },
        "q8Validation": json.loads(q8_report.read_text())
        if q8_report.exists()
        else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
