# Custom LLVC voice pipeline

The automated pipeline creates one browser-compatible, any-to-one LLVC model
for a consented target voice and then produces a calibrated Q8 version.

```text
target speaker audio
  → OpenVoice Q8 parallel-pair generation
  → bounded LLVC fine-tune from the official LLVC-NC checkpoint
  → FP32 ONNX export
  → static Q8-High calibration and validation
```

The final names are deterministic:

```text
public/models/custom/llvc-base-<voice-name>.onnx
public/models/custom/llvc-base-<voice-name>.onnx.data
public/models/custom/llvc-optimized-q8-<voice-name>.onnx
public/models/custom/llvc-<voice-name>.json
```

`base` means the fine-tuned FP32 ONNX model. `optimized-q8` is its deployment
variant. The JSON manifest records the inputs, training settings, byte sizes,
and SHA-256 hashes.

## Why a single WAV is not trained directly

LLVC requires aligned pairs containing identical words:

```text
000001_original.wav
000001_converted.wav
```

The original side contains many source speakers. The converted side contains
the same utterance in one target voice. Given only a custom target recording,
the pipeline uses OpenVoice V2 Q8 to render this parallel side automatically.
This is synthetic supervision, so OpenVoice artifacts can be learned by LLVC;
inspect the generated pairs before a long run.

## Setup

Python 3.11 and `ffmpeg` are required.

```bash
uv venv --python 3.11 work/custom-llvc-env
uv pip install --python work/custom-llvc-env/bin/python \
  -r scripts/requirements-custom-llvc.txt
```

The official LLVC source and base checkpoints must exist under `work/LLVC`.
This workspace already uses that checkout. A fresh checkout can run
`python download_models.py` from inside `work/LLVC`.

## Complete run

```bash
work/custom-llvc-env/bin/python scripts/customize-llvc.py \
  --voice-name "Alice Studio" \
  --target-audio /absolute/path/alice.wav \
  --confirm-rights \
  --device cpu \
  --steps 5000
```

Audio files or directories may be repeated:

```bash
  --target-audio /path/session-1.wav \
  --target-audio /path/session-2.flac \
  --source-audio /path/multispeaker-corpus-a \
  --source-audio /path/multispeaker-corpus-b
```

Without `--source-audio`, the pipeline uses the repository's unseen short and
long English/French/Italian/Spanish clips. This is enough for a functional
experiment, not a production voice. A diverse, licensed multi-speaker corpus
is strongly recommended.

The target reference is normalized to 16 kHz mono PCM and capped at 30 seconds.
Source files are converted to the target voice, resampled to 16 kHz, split by
source file into train/validation/development sets, and chunked into aligned
4.096-second LLVC pairs.

## Existing paired dataset

To avoid OpenVoice generation, supply a prepared LLVC dataset:

```bash
work/custom-llvc-env/bin/python scripts/customize-llvc.py \
  --voice-name "Alice Studio" \
  --target-audio /path/alice.wav \
  --paired-dataset /path/alice-pairs \
  --confirm-rights
```

The directory must contain `train`, `val`, and `dev`, each with corresponding
`*_original.wav` and `*_converted.wav` files at 16 kHz.

## Stages and resuming

Every stage can be run separately:

```bash
# Inspect paths without downloads or writes
python3 scripts/customize-llvc.py --voice-name "Alice Studio" --dry-run

# Prepare pairs only
work/custom-llvc-env/bin/python scripts/customize-llvc.py \
  --voice-name "Alice Studio" --target-audio /path/alice.wav \
  --confirm-rights --stage prepare

# Subsequent stages do not require the reference again
work/custom-llvc-env/bin/python scripts/customize-llvc.py \
  --voice-name "Alice Studio" --confirm-rights --stage train --steps 5000
work/custom-llvc-env/bin/python scripts/customize-llvc.py \
  --voice-name "Alice Studio" --stage export
work/custom-llvc-env/bin/python scripts/customize-llvc.py \
  --voice-name "Alice Studio" --stage quantize
```

Use `--force` only when intentionally replacing an existing generated dataset
or ONNX model. Intermediate checkpoints are saved every 500 steps under
`work/custom-voices/<voice-name>/checkpoints`.

## Hardware and quality expectations

- `--device auto` uses CUDA when available and otherwise CPU.
- PyTorch 2.5 cannot backpropagate the LLVC discriminator's MPS/CPU fallback
  reliably. An explicit `--device mps` request warns and falls back to CPU.
- CPU training is supported but can take a long time.
- Fine-tuning starts from the official LLVC-NC generator; it is not training the
  architecture from scratch.
- The generator starts from the official weights; the adversarial discriminator
  is initialized fresh for the custom parallel dataset.
- The content/fairseq loss is disabled to avoid an additional large HuBERT
  dependency. Adversarial, feature-matching, and multi-resolution mel losses
  remain enabled.
- Q8 uses the project's conservative Q8-High policy: only the eight dominant
  512×512 pointwise convolutions are quantized.
- The quantizer runs a finite-output test and reports FP32/Q8 correlation and
  mean absolute error on calibration states.
- Training records before/after validation and development waveform MAE in the
  manifest so obvious divergence is visible before deployment.

Use only voices and recordings for which you have explicit cloning and
deployment rights. `--confirm-rights` is required before dataset generation.
