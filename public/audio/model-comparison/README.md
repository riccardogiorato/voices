# Voice-model comparison

All outputs use the same 13.07-second English/Samantha source. MeanVC,
OpenVoice V2 Q8, and FAcodec Q8 each use all three 15-second target references.
The current LLVC Q8 High checkpoint is an any-to-one model, so it can only emit
its trained fixed target without new training.

| Model | Target | Output |
|---|---|---|
| MeanVC | LJ Speech / Linda Johnson | [listen](outputs/meanvc-lj-speech.wav) |
| MeanVC | CMU ARCTIC RMS | [listen](outputs/meanvc-cmu-rms.wav) |
| MeanVC | LLVC derived target | [listen](outputs/meanvc-llvc-target.wav) |
| OpenVoice V2 Q8 | LJ Speech / Linda Johnson | [listen](outputs/openvoice-q8-lj-speech.wav) |
| OpenVoice V2 Q8 | CMU ARCTIC RMS | [listen](outputs/openvoice-q8-cmu-rms.wav) |
| OpenVoice V2 Q8 | LLVC derived target | [listen](outputs/openvoice-q8-llvc-target.wav) |
| FAcodec Q8 | LJ Speech / Linda Johnson | [listen](outputs/facodec-q8-lj-speech.wav) |
| FAcodec Q8 | CMU ARCTIC RMS | [listen](outputs/facodec-q8-cmu-rms.wav) |
| FAcodec Q8 | LLVC derived target | [listen](outputs/facodec-q8-llvc-target.wav) |
| Current LLVC Q8 High | Fixed LLVC target | [listen](outputs/llvc-q8-high-llvc-target.wav) |

The reference clips and their exact provenance are recorded in
[`references/metadata.json`](references/metadata.json). Automatic speaker and
ASR metrics are screening signals; the listening files are the deciding quality
test.

## Reproduce

The conversion harness is Python, not TensorFlow. OpenVoice and FAcodec run as
quantized ONNX through `voiceclonnx`/ONNX Runtime. MeanVC uses its official FP32
PyTorch checkpoint and official inference code. LLVC uses the existing Q8 ONNX
checkpoint through ONNX Runtime.

```sh
uv venv --python 3.11 /tmp/voices-model-bench/venv
uv pip install --python /tmp/voices-model-bench/venv/bin/python \
  -r scripts/requirements-voice-model-benchmark.txt
npm run prepare:voice-targets
HF_HOME=/tmp/voices-model-bench/hf \
  /tmp/voices-model-bench/venv/bin/python scripts/benchmark-voice-models.py
HF_HOME=/tmp/voices-model-bench/hf \
  /tmp/voices-model-bench/venv/bin/python scripts/evaluate-voice-model-quality.py
```

`benchmark-meanvc.py` additionally expects the official MeanVC repository and
checkpoints at `/tmp/voices-model-bench/MeanVC` (or `MEANVC_ROOT`). MeanVC's
speaker verifier currently needs a small compatibility edit that removes unused
training-only imports from that temporary checkout; the report records this
explicitly.
