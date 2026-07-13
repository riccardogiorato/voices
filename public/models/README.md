# Common Voice Neural model

`common-voice-llvc.onnx` is exported from the official KoeAI LLVC `G_500000.pth` checkpoint using `scripts/export-llvc.py`.

`common-voice-llvc-q8-high.onnx` is the default browser model. It keeps the
teacher's complete 512/256 architecture and learned weights, but applies static
8-bit post-training quantization to the eight dominant 512×512 pointwise
convolutions. `common-voice-llvc-q8-max.onnx` quantizes every supported Conv,
MatMul, and Gemm operation for a smaller, faster, experimental alternative.
Q8 Max can degrade sharply on inputs outside its calibration distribution (an
unseen Italian/Flo test measured 0.818 waveform correlation); it is exposed for
comparison, not recommended as the production-quality default.
Neither variant was trained or fine-tuned. The original FP32 graph remains the
reference and all three are selectable in the web app.

- Source: https://github.com/KoeAI/LLVC
- Weights: https://huggingface.co/KoeAI/llvc_models
- License: MIT
- Input/output sample rate: 16 kHz
- Streaming input: 208 new samples plus 32 samples of preceding context

The ONNX graphs run locally with ONNX Runtime Web. No microphone audio is uploaded.

## Custom target voices

`scripts/customize-llvc.py` builds one same-shape LLVC model per consented
speaker. It generates or accepts aligned parallel pairs, fine-tunes the official
LLVC-NC generator, exports FP32 ONNX, and creates the conservative Q8-High
deployment variant:

```text
custom/llvc-base-<voice>.onnx
custom/llvc-optimized-q8-<voice>.onnx
custom/llvc-<voice>.json
```

See [`CUSTOM-VOICE.md`](../../CUSTOM-VOICE.md) for setup, data requirements,
staged execution, hardware limitations, and the complete command.

Generate and evaluate all four experimental precision variants with:

```bash
bun run quantize:variants
bun run evaluate:variants
bun run benchmark:variants
```
