# Common Voice Neural model

`common-voice-llvc.onnx` is exported from the official KoeAI LLVC `G_500000.pth` checkpoint using `scripts/export-llvc.py`.

- Source: https://github.com/KoeAI/LLVC
- Weights: https://huggingface.co/KoeAI/llvc_models
- License: MIT
- Input/output sample rate: 16 kHz
- Streaming input: 208 new samples plus 32 samples of preceding context

The ONNX graph and external-data file run locally with ONNX Runtime Web. No microphone audio is uploaded.
