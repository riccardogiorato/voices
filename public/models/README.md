# Common Voice Neural model

`common-voice-llvc.onnx` is exported from the official KoeAI LLVC `G_500000.pth` checkpoint using `scripts/export-llvc.py`.

`common-voice-llvc-student128.onnx` is the production browser model. It uses a
128-channel encoder and 64-channel decoder, initialized by structured slicing
of KoeAI's official LLVC-NC checkpoint and distilled against the full teacher's
streaming outputs. Four bundled languages were used for training and Japanese
was held out for validation. It is optimized for this fixed multilingual demo;
the full 512/256 model remains the general teacher/reference model.

- Source: https://github.com/KoeAI/LLVC
- Weights: https://huggingface.co/KoeAI/llvc_models
- License: MIT
- Input/output sample rate: 16 kHz
- Streaming input: 208 new samples plus 32 samples of preceding context

The ONNX graph and external-data file run locally with ONNX Runtime Web. No microphone audio is uploaded.
