# Custom Common Voices

The bundled LLVC model is **any-to-one**: its female target voice is embedded in the trained weights. It cannot switch target voices through an inference parameter.

## Recommended approach: one ONNX model per voice

Train a separate LLVC checkpoint for each consented target voice, then export every checkpoint with the same ONNX inputs, outputs, and recurrent-state shapes.

```text
common-neutral-low.onnx
common-neutral-mid.onnx
common-neutral-high.onnx
common-synthetic.onnx
```

Because each model uses the same architecture, the browser Worker, AudioWorklet, buffering logic, and ONNX Runtime integration can remain unchanged. The application only needs to select a different model URL.

## Training data

LLVC expects paired audio files:

```text
001_original.wav
001_converted.wav
002_original.wav
002_converted.wav
```

- `original` contains speech from many different source speakers.
- `converted` contains the same utterance rendered as the desired target voice.

The official LLVC workflow creates this parallel dataset by converting a large multi-speaker corpus into one target voice. LLVC is then trained to reproduce that conversion with a small, streaming model.

Use only target recordings and generated voices for which you have appropriate consent and usage rights.

Official training documentation: <https://github.com/KoeAI/LLVC#dataset>

## Exporting a trained checkpoint

After training, export the checkpoint to ONNX:

```bash
python scripts/export-llvc.py \
  --checkpoint experiments/neutral-low/G_500000.pth \
  --output public/models/common-neutral-low.onnx
```

The current `scripts/export-llvc.py` assumes the bundled official checkpoint. It must be extended with a `--checkpoint` argument before using the command above.

Keep the original LLVC configuration unchanged if the exported models must remain drop-in compatible with the current browser runtime. In particular, preserve:

- 16 kHz input and output
- 240-sample model input
- 208-sample streaming output
- Encoder, decoder, output, and convolution state shapes

Each exported model may also produce an external `.onnx.data` file. Both files must be deployed together.

## Multiple voices in one ONNX model

A single model containing several selectable voices requires an **any-to-many** architecture conditioned on a speaker ID or speaker embedding:

```text
audio + speaker_id + recurrent_states
  → converted_audio + next_recurrent_states
```

This requires changing LLVC's label-conditioning logic, preparing data for every target voice, retraining the model, and updating the browser inference interface. It cannot be achieved by editing or blending the existing female checkpoint.

## Suggested first voice set

For the shared-voice use case, begin with three clearly synthetic, consented targets:

1. Neutral low
2. Neutral mid
3. Neutral high

Separate same-shape ONNX files are currently the simplest, safest, and most reliable implementation.
