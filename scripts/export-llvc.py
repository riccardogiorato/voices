#!/usr/bin/env python3
"""Export the official LLVC streaming checkpoint to a browser-ready ONNX graph."""

import argparse
import json
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("work/LLVC"))
    parser.add_argument("--output", type=Path, default=Path("public/models/common-voice-llvc.onnx"))
    args = parser.parse_args()

    source = args.source.resolve()
    sys.path.insert(0, str(source))
    from model import Net

    config_path = source / "experiments/llvc/config.json"
    checkpoint_path = source / "llvc_models/models/checkpoints/llvc/G_500000.pth"
    config = json.loads(config_path.read_text())
    model = Net(**config["model_params"])
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"])
    model.eval()

    enc, dec, out = model.init_buffers(1, torch.device("cpu"))
    conv = model.convnet_pre.init_ctx_buf(1, torch.device("cpu"))
    chunk_samples = model.dec_chunk_size * model.L
    audio = torch.zeros(1, 1, chunk_samples + model.L * 2)

    class StreamingLLVC(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, audio, enc_state, dec_state, out_state, conv_state):
            return self.inner(audio, enc_state, dec_state, out_state, conv_state, pad=False)

    wrapper = StreamingLLVC(model)
    wrapper.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (audio, enc, dec, out, conv),
        args.output,
        input_names=["audio", "enc_state", "dec_state", "out_state", "conv_state"],
        output_names=["converted", "enc_state_next", "dec_state_next", "out_state_next", "conv_state_next"],
        opset_version=18,
        do_constant_folding=True,
    )
    print(json.dumps({
        "output": str(args.output),
        "sampleRate": config["data"]["sr"],
        "inputSamples": int(audio.shape[-1]),
        "outputSamples": chunk_samples,
        "states": {
            "enc_state": list(enc.shape),
            "dec_state": list(dec.shape),
            "out_state": list(out.shape),
            "conv_state": list(conv.shape),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
