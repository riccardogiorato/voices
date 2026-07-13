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
    parser.add_argument("--experiment", default="llvc")
    parser.add_argument(
        "--config",
        type=Path,
        help="Explicit config.json (overrides --experiment config)",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("public/models/common-voice-llvc.onnx")
    )
    args = parser.parse_args()

    source = args.source.resolve()
    sys.path.insert(0, str(source))
    from model import Net

    config_path = (
        args.config.resolve()
        if args.config
        else source / f"experiments/{args.experiment}/config.json"
    )
    checkpoint_path = (
        args.checkpoint
        or source / f"llvc_models/models/checkpoints/{args.experiment}/G_500000.pth"
    )
    config = json.loads(config_path.read_text())
    model = Net(**config["model_params"])
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]
    )
    model.eval()

    enc, dec, out = model.init_buffers(1, torch.device("cpu"))
    # LLVC-NC has no convolutional prenet. Preserve the standard 24-sample
    # state shape as an identity passthrough so exported variants remain
    # drop-in compatible with the browser worker.
    conv = (
        model.convnet_pre.init_ctx_buf(1, torch.device("cpu"))
        if hasattr(model, "convnet_pre")
        else torch.zeros(1, 1, 24)
    )
    chunk_samples = model.dec_chunk_size * model.L
    audio = torch.zeros(1, 1, chunk_samples + model.L * 2)

    class StreamingLLVC(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, audio, enc_state, dec_state, out_state, conv_state):
            return self.inner(
                audio, enc_state, dec_state, out_state, conv_state, pad=False
            )

    wrapper = StreamingLLVC(model)
    wrapper.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # PyTorch's external-data writer appends to an existing sidecar. Remove a
    # previous export so repeated --force runs cannot silently grow the model.
    args.output.unlink(missing_ok=True)
    args.output.with_name(args.output.name + ".data").unlink(missing_ok=True)
    torch.onnx.export(
        wrapper,
        (audio, enc, dec, out, conv),
        args.output,
        input_names=["audio", "enc_state", "dec_state", "out_state", "conv_state"],
        output_names=[
            "converted",
            "enc_state_next",
            "dec_state_next",
            "out_state_next",
            "conv_state_next",
        ],
        opset_version=18,
        do_constant_folding=True,
    )

    # Graph-level simplification (onnx-simplifier): constant folding, dead-code
    # elimination, op fusion. Verified bit-identical on this model (check_n=3 +
    # ORT cross-check, max_abs_diff = 0.0). Removes the always-zero label-embedding
    # MLP, the over-sized [1,200,256] positional-encoding buffer (only 26 positions
    # are used), 22 Transposes and all 18 Pads — ~11% fewer nodes, ~14% smaller.
    # See PROGRESS.md and the LLVC paper research (arXiv:2311.00873).
    try:
        import os
        import onnx
        from onnxsim import simplify

        raw = onnx.load(str(args.output), load_external_data=True)
        simplified, ok = simplify(
            raw,
            overwrite_input_shapes={
                "audio": [1, 1, int(audio.shape[-1])],
                "enc_state": list(enc.shape),
                "dec_state": list(dec.shape),
                "out_state": list(out.shape),
                "conv_state": list(conv.shape),
            },
            perform_optimization=True,
            check_n=3,
        )
        tmp = args.output.with_suffix(".simp.onnx")
        tmp_data = args.output.with_suffix(".simp.onnx.data")
        data_target = args.output.with_name(args.output.name + ".data")
        tmp.unlink(missing_ok=True)
        tmp_data.unlink(missing_ok=True)
        onnx.save_model(
            simplified,
            str(tmp),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=tmp.name + ".data",
            size_threshold=1024,
        )
        raw_data = args.output.with_name(args.output.name + ".data")
        raw_bytes = args.output.stat().st_size + (
            raw_data.stat().st_size if raw_data.exists() else 0
        )
        simplified_bytes = tmp.stat().st_size + tmp_data.stat().st_size
        if simplified_bytes >= raw_bytes:
            tmp.unlink()
            tmp_data.unlink()
            print(
                f"onnx-simplifier: rejected size regression {raw_bytes} -> {simplified_bytes} bytes "
                f"(nodes {len(raw.graph.node)} -> {len(simplified.graph.node)})"
            )
        else:
            # The graph initially points at the temporary external-data filename.
            # Rewrite that metadata before both files receive their final names.
            external_model = onnx.load(str(tmp), load_external_data=False)
            for initializer in external_model.graph.initializer:
                for entry in initializer.external_data:
                    if entry.key == "location":
                        entry.value = data_target.name
            onnx.save_model(external_model, str(tmp))
            os.replace(tmp, args.output)
            os.replace(tmp_data, data_target)
            print(
                f"onnx-simplifier: ok={ok} nodes {len(raw.graph.node)} -> {len(simplified.graph.node)}, "
                f"bytes {raw_bytes} -> {simplified_bytes}"
            )
    except ImportError:
        print("onnx-simplifier: onnxsim not installed; skipping (pip install onnxsim)")
    except Exception as exc:  # noqa: BLE001 - never corrupt a successful raw export
        print(f"onnx-simplifier: skipped ({exc}); keeping raw export")

    print(
        json.dumps(
            {
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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
