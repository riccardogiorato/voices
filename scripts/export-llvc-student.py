#!/usr/bin/env python3
"""Export a narrower LLVC student initialized by structured teacher slicing."""

import argparse
import json
import sys
from pathlib import Path

import torch


def sliced_tensor(name, source, target):
    if source.ndim != target.ndim or any(a < b for a, b in zip(source.shape, target.shape)):
        return target
    if name.endswith("in_proj_weight") and source.shape[0] == 3 * source.shape[1] and target.shape[0] == 3 * target.shape[1]:
        old_dim, new_dim = source.shape[1], target.shape[1]
        return torch.cat([source[i * old_dim:i * old_dim + new_dim, :new_dim] for i in range(3)])
    if name.endswith("in_proj_bias") and source.shape[0] % 3 == 0 and target.shape[0] % 3 == 0:
        old_dim, new_dim = source.shape[0] // 3, target.shape[0] // 3
        return torch.cat([source[i * old_dim:i * old_dim + new_dim] for i in range(3)])
    return source[tuple(slice(0, size) for size in target.shape)].clone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("work/LLVC"))
    parser.add_argument("--enc-dim", type=int, required=True)
    parser.add_argument("--dec-dim", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    sys.path.insert(0, str(source))
    from model import Net

    config = json.loads((source / "experiments/llvc_nc/config.json").read_text())
    params = config["model_params"]
    params["enc_dim"] = args.enc_dim
    params["dec_dim"] = args.dec_dim
    model = Net(**params)
    teacher = torch.load(source / "llvc_models/models/checkpoints/llvc_nc/G_500000.pth", map_location="cpu")["model"]
    student = model.state_dict()
    for name, target in student.items():
        if name in teacher:
            student[name] = sliced_tensor(name, teacher[name], target)
    model.load_state_dict(student)
    if args.student_checkpoint:
        model.load_state_dict(torch.load(args.student_checkpoint, map_location="cpu"))
    model.eval()

    enc, dec, out = model.init_buffers(1, torch.device("cpu"))
    conv = torch.zeros(1, 1, 24)
    chunk_samples = model.dec_chunk_size * model.L
    audio = torch.zeros(1, 1, chunk_samples + model.L * 2)

    class StreamingStudent(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, audio, enc_state, dec_state, out_state, conv_state):
            return self.inner(audio, enc_state, dec_state, out_state, conv_state, pad=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        StreamingStudent(model).eval(),
        (audio, enc, dec, out, conv),
        args.output,
        input_names=["audio", "enc_state", "dec_state", "out_state", "conv_state"],
        output_names=["converted", "enc_state_next", "dec_state_next", "out_state_next", "conv_state_next"],
        opset_version=18,
        do_constant_folding=True,
    )
    print(json.dumps({"output": str(args.output), "enc_dim": args.enc_dim, "dec_dim": args.dec_dim, "states": [list(enc.shape), list(dec.shape), list(out.shape)]}))


if __name__ == "__main__":
    main()
