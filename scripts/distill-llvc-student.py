#!/usr/bin/env python3
"""Fine-tune a narrow streaming LLVC student against teacher-generated chunks."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


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


def detach_states(values):
    return tuple(value.detach().clone() for value in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("work/LLVC"))
    parser.add_argument("--data", type=Path, default=Path("/tmp/voices-llvc-distill.npz"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/voices-llvc-student128-trained.pth"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    source = args.source.resolve()
    sys.path.insert(0, str(source))
    from model import Net

    config = json.loads((source / "experiments/llvc_nc/config.json").read_text())
    config["model_params"]["enc_dim"] = 128
    config["model_params"]["dec_dim"] = 64
    model = Net(**config["model_params"])
    teacher = torch.load(source / "llvc_models/models/checkpoints/llvc_nc/G_500000.pth", map_location="cpu")["model"]
    state = model.state_dict()
    for name, target in state.items():
        if name in teacher:
            state[name] = sliced_tensor(name, teacher[name], target)
    model.load_state_dict(state)
    if args.student_checkpoint:
        model.load_state_dict(torch.load(args.student_checkpoint, map_location="cpu"))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).train()
    data = np.load(args.data)
    train_names = ["english", "italian", "spanish", "french"]
    steps = min(len(data[f"inputs_{name}"]) for name in train_names)
    inputs = torch.from_numpy(np.stack([data[f"inputs_{name}"][:steps] for name in train_names], axis=1)).unsqueeze(2)
    targets = torch.from_numpy(np.stack([data[f"targets_{name}"][:steps] for name in train_names], axis=1)).unsqueeze(2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    for epoch in range(args.epochs):
        enc, dec, out = model.init_buffers(len(train_names), device)
        conv = torch.zeros(len(train_names), 1, 24, device=device)
        total = 0.0
        for step in range(steps):
            x = inputs[step].to(device)
            target = targets[step].to(device)
            prediction, enc, dec, out, conv = model(x, enc, dec, out, conv, pad=False)
            loss = F.l1_loss(prediction, target) + F.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            enc, dec, out, conv = detach_states((enc, dec, out, conv))
            total += float(loss.detach())
        print(json.dumps({"epoch": epoch + 1, "loss": total / steps, "device": str(device)}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, args.output)


if __name__ == "__main__":
    main()
