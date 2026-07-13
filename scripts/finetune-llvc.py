#!/usr/bin/env python3
"""Bounded, single-device LLVC fine-tuning from an existing generator checkpoint."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def choose_device(requested):
    if requested == "mps":
        print(
            "MPS warning: LLVC discriminator fallback cannot backpropagate reliably in PyTorch 2.5; using CPU.",
            flush=True,
        )
        return torch.device("cpu")
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(model, optimizer, step, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch": 0,
            "step": step,
        },
        path,
    )


@torch.inference_mode()
def validation_mae(model, dataset, device, max_batches=4):
    model.eval()
    values = []
    for index, (original, target) in enumerate(
        DataLoader(dataset, batch_size=1, shuffle=False)
    ):
        if index >= max_batches:
            break
        output = model(original.to(device))
        target = target.to(device)
        usable = min(output.shape[-1], target.shape[-1])
        values.append(
            torch.nn.functional.l1_loss(
                output[..., :usable], target[..., :usable]
            ).item()
        )
    model.train()
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llvc-root", type=Path, default=Path("work/LLVC"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda")
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")

    llvc_root = args.llvc_root.resolve()
    sys.path.insert(0, str(llvc_root))
    from dataset import LLVCDataset
    from discriminators import (
        MultiPeriodDiscriminator,
        discriminator_loss,
        feature_loss,
        generator_loss,
    )
    from hfg_disc import ComboDisc
    from model import Net
    import utils

    config = json.loads(args.config.read_text())
    seed = int(config.get("seed", 1234))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(args.device)
    print(f"Fine-tuning on {device} for {args.steps} steps", flush=True)

    dataset = LLVCDataset(**config["data"], dset="train")
    if not len(dataset):
        raise SystemExit(f"No training pairs found below {config['data']['dir']}/train")
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    generator = Net(**config["model_params"]).to(device)
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    generator.load_state_dict(base["model"])
    validation_sets = {
        split: LLVCDataset(**config["data"], dset=split) for split in ("val", "dev")
    }
    baseline_validation = {
        split: validation_mae(generator, dataset, device)
        for split, dataset in validation_sets.items()
    }
    if config.get("discriminator") == "hfg":
        discriminator = ComboDisc().to(device)
    else:
        discriminator = MultiPeriodDiscriminator(periods=config["periods"]).to(device)
    optim_g = torch.optim.AdamW(generator.parameters(), **config["optim"])
    optim_d = torch.optim.AdamW(discriminator.parameters(), **config["optim"])
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config["lr_sched"]["lr_decay"]
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config["lr_sched"]["lr_decay"]
    )

    generator.train()
    discriminator.train()
    iterator = iter(loader)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            original, target = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            original, target = next(iterator)
            scheduler_g.step()
            scheduler_d.step()
        original = original.to(device)
        target = target.to(device)
        output = generator(original)
        usable = min(output.shape[-1], target.shape[-1])
        output, target = output[..., :usable], target[..., :usable]
        segment_size = min(int(config["segment_size"]), usable)
        start = random.randint(0, usable - segment_size) if usable > segment_size else 0
        predicted_slice = output[..., start : start + segment_size]
        target_slice = target[..., start : start + segment_size]

        optim_d.zero_grad(set_to_none=True)
        real_scores, fake_scores, _, _ = discriminator(
            target_slice, predicted_slice.detach()
        )
        loss_d, _, _ = discriminator_loss(real_scores, fake_scores)
        loss_d.backward()
        if config.get("grad_clip_threshold") is not None:
            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(), config["grad_clip_threshold"]
            )
        optim_d.step()

        optim_g.zero_grad(set_to_none=True)
        real_scores, fake_scores, real_features, fake_features = discriminator(
            target, output
        )
        loss_adv, _ = generator_loss(fake_scores)
        loss_features = feature_loss(real_features, fake_features) * float(
            config["feature_loss_c"]
        )
        loss_mel = utils.aux_mel_loss(output, target, config) * float(
            config["aux_mel"]["c"]
        )
        loss_g = loss_adv * float(config["disc_loss_c"]) + loss_features + loss_mel
        if not torch.isfinite(loss_g) or not torch.isfinite(loss_d):
            raise RuntimeError(
                f"Non-finite loss at step {step}: generator={loss_g} discriminator={loss_d}"
            )
        loss_g.backward()
        if config.get("grad_clip_threshold") is not None:
            torch.nn.utils.clip_grad_norm_(
                generator.parameters(), config["grad_clip_threshold"]
            )
        optim_g.step()

        if step == 1 or step % args.log_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"step={step}/{args.steps} loss_g={loss_g.item():.4f} "
                f"loss_d={loss_d.item():.4f} steps_per_second={step / elapsed:.3f}",
                flush=True,
            )
        if step % args.save_every == 0 and step != args.steps:
            interim = args.output_checkpoint.with_name(
                f"{args.output_checkpoint.stem}-step-{step}{args.output_checkpoint.suffix}"
            )
            save_checkpoint(generator, optim_g, step, interim)

    save_checkpoint(generator, optim_g, args.steps, args.output_checkpoint)
    finetuned_validation = {
        split: validation_mae(generator, dataset, device)
        for split, dataset in validation_sets.items()
    }
    result = {
        "checkpoint": str(args.output_checkpoint),
        "steps": args.steps,
        "device": str(device),
        "validationMeanAbsoluteError": {
            "base": baseline_validation,
            "fineTuned": finetuned_validation,
        },
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
