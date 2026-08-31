#!/usr/bin/env python
"""Measure the cost side of the depth frontier: parameters and throughput per tap.

`analyze_depth.py` says how well each depth scores. This says what each depth
costs, so the two together are a frontier rather than a ranking. Truncating a
tower at layer k is not a simulation here -- the blocks above k are deleted and
the remaining model is actually timed on the GPU, because an analytic FLOP count
would not notice that attention at 196 tokens is memory-bound.

    python scripts/bench_depth.py --out runs/depth_cost.md

Run it on the same node class as the extraction, and say which one in the
write-up: throughput is a property of the pair, not of the model.

Parameters are counted for what a deployed detector at that depth would actually
carry -- patch embedding, position embedding, k transformer blocks, the final
layer norm -- and exclude the attention-pooling head, which only exists at full
depth. The competition's <2B budget is a parameter budget, so this column is the
one that speaks to it; the throughput column is the one a platform feels.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from byteprint_depth import tap_layers

REPO = "google/siglip2-so400m-patch16-naflex"
PATCH = 16


def build_tower():
    from transformers import AutoModel

    model = AutoModel.from_pretrained(REPO, local_files_only=True)
    return model.vision_model


def truncate(tower, depth: int):
    """Keep the first `depth` transformer blocks. Destructive, so go downwards."""
    tower.encoder.layers = tower.encoder.layers[:depth]
    return tower


def carried_parameters(tower, depth: int) -> int:
    """What a detector truncated at `depth` would have to load.

    The encoder is sliced already, so `tower.embeddings` plus the surviving
    blocks is the honest count. `head` -- the attention pooler -- is excluded
    because a truncated model mean-pools instead and never instantiates it.
    """
    counted = 0
    for name, module in tower.named_children():
        if name == "head":
            continue
        counted += sum(p.numel() for p in module.parameters())
    return counted


def patches_for(batch: int, crop: int, device: str) -> dict:
    rows = columns = crop // PATCH
    generator = torch.Generator().manual_seed(0)
    pixels = torch.rand(batch, 3, crop, crop, generator=generator)
    flattened = (
        pixels.reshape(batch, 3, rows, PATCH, columns, PATCH)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(batch, rows * columns, PATCH * PATCH * 3)
    )
    return {
        "pixel_values": flattened.to(device),
        "pixel_attention_mask": torch.ones(batch, rows * columns, dtype=torch.long, device=device),
        "spatial_shapes": torch.tensor([[rows, columns]] * batch, dtype=torch.long, device=device),
    }


def throughput(tower, inputs: dict, *, warmup: int, iters: int, device: str) -> float:
    """Crops per second, timed after warmup with the device synchronised."""
    with torch.inference_mode():
        for _ in range(warmup):
            tower(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            tower(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return inputs["pixel_values"].shape[0] * iters / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tower = build_tower().to(args.device).eval()
    depth = int(tower.config.num_hidden_layers)
    taps = tap_layers(depth)
    full_params = carried_parameters(tower, depth)
    inputs = patches_for(args.batch_size, args.crop_size, args.device)

    print(f"{REPO} on {args.device}: {depth} layers, {full_params / 1e6:.0f}M carried")
    print(f"batch {args.batch_size} x {args.crop_size}px, {args.iters} timed iterations\n")

    rows = []
    # Descending, because truncation deletes blocks in place.
    for tap in sorted(taps, reverse=True):
        truncate(tower, tap)
        params = carried_parameters(tower, tap)
        rate = throughput(tower, inputs, warmup=args.warmup, iters=args.iters, device=args.device)
        rows.append((tap, params, rate))
        print(f"  layer {tap:>3}  {params / 1e6:>7.0f}M  {rate:>8.1f} crops/s", flush=True)

    rows.reverse()
    # `tap_layers` always includes the final layer, so this is the full model.
    full_rate = next(rate for tap, _, rate in rows if tap == depth)

    lines = [
        "| tap | carried params | share of full | crops/s | speed-up |",
        "|---|---|---|---|---|",
    ]
    for tap, params, rate in rows:
        lines.append(
            f"| layer {tap} | {params / 1e6:.0f}M | {params / full_params:.2f}x | "
            f"{rate:.1f} | {rate / full_rate:.2f}x |"
        )
    body = (
        f"## The depth frontier — cost by tap\n\n"
        f"`{REPO}`, {args.crop_size}px crops, batch {args.batch_size}, "
        f"{args.device}. Attention-pooling head excluded; a truncated tower mean-pools.\n\n"
        + "\n".join(lines)
        + "\n"
    )
    print("\n" + body)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body)
        print(f"written: {args.out}")


if __name__ == "__main__":
    main()
