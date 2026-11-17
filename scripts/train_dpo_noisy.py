"""Hand-written DPO training on pairs with a share of flipped labels (M6 ablation).

Same config files and flags as scripts/train_dpo.py, plus:
    --noise-fraction F    flip chosen and rejected in this share of the TRAINING pairs
    --noise-seed S        seed for which pairs are flipped (default: the config's seed)
    --label-smoothing E   train with the robust DPO loss for an assumed flip rate E
                          (0, the default, means the ordinary DPO loss)

Held-out pairs are never flipped.

Usage:
    python scripts/train_dpo_noisy.py --config configs/dpo_full.yaml \\
        --run-name noise_0.3 --noise-fraction 0.3
    python scripts/train_dpo_noisy.py --config configs/dpo_full.yaml \\
        --run-name noise_0.3_robust --noise-fraction 0.3 --label-smoothing 0.3

Writes the same files as scripts/train_dpo.py under results/<run_name>/. run_info.json
also records the flipped count, a hash of the flipped indices, and the loss used.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

# Lets the GPU allocator return fragmented memory instead of failing; set before torch loads.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from tiny_rlhf.data import load_pairs
from tiny_rlhf.noise import train_with_label_noise
from tiny_rlhf.train_dpo import load_config, load_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--run-name", help="overrides run_name (the results/ subdirectory)")
    parser.add_argument("--max-steps", type=int, help="stop after this many optimizer steps")
    parser.add_argument("--device", help="cpu, cuda, mps or auto")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--noise-fraction", type=float, default=0.0)
    parser.add_argument("--noise-seed", type=int)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    return parser.parse_args()


def collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        overrides[key.strip()] = yaml.safe_load(value)
    if args.run_name is not None:
        overrides["run_name"] = args.run_name
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.device is not None:
        overrides["device"] = args.device
    return overrides


if __name__ == "__main__":
    cli_args = parse_args()
    cfg = load_config(cli_args.config, collect_overrides(cli_args))
    tokenizer, policy, reference = load_models(cfg)
    train_pairs = load_pairs(
        cfg.dataset_name, "train", cfg.n_train, cfg.seed, tokenizer, cfg.max_length
    )
    eval_pairs = load_pairs(
        cfg.dataset_name, "test", cfg.n_eval, cfg.seed, tokenizer, cfg.max_length
    )
    info = train_with_label_noise(
        cfg,
        policy,
        reference,
        train_pairs,
        eval_pairs,
        pad_token_id=tokenizer.pad_token_id,
        out_dir=Path("results") / cfg.run_name,
        noise_fraction=cli_args.noise_fraction,
        noise_seed=cfg.seed if cli_args.noise_seed is None else cli_args.noise_seed,
        label_smoothing=cli_args.label_smoothing,
        checkpoint_dir=Path("checkpoints") / cfg.run_name,
    )
    print(
        f"done: {info['optimizer_steps']} steps in {info['wall_time_s']} s on {info['device']}, "
        f"{info['label_noise']['n_flipped']} of {info['label_noise']['n_train_pairs']} training "
        f"pairs flipped, loss {info['loss_type']}"
    )
