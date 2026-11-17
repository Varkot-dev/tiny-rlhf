"""Train the explicit reward model (Bradley-Terry loss) from a config file.

Usage:
    python scripts/train_reward.py --config configs/reward_tiny.yaml
    python scripts/train_reward.py --config configs/reward_full.yaml --set lr=1e-4

Writes results/<run_name>/{config.yaml,train.csv,eval.csv,eval_pairs_final.csv,run_info.json}.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import yaml

# Lets the GPU allocator return fragmented memory instead of failing; set before torch loads.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from tiny_rlhf.reward_model import load_reward_config, main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--run-name", help="overrides run_name (the results/ subdirectory)")
    parser.add_argument("--max-steps", type=int, help="stop after this many optimizer steps")
    parser.add_argument("--device", help="cpu, cuda, mps or auto")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config field, e.g. --set head_lr=1e-3 (repeatable)",
    )
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
    info = main(load_reward_config(cli_args.config, collect_overrides(cli_args)))
    low, high = info["heldout_accuracy_ci95"]
    print(
        f"done: {info['optimizer_steps']} steps in {info['wall_time_s']} s on {info['device']}, "
        f"first eval loss {info['first_eval_loss']:.6f}, held-out accuracy "
        f"{info['heldout_accuracy']:.3f} (95% bootstrap interval {low:.3f} to {high:.3f}, "
        f"{info['n_eval_pairs']} pairs)"
    )
