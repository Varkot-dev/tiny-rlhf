"""Run the same DPO config through Hugging Face TRL's DPOTrainer (the M2 parity run).

Usage:
    python scripts/train_dpo.py --config configs/parity_tiny.yaml
    python scripts/train_trl.py --config configs/parity_tiny.yaml
    python scripts/plot_parity.py --ours results/parity_tiny --trl results/parity_tiny_trl

Takes the same flags as scripts/train_dpo.py. Unless --run-name is given, the run is
named after the config's run_name with "_trl" appended, so both runs can sit side by
side under results/.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from typing import Any

import yaml

# Lets the GPU allocator return fragmented memory instead of failing; set before torch loads.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from tiny_rlhf.train_dpo import load_config
from tiny_rlhf.train_trl import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--run-name", help="results/ subdirectory (default: <run_name>_trl)")
    parser.add_argument("--max-steps", type=int, help="stop after this many optimizer steps")
    parser.add_argument("--device", help="cpu, cuda, mps or auto")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config field, e.g. --set beta=0.5 (repeatable)",
    )
    return parser.parse_args()


def collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        overrides[key.strip()] = yaml.safe_load(value)
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.device is not None:
        overrides["device"] = args.device
    return overrides


if __name__ == "__main__":
    cli_args = parse_args()
    cfg = load_config(cli_args.config, collect_overrides(cli_args))
    cfg = dataclasses.replace(cfg, run_name=cli_args.run_name or f"{cfg.run_name}_trl")
    info = main(cfg)
    print(
        f"done (TRL): {info['optimizer_steps']} steps in {info['wall_time_s']} s on "
        f"{info['device']}, first eval loss {info['first_eval_loss']:.6f}"
    )
