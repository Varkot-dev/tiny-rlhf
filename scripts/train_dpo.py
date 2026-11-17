"""Run hand-written DPO training from a config file.

Usage:
    python scripts/train_dpo.py --config configs/dpo_tiny.yaml
    python scripts/train_dpo.py --config configs/dpo_full.yaml --set beta=0.5 --run-name dpo_beta0.5

Writes results/<run_name>/{config.yaml,train.csv,eval.csv,eval_pairs_final.csv,run_info.json}
and, when save_adapter is true, the LoRA adapter to checkpoints/<run_name>/.
"""

from __future__ import annotations

import argparse
from typing import Any

import yaml

from tiny_rlhf.train_dpo import load_config, main


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
        help="override any config field, e.g. --set beta=0.5 (repeatable)",
    )
    return parser.parse_args()


def collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        overrides[key.strip()] = yaml.safe_load(value)  # "0.5" -> 0.5, "true" -> True
    if args.run_name is not None:
        overrides["run_name"] = args.run_name
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.device is not None:
        overrides["device"] = args.device
    return overrides


if __name__ == "__main__":
    cli_args = parse_args()
    info = main(load_config(cli_args.config, collect_overrides(cli_args)))
    print(
        f"done: {info['optimizer_steps']} steps in {info['wall_time_s']} s on {info['device']}, "
        f"first eval loss {info['first_eval_loss']:.6f}"
    )
