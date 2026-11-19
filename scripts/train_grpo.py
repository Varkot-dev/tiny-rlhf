"""Train on Track B prompts with GRPO and the checker reward (M8).

Usage:
    python scripts/train_grpo.py --config configs/trackb_grpo_tiny.yaml
    python scripts/train_grpo.py --config configs/trackb_grpo.yaml
    python scripts/train_grpo.py --config configs/trackb_grpo.yaml --set beta=0.0 \\
        --run-name trackb_grpo_beta0

TRL's GRPOTrainer does the sampling and the update; the reward is
tiny_rlhf.grpo_reward (strict automatic checks). Writes
results/<run_name>/{config.yaml,train.csv,samples.jsonl,run_info.json} and saves the LoRA
adapter to checkpoints/<run_name>/, where scripts/eval_trackb.py reads it.
"""

from __future__ import annotations

import argparse
from typing import Any

import yaml

from tiny_rlhf.train_grpo import load_grpo_config, main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--run-name", help="overrides run_name (the results/ subdirectory)")
    parser.add_argument("--max-steps", type=int, help="stop after this many optimizer steps")
    parser.add_argument("--device", help="cpu, cuda or auto")
    parser.add_argument("--results-root", default="results", help="where run directories go")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config field, e.g. --set beta=0.0 (repeatable)",
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
    info = main(
        load_grpo_config(cli_args.config, collect_overrides(cli_args)),
        results_root=cli_args.results_root,
    )
    last = info["final_logged"] or {}
    print(
        f"done: {info['optimizer_steps']} steps in {info['wall_time_s']} s on {info['device']} "
        f"({info['seconds_per_step']} s per step), last reward mean {last.get('reward_mean')}, "
        f"peak GPU memory {info['peak_gpu_memory_mib']} MiB"
    )
