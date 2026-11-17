"""Train on the Track B dataset with the hand-written KTO loop (M7).

Usage:
    python scripts/train_kto.py --config configs/trackb_kto_tiny.yaml
    python scripts/train_kto.py --config configs/trackb_kto.yaml --set beta=0.5 \\
        --run-name trackb_kto_beta0.5

Same flags as scripts/train_dpo.py and scripts/train_trackb.py. Every Track B pair is
split into one good and one bad answer (see tiny_rlhf.kto). Writes
results/<run_name>/{config.yaml,train.csv,eval.csv,eval_pairs_final.csv,run_info.json}
and saves the LoRA adapter to checkpoints/<run_name>/.
"""

from __future__ import annotations

import argparse
from typing import Any

import yaml

from tiny_rlhf.kto import load_kto_config, main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--run-name", help="overrides run_name (the results/ subdirectory)")
    parser.add_argument("--max-steps", type=int, help="stop after this many optimizer steps")
    parser.add_argument("--device", help="cpu, cuda, mps or auto")
    parser.add_argument("--results-root", default="results", help="where run directories go")
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
        load_kto_config(cli_args.config, collect_overrides(cli_args)),
        results_root=cli_args.results_root,
    )
    print(
        f"done: {info['optimizer_steps']} steps in {info['wall_time_s']} s on {info['device']}, "
        f"first eval loss {info['first_eval_loss']:.6f} "
        f"(expected {info['expected_first_eval_loss']:.6f})"
    )
