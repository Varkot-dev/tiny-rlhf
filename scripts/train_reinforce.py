"""Run the toy RL experiment: REINFORCE with three baselines, and DPO, on a task whose
exact optimum is known by enumeration.

Usage:
    python scripts/train_reinforce.py --config configs/reinforce_toy.yaml

Writes results/<run_name>/:
    config.yaml          the resolved config
    curves.csv           exact metrics over training, one row per (method, seed, step)
    extended_curves.csv  one longer DPO run per label mode
    grad_variance.json   spread of the REINFORCE gradient estimate per baseline
    summary.json         final-step metrics per method (mean, min, max over seeds) and
                         the exact optimum
    run_info.json        wall time, seeds, library versions, git commit
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from pathlib import Path

import torch
import yaml

from tiny_rlhf.reinforce import load_toy_config, run_experiment
from tiny_rlhf.train_dpo import git_sha, library_versions

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = (
    "method",
    "seed",
    "step",
    "sequences_seen",
    "expected_reward",
    "kl_from_ref",
    "objective",
    "kl_to_opt",
    "tv_to_opt",
)


def write_curves(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--run-name", help="overrides run_name (the results/ subdirectory)")
    args = parser.parse_args()

    torch.set_num_threads(1)  # the toy is tiny; one core is enough and keeps timing honest
    overrides = {"run_name": args.run_name} if args.run_name else None
    cfg = load_toy_config(args.config, overrides)
    out_dir = ROOT / "results" / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))

    start = time.perf_counter()
    out = run_experiment(cfg)
    wall_time = time.perf_counter() - start

    write_curves(out_dir / "curves.csv", out["curves"])
    write_curves(out_dir / "extended_curves.csv", out["extended_curves"])
    (out_dir / "grad_variance.json").write_text(json.dumps(out["grad_variance"], indent=2) + "\n")
    extended_final = {
        method: next(row for row in reversed(out["extended_curves"]) if row["method"] == method)
        for method in dict.fromkeys(row["method"] for row in out["extended_curves"])
    }
    summary = {"task": out["task"], "final": out["summary"], "extended_final": extended_final}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    run_info = {
        "run_name": cfg.run_name,
        "wall_time_s": round(wall_time, 2),
        "torch_threads": 1,
        "device": "cpu",
        "init_seed": cfg.init_seed,
        "sampling_seeds": cfg.seeds,
        "git_sha": git_sha(),
        "versions": library_versions(),
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")

    task = out["task"]
    print(
        f"optimum: expected reward {task['optimal_expected_reward']:.4f}, "
        f"KL from reference {task['optimal_kl_from_ref']:.4f}"
    )
    for method, stats in out["summary"].items():
        print(
            f"{method:28s} reward {stats['expected_reward']['mean']:.4f}  "
            f"KL from ref {stats['kl_from_ref']['mean']:.4f}  "
            f"KL to optimum {stats['kl_to_opt']['mean']:.4f}"
        )
    print(f"done in {wall_time:.1f} s, wrote {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
