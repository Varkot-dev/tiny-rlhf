"""Paired comparison of two DPO runs on the same held-out preference pairs.

Two runs evaluated on the same pairs can be compared pair by pair, which cancels the
variation between easy and hard pairs that both runs share and gives a narrower, correct
interval than comparing the two accuracies' own intervals. Used for the label-noise sweep:
does the robust loss do better than the ordinary loss on the same 30%-flipped data?

The two runs must have scored the same pairs: their run_info.json files must carry the
same eval_pairs_sha256 fingerprint, and their eval_pairs_final.csv the same pair indices.
A pair counts as ranked right when its final reward margin is above 0 (a tie is wrong,
as in tiny_rlhf.losses.dpo_loss).

Usage:
    python scripts/compare_pairs.py --a results/sweep_noise_0.3 \\
        --b results/sweep_noise_0.3_robust \\
        --out results/sweeps/label_noise/robust_vs_ordinary.json
Writes the JSON file and prints it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tiny_rlhf.stats import DEFAULT_RESAMPLES, paired_bootstrap_difference

ROOT = Path(__file__).resolve().parents[1]


def read_margins(run_dir: Path) -> dict[int, float]:
    with (run_dir / "eval_pairs_final.csv").open() as f:
        return {int(row["pair_index"]): float(row["reward_margin"]) for row in csv.DictReader(f)}


def fingerprint(run_dir: Path) -> str | None:
    return json.loads((run_dir / "run_info.json").read_text()).get("eval_pairs_sha256")


def compare_pairs(
    a: dict[int, float], b: dict[int, float], n_resamples: int = DEFAULT_RESAMPLES, seed: int = 0
) -> dict[str, Any]:
    """Accuracy and margin of b minus a on the same pairs, with paired intervals."""
    if set(a) != set(b):
        raise ValueError("the two runs did not score the same pairs")
    ids = sorted(a)
    right_a = [float(a[i] > 0) for i in ids]
    right_b = [float(b[i] > 0) for i in ids]
    return {
        "n_pairs": len(ids),
        "accuracy_a": sum(right_a) / len(ids),
        "accuracy_b": sum(right_b) / len(ids),
        "accuracy_change_b_minus_a": paired_bootstrap_difference(
            right_a, right_b, n_resamples, seed
        ),
        "margin_change_b_minus_a": paired_bootstrap_difference(
            [a[i] for i in ids], [b[i] for i in ids], n_resamples, seed
        ),
        "both_right": int(sum(x and y for x, y in zip(right_a, right_b, strict=True))),
        "only_a_right": int(sum(x and not y for x, y in zip(right_a, right_b, strict=True))),
        "only_b_right": int(sum(y and not x for x, y in zip(right_a, right_b, strict=True))),
        "both_wrong": int(sum(not x and not y for x, y in zip(right_a, right_b, strict=True))),
        "interval": "paired 95% percentile bootstrap over held-out pairs",
        "n_resamples": n_resamples,
        "seed": seed,
    }


def shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", required=True, help="first run directory")
    parser.add_argument("--b", required=True, help="second run directory (changes are b - a)")
    parser.add_argument("--out", required=True, help="JSON file to write")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_a, run_b = Path(args.a), Path(args.b)
    fa, fb = fingerprint(run_a), fingerprint(run_b)
    if fa is None or fa != fb:
        raise SystemExit(f"the runs' held-out pair fingerprints differ or are missing: {fa} {fb}")
    result = {
        "a": shown(run_a),
        "b": shown(run_b),
        "eval_pairs_sha256": fa,
        **compare_pairs(read_margins(run_a), read_margins(run_b), args.resamples, args.seed),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {shown(out)}")


if __name__ == "__main__":
    main()
