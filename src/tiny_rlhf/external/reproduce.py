"""Does a retrained adapter stand in for the saved run it was retrained from?

LoRA adapters are not kept in the repository (checkpoints/ is ignored), so evaluating a
trained Track B model on the external set means training it again from its config and
seed. GPU training is not bit-for-bit deterministic, so the new adapter is not the saved
one. Before external results are reported for a run, this check compares the new
adapter's Track B evaluation with the saved one on the same examples:

    config      the two runs' config.yaml files must agree on every key both contain,
                apart from run_name. A key only one contains is listed but does not
                block: the trainer writes settings added after a run was made (for
                example sft_weight, default 0.0, is missing from trackb_full's file)
    final eval  the last row of each run's eval.csv (held-out loss, accuracy, margin),
                reported side by side
    responses   share of Track B examples where the two adapters wrote identical text
    rates       pooled over the five test splits (750 examples at --limit 150): the paired
                change, rerun minus saved, in followed_injection (injected), on_task
                (injected), task_correct and task_correct_strict (no injection), with a
                paired bootstrap interval. task_correct_strict, the rule the README
                reports, is recomputed from each saved response with the current rule on
                both sides: older evaluations stored none, or one from the first strict
                rule (fixed since; see the decision log)

The rerun reproduces the saved run when the configs match and, for each of the three
rates, the interval contains 0 and the change is at most TOLERANCE in size. If it does
not, the external results describe the rerun adapter, and are reported next to the
rerun's own Track B numbers, not the saved run's.

Usage:
    python -m tiny_rlhf.external.reproduce --saved results/trackb_full \\
        --rerun results/trackb_full_m9
Writes <rerun>/reproduction.json and prints the verdict.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tiny_rlhf.stats import DEFAULT_RESAMPLES, paired_bootstrap_difference
from tiny_rlhf.trackb.checks import task_correct_strict
from tiny_rlhf.trackb.evaluate import DEFAULT_SPLITS, group_rows, read_rows

TOLERANCE = 0.03
CHECKED = (
    ("injected", "followed_injection"),
    ("injected", "on_task"),
    ("no_injection", "task_correct"),
    ("no_injection", "task_correct_strict"),
)
# Recomputed from the response rather than read from the saved scores (see above).
RECOMPUTED = {"task_correct_strict": task_correct_strict}


def metric_value(row: dict[str, Any], metric: str) -> float:
    if metric in RECOMPUTED:
        return float(RECOMPUTED[metric](row["response"], row))
    return float(row["scores"][metric])


def config_differences(saved: dict[str, Any], rerun: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(differing values of shared keys, keys only one config has), ignoring run_name."""
    shared = sorted((set(saved) & set(rerun)) - {"run_name"})
    differing = [
        f"{k}: saved={saved[k]!r} rerun={rerun[k]!r}" for k in shared if saved[k] != rerun[k]
    ]
    one_sided = [
        f"{k}: saved={saved.get(k, '(absent)')!r} rerun={rerun.get(k, '(absent)')!r}"
        for k in sorted(set(saved) ^ set(rerun))
    ]
    return differing, one_sided


def last_eval_row(run_dir: Path) -> dict[str, float] | None:
    path = run_dir / "eval.csv"
    if not path.exists():
        return None
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {k: float(v) for k, v in rows[-1].items()} if rows else None


def pooled_rows(summary_path: Path, splits: tuple[str, ...]) -> list[dict[str, Any]]:
    summary = json.loads(summary_path.read_text())
    rows: list[dict[str, Any]] = []
    for split in splits:
        if split not in summary["files"]:
            raise ValueError(f"{summary_path} has no {split} evaluation")
        rows += read_rows(summary_path.parent / summary["files"][split])
    return rows


def compare_track_b(
    saved_rows: list[dict[str, Any]],
    rerun_rows: list[dict[str, Any]],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    tolerance: float = TOLERANCE,
) -> dict[str, Any]:
    """Identical-response share and the paired change in the CHECKED rates."""
    saved_by_id = {r["id"]: r for r in saved_rows}
    rerun_by_id = {r["id"]: r for r in rerun_rows}
    if set(saved_by_id) != set(rerun_by_id):
        raise ValueError("the two evaluations did not score the same examples")
    ids = sorted(saved_by_id)
    saved = [saved_by_id[i] for i in ids]
    rerun = [rerun_by_id[i] for i in ids]
    if any(s["prompt"] != r["prompt"] for s, r in zip(saved, rerun, strict=True)):
        raise ValueError("the two evaluations used different prompts for the same ids")
    identical = sum(s["response"] == r["response"] for s, r in zip(saved, rerun, strict=True))
    saved_groups, rerun_groups = group_rows(saved), group_rows(rerun)
    rates = {}
    for group, metric in CHECKED:
        before = [metric_value(r, metric) for r in saved_groups[group]]
        after = [metric_value(r, metric) for r in rerun_groups[group]]
        change = paired_bootstrap_difference(before, after, n_resamples, seed)
        within = (
            change["difference"] is not None
            and abs(change["difference"]) <= tolerance
            and change["ci_low"] <= 0 <= change["ci_high"]
        )
        rates[f"{group}/{metric}"] = {
            "saved": sum(before) / len(before) if before else None,
            "rerun": sum(after) / len(after) if after else None,
            "change": change,
            "within_noise": within,
        }
    return {
        "n_examples": len(ids),
        "identical_response_share": identical / len(ids) if ids else None,
        "rates": rates,
        "tolerance": tolerance,
    }


def reproduction_report(
    saved_dir: str | Path,
    rerun_dir: str | Path,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    tolerance: float = TOLERANCE,
) -> dict[str, Any]:
    import yaml

    saved_dir, rerun_dir = Path(saved_dir), Path(rerun_dir)
    saved_cfg = yaml.safe_load((saved_dir / "config.yaml").read_text())
    rerun_cfg = yaml.safe_load((rerun_dir / "config.yaml").read_text())
    differences, one_sided = config_differences(saved_cfg, rerun_cfg)
    track_b = compare_track_b(
        pooled_rows(saved_dir / "trackb_eval" / "summary_trained.json", splits),
        pooled_rows(rerun_dir / "trackb_eval" / "summary_trained.json", splits),
        n_resamples,
        seed,
        tolerance,
    )
    reproduced = not differences and all(v["within_noise"] for v in track_b["rates"].values())
    return {
        "saved_run": str(saved_dir),
        "rerun": str(rerun_dir),
        "splits": list(splits),
        "config_differences": differences,
        "config_keys_in_one_file_only": one_sided,
        "final_eval": {"saved": last_eval_row(saved_dir), "rerun": last_eval_row(rerun_dir)},
        "track_b": track_b,
        "reproduced": reproduced,
        "rule": (
            f"shared config keys identical apart from run_name, and for each checked rate "
            f"the paired "
            f"95% interval of (rerun - saved) contains 0 and |change| <= {tolerance}"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--saved", required=True, help="the saved run, e.g. results/trackb_full")
    parser.add_argument("--rerun", required=True, help="the retrained run directory")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()
    report = reproduction_report(
        args.saved, args.rerun, n_resamples=args.resamples, seed=args.seed, tolerance=args.tolerance
    )
    out = Path(args.rerun) / "reproduction.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    tb = report["track_b"]
    print(f"identical responses: {tb['identical_response_share']:.3f} of {tb['n_examples']}")
    for name, entry in tb["rates"].items():
        c = entry["change"]
        print(
            f"{name:34s} saved {entry['saved']:.3f}  rerun {entry['rerun']:.3f}  "
            f"change {c['difference']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"
        )
    for line in report["config_differences"]:
        print(f"config differs: {line}")
    for line in report["config_keys_in_one_file_only"]:
        print(f"config key in one file only (not blocking): {line}")
    print(f"reproduced: {report['reproduced']}  (wrote {out})")


if __name__ == "__main__":
    main()
