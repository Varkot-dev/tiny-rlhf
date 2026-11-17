"""Rescore saved Track B generations with strict task correctness (no GPU needed).

The lenient check used in the original evaluations accepts an answer that appears
anywhere in the response, so numbered lists and lists of every item passed "how many"
and "which item" tasks by accident. This recomputes every check from the saved
responses, confirms the lenient values are reproduced exactly, and reports lenient and
strict scores side by side. See tiny_rlhf.trackb.rescore and
tiny_rlhf.trackb.checks.task_correct_strict.

Usage:
    python scripts/rescore_trackb.py                     # every run with a trained evaluation
    python scripts/rescore_trackb.py --runs trackb_full trackb_beta0.5
Writes (new files; the original summaries are not changed):
    results/<run>/trackb_eval/rescored_<label>.json
    results/trackb_rescored_compare.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tiny_rlhf.trackb.rescore import rescore_all

ROOT = Path(__file__).resolve().parents[1]


def fmt(stat: dict, key: str = "rate") -> str:
    value = stat.get(key)
    return "  n/a" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--base-run", default="trackb_full")
    parser.add_argument("--runs", nargs="*", help="default: every run with summary_trained.json")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    comparison = rescore_all(
        Path(args.results), args.base_run, args.runs, args.resamples, args.seed
    )
    bad = {k: v for k, v in comparison["lenient_mismatches_on_rescoring"].items() if v}
    if bad:
        raise SystemExit(
            f"lenient scores were not reproduced from the saved responses: {bad}. "
            "The checks changed since these runs; strict and lenient numbers would not be "
            "comparable."
        )

    print("Task-correct by task category, five test splits: lenient -> strict")
    for model, tasks in comparison["by_task_category"].items():
        for task, stat in tasks.items():
            print(
                f"  {model:16} {task:16} n={stat['n']:3d}  "
                f"{fmt(stat['task_correct'])} -> {fmt(stat['task_correct_strict'])}"
            )
    print("\nPooled over the five test splits, strict scoring (change is paired):")
    for run, result in comparison["runs"].items():
        pooled = result["pooled_test_splits"]
        on = pooled["injected"]["metrics"]["on_task_strict"]
        ok = pooled["no_injection"]["metrics"]["task_correct_strict"]
        for name, stat in (("on-task, planted instruction", on), ("task-correct, none", ok)):
            ch = stat["change"]
            print(
                f"  {run:16} {name:30} base {fmt(stat['base'])} trained {fmt(stat['trained'])}"
                f"  change {ch['difference']:+.3f} [{ch['ci_low']:+.3f}, {ch['ci_high']:+.3f}]"
            )
    for note in comparison["not_compared"]:
        print(f"not compared: {note}")
    for path in comparison["written"]:
        print(f"wrote {Path(path).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
