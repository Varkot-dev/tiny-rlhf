"""Run the remaining GPU jobs of the project back to back, pushing after each one.

Meant to be started from the cell in docs/COLAB_QUEUE.md, inside a fresh clone:

    python scripts/colab_queue.py                   # everything, in priority order
    python scripts/colab_queue.py --list            # the queue and what is already done
    python scripts/colab_queue.py --stages m5,m3    # only some stages
    python scripts/colab_queue.py --dry-run         # print the commands, run nothing
    python scripts/colab_queue.py --budget-min 50   # start no job that will not fit

Stages, in priority order: m5 (Track B), m3 (reward model), m2 (TRL parity on the full
model), then the four sweeps: beta, lora_rank, data_size, label_noise.

A job is skipped when its completion file is already in the repo, so running this again
after a disconnect resumes where it stopped. A job that fails is recorded in
results/queue_status.json, the end of its output is committed to
results/colab_failures/<job>.log, its partial files are discarded, jobs that depend on it
are skipped, and the queue carries on with the rest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tiny_rlhf.gpu_queue import STAGES, build_queue, describe, format_table, run_queue, select

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help=f"comma-separated stages to run, in this order (default: {','.join(STAGES)})",
    )
    parser.add_argument("--list", action="store_true", help="show the queue and stop")
    parser.add_argument("--dry-run", action="store_true", help="print the commands and stop")
    parser.add_argument("--no-push", action="store_true", help="commit locally, do not push")
    parser.add_argument(
        "--budget-min",
        type=float,
        help="minutes available; a job whose (rough) estimate does not fit is not started",
    )
    parser.add_argument("--branch", help="branch to push to (default: the checked-out one)")
    parser.add_argument(
        "--retry-oom",
        action="store_true",
        help="try again the jobs an earlier session saw run out of GPU memory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    jobs = build_queue(ROOT)
    if args.list:
        total = 0.0
        for row in describe(select(jobs, stages), ROOT):
            print(
                f"{row['stage']:<12} {row['job']:<26} about {row['estimate_min']:>3g} min  "
                f"{row['status']}"
            )
            if row["status"] == "pending":
                total += row["estimate_min"]
        print(f"\npending, excluding fallbacks: about {total:g} min (rough guesses)")
        return 0
    results = run_queue(
        jobs,
        root=ROOT,
        stages=stages,
        dry_run=args.dry_run,
        do_push=not args.no_push,
        budget_min=args.budget_min,
        branch=args.branch,
        retry_oom=args.retry_oom,
    )
    if args.dry_run:
        return 0
    print("\n" + format_table(results))
    failed = [name for name, row in results.items() if row["status"] == "failed"]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}. Details: results/colab_failures/ and")
        print("results/queue_status.json (both pushed).")
    # Everything else still ran and was pushed; the exit status tells the cell.
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
