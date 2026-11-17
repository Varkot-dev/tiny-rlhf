"""Run an ablation sweep: one training run per value, one process per run.

Usage:
    python scripts/run_sweep.py --sweep configs/sweeps/beta.yaml --dry-run
    python scripts/run_sweep.py --sweep configs/sweeps/beta.yaml --skip-existing
    python scripts/run_sweep.py --sweep configs/sweeps/beta.yaml --only 0.5

Each run writes its usual files under results/<run_name>/. The sweep also keeps
results/sweeps/<name>/index.json, listing every run with its swept value and status.
A run that fails is recorded as failed and the sweep carries on with the next one.
"""

from __future__ import annotations

import argparse
import sys

from tiny_rlhf.sweeps import load_sweep, run_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", required=True, help="path to a file under configs/sweeps/")
    parser.add_argument(
        "--only", nargs="+", metavar="LABEL", help="run only these entries (label or run name)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip runs that already have results/<run_name>/run_info.json",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the commands and stop")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    index = run_sweep(
        load_sweep(args.sweep),
        only=args.only,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        for record in index["runs"]:
            print(f"{record['status']:>16}  {record['run_name']}")
        requested = set(args.only) if args.only else None
        failed = [
            r["run_name"]
            for r in index["runs"]
            if r["status"] == "failed"
            and (requested is None or {r["label"], r["run_name"]} & requested)
        ]
        sys.exit(1 if failed else 0)
