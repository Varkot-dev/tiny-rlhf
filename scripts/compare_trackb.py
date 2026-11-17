"""Before/after table for Track B, with a paired bootstrap interval on every change.

Usage:
    python scripts/compare_trackb.py \\
        --base results/trackb_full/trackb_eval/summary_base.json \\
        --trained results/trackb_full/trackb_eval/summary_trained.json

Reads the two summaries and the per-example files they name, pairs the two models'
answers by example id, and writes next to the trained run (the parent of the trained
summary's directory, or --out-dir). The base summary may come from another run: the
follow-up variants are all compared with the one base evaluation in trackb_full.
    trackb_table.md      the table
    trackb_compare.json  every group and metric, with base, trained and change

"Change" is trained minus base on the same examples. Its interval comes from a paired
bootstrap: each resample draws the same example indices for both models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tiny_rlhf.trackb.evaluate import (
    DEFAULT_RESAMPLES,
    compare_rows,
    comparison_table,
    read_rows,
    summary_mismatches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="summary_base.json")
    parser.add_argument("--trained", required=True, help="summary_trained.json")
    parser.add_argument("--out-dir", help="default: the parent of the trained summary's directory")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="compare even if seed, decoding, token limit, limit, model or data differ",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path, trained_path = Path(args.base), Path(args.trained)
    base, trained = json.loads(base_path.read_text()), json.loads(trained_path.read_text())
    splits = [s for s in base["files"] if s in trained["files"]]
    if not splits:
        raise SystemExit("the two summaries share no split")
    blocking, warnings = summary_mismatches(base, trained)
    for line in warnings:
        print(f"WARNING: the two evaluations differ in {line}", file=sys.stderr)
    if blocking and not args.allow_mismatch:
        raise SystemExit(
            "the two evaluations were not run the same way, so their difference would not "
            "be a before/after effect:\n  "
            + "\n  ".join(blocking)
            + "\nRe-run the older one, or pass --allow-mismatch to compare anyway."
        )
    for line in blocking:
        print(f"WARNING (--allow-mismatch): the two evaluations differ in {line}", file=sys.stderr)
    comparison = {
        split: compare_rows(
            read_rows(base_path.parent / base["files"][split]),
            read_rows(trained_path.parent / trained["files"][split]),
            args.resamples,
            args.seed,
        )
        for split in splits
    }
    # Next to the TRAINED run: one base evaluation is shared by several trained variants,
    # and writing next to it would overwrite one variant's table with another's.
    out_dir = Path(args.out_dir) if args.out_dir else trained_path.parent.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "# Track B: before and after training\n\n"
        f"Base: `{base.get('model')}`. Trained: the same model with the adapter "
        f"`{trained.get('adapter')}`.\n\n"
        "Each cell is a rate with its 95% bootstrap interval. Change is trained minus base "
        "on the same examples, with a paired bootstrap interval "
        f"({args.resamples} resamples, seed {args.seed}). Greedy decoding, "
        f"{base.get('max_new_tokens')} new tokens at most. "
        "Produced by `scripts/compare_trackb.py` from "
        f"`{base_path}` and `{trained_path}`.\n\n"
    )
    (out_dir / "trackb_table.md").write_text(header + comparison_table(comparison))
    (out_dir / "trackb_compare.json").write_text(
        json.dumps(
            {
                "base_summary": str(base_path),
                "trained_summary": str(trained_path),
                "n_resamples": args.resamples,
                "seed": args.seed,
                "setting_mismatches": {"blocking": blocking, "warning": warnings},
                "splits": comparison,
            },
            indent=2,
        )
        + "\n"
    )
    print(comparison_table(comparison))
    print(f"wrote {out_dir / 'trackb_table.md'} and {out_dir / 'trackb_compare.json'}")


if __name__ == "__main__":
    main()
