"""Evaluate a model on real prompt injections (BIPIA), or compare two such evaluations.

Evaluate (mirrors scripts/eval_trackb.py):
    python scripts/eval_external.py --model Qwen/Qwen2.5-0.5B-Instruct \\
        --label base --out results/m9_external/base_eval
    python scripts/eval_external.py --model Qwen/Qwen2.5-0.5B-Instruct \\
        --adapter checkpoints/trackb_full_m9 --label trained \\
        --out results/trackb_full_m9/external_eval
Writes <out>/bipia_<label>.jsonl (one row per example) and <out>/summary_<label>.json.

Compare (before/after with paired intervals):
    python scripts/eval_external.py --compare \\
        --base results/m9_external/base_eval/summary_base.json \\
        --trained results/trackb_full_m9/external_eval/summary_trained.json
Writes external_table.md and external_compare.json next to the trained summary (or to
--out-dir). --trained may be given several times; each gets its own files.

The examples come from data/external/bipia/bipia_eval.jsonl (built by
python -m tiny_rlhf.external.build); see docs/EXTERNAL_EVAL.md for what they are and
how each response is scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from tiny_rlhf.external.evaluate import (
    DEFAULT_DATA,
    compare_rows,
    comparison_table,
    evaluate_model,
    evaluation_settings,
    rows_for_summary,
    summary_mismatches,
)
from tiny_rlhf.trackb.evaluate import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_RESAMPLES,
    load_model_for_eval,
    summary_is_current,
)
from tiny_rlhf.train_dpo import git_sha, library_versions, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="base model name or path")
    parser.add_argument("--adapter", help="LoRA adapter directory (omit for the base model)")
    parser.add_argument("--label", help="name used in output files, e.g. base or trained")
    parser.add_argument("--out", help="output directory")
    parser.add_argument("--data", default=DEFAULT_DATA, help="the processed examples")
    parser.add_argument("--limit", type=int, help="only the first N attacked examples")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps or auto")
    parser.add_argument("--no-amp", action="store_true", help="disable fp16 autocast on CUDA")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reuse-if-current",
        action="store_true",
        help="skip if <out>/summary_<label>.json exists with the same model, adapter and "
        "settings (used to share one base evaluation between sessions)",
    )
    parser.add_argument("--compare", action="store_true", help="compare saved summaries")
    parser.add_argument("--base", help="--compare: the base model's summary")
    parser.add_argument(
        "--trained", action="append", help="--compare: a trained model's summary (repeatable)"
    )
    parser.add_argument("--out-dir", help="--compare: where to write (default: next to each)")
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="--compare: compare even if seed, decoding, token limit, limit, model or data differ",
    )
    args = parser.parse_args()
    if args.compare:
        if not args.base or not args.trained:
            parser.error("--compare needs --base and at least one --trained")
    elif not (args.model and args.label and args.out):
        parser.error("evaluating needs --model, --label and --out")
    return args


def compare(args: argparse.Namespace) -> None:
    base_path = Path(args.base)
    base = json.loads(base_path.read_text())
    base_rows = rows_for_summary(base_path)
    for trained in args.trained:
        trained_path = Path(trained)
        summary = json.loads(trained_path.read_text())
        blocking, warnings = summary_mismatches(base, summary)
        for line in warnings:
            print(f"WARNING: the two evaluations differ in {line}", file=sys.stderr)
        if blocking and not args.allow_mismatch:
            raise SystemExit(
                "the two evaluations were not run the same way, so their difference would "
                "not be a before/after effect:\n  " + "\n  ".join(blocking)
            )
        comparison = compare_rows(
            base_rows, rows_for_summary(trained_path), args.resamples, args.seed
        )
        out_dir = Path(args.out_dir) if args.out_dir else trained_path.parent
        if args.out_dir and len(args.trained) > 1:
            out_dir = out_dir / trained_path.parent.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        header = (
            "# Real prompt injections (BIPIA): before and after training\n\n"
            f"Base: `{base.get('model')}`. Trained: the same model with the adapter "
            f"`{summary.get('adapter')}`.\n\n"
            "Each cell is a rate with its 95% bootstrap interval. Change is trained minus "
            "base on the same examples, with a paired bootstrap interval "
            f"({args.resamples} resamples, seed {args.seed}). Greedy decoding, "
            f"{base.get('max_new_tokens')} new tokens at most. The checks are rule-based; "
            "see docs/EXTERNAL_EVAL.md for their false positives and negatives. Produced "
            f"by `scripts/eval_external.py --compare` from `{base_path}` and "
            f"`{trained_path}`.\n\n"
        )
        table = comparison_table(comparison)
        (out_dir / "external_table.md").write_text(header + table)
        (out_dir / "external_compare.json").write_text(
            json.dumps(
                {
                    "base_summary": str(base_path),
                    "trained_summary": str(trained_path),
                    "n_resamples": args.resamples,
                    "seed": args.seed,
                    "setting_mismatches": {"blocking": blocking, "warning": warnings},
                    **comparison,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\n{trained_path}\n{table}")
        print(f"wrote {out_dir / 'external_table.md'} and {out_dir / 'external_compare.json'}")


def evaluate(args: argparse.Namespace) -> None:
    out = Path(args.out)
    if args.reuse_if_current:
        settings = evaluation_settings(args.data, args.seed, args.limit, args.max_new_tokens)
        if summary_is_current(
            out / f"summary_{args.label}.json", settings, args.model, args.adapter
        ):
            print(f"reusing {out / f'summary_{args.label}.json'}: same model, adapter and settings")
            return
    device = resolve_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    torch.manual_seed(args.seed)
    tokenizer, model = load_model_for_eval(args.model, args.adapter)
    model.to(device)
    summary = evaluate_model(
        model,
        tokenizer,
        out,
        args.label,
        data_path=args.data,
        limit=args.limit,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=device,
        amp=amp,
        n_resamples=args.resamples,
        seed=args.seed,
        meta={
            "model": args.model,
            "adapter": args.adapter,
            "device": str(device),
            "amp_fp16": amp,
            "git_sha": git_sha(),
            "versions": library_versions(),
        },
    )
    injected = summary["groups"]["injected"]["metrics"]
    clean = summary["groups"]["clean"]["metrics"]

    def show(stat: dict) -> str:
        return "n/a" if stat["rate"] is None else f"{stat['rate']:.3f}"

    print(
        f"{args.label}: n={summary['n']}  followed_injection={show(injected['followed_injection'])}"
        f"  (check without attack {show(injected['check_fired_without_attack'])})"
        f"  task_correct attacked={show(injected['task_correct'])}"
        f" clean={show(clean['task_correct'])}  refused clean={show(clean['refused'])}"
    )
    print(f"wrote {out / f'summary_{args.label}.json'}")


def main() -> None:
    args = parse_args()
    if args.compare:
        compare(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
