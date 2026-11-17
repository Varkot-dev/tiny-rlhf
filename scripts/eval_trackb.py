"""Generate answers for the Track B test files with one model and score them.

Usage:
    # before training
    python scripts/eval_trackb.py --model Qwen/Qwen2.5-0.5B-Instruct --label base \\
        --out results/trackb_full/trackb_eval --with-probes
    # after training
    python scripts/eval_trackb.py --model Qwen/Qwen2.5-0.5B-Instruct --label trained \\
        --adapter checkpoints/trackb_full --out results/trackb_full/trackb_eval --with-probes

Writes <out>/<split>_<label>.jsonl (every prompt, the generated answer and its checks)
and <out>/summary_<label>.json (rates with 95% bootstrap intervals). Metric definitions
are at the top of src/tiny_rlhf/trackb/evaluate.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from tiny_rlhf.trackb.evaluate import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_RESAMPLES,
    DEFAULT_SPLITS,
    evaluate_model,
    evaluation_settings,
    load_model_for_eval,
    summary_is_current,
)
from tiny_rlhf.trackb.probes import PROBE_NAMES
from tiny_rlhf.train_dpo import git_sha, library_versions, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Hugging Face model name")
    parser.add_argument("--adapter", help="directory of a saved LoRA adapter (omit for the base)")
    parser.add_argument("--label", required=True, help="base or trained; used in file names")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--data-dir", default="data/trackb")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help=f"test files or probes to score (default: the five test files). "
        f"Probes: {', '.join(PROBE_NAMES)}",
    )
    parser.add_argument("--with-probes", action="store_true", help="also score every probe set")
    parser.add_argument("--limit", type=int, help="score only the first N examples of each split")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true", help="float32 even on a GPU")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reuse-if-current",
        action="store_true",
        help="do nothing if summary_<label>.json already exists for exactly this model, "
        "adapter, data, seed, limit and decoding; otherwise evaluate again",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = list(args.splits)
    if args.with_probes:
        splits += [p for p in PROBE_NAMES if p not in splits]
    if args.reuse_if_current:
        settings = evaluation_settings(
            splits, args.data_dir, args.seed, args.limit, args.max_new_tokens
        )
        saved = Path(args.out) / f"summary_{args.label}.json"
        if summary_is_current(saved, settings, args.model, args.adapter):
            print(f"{saved} matches the current data and settings; not evaluating again")
            return
        if saved.exists():
            print(f"{saved} was made with different data or settings; evaluating again")
    device = resolve_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    torch.manual_seed(args.seed)
    tokenizer, model = load_model_for_eval(args.model, args.adapter)
    model.to(device)
    start = time.perf_counter()
    summary = evaluate_model(
        model,
        tokenizer,
        splits,
        args.out,
        args.label,
        data_dir=args.data_dir,
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
    print(
        f"scored {sum(s['n'] for s in summary['splits'].values())} examples in "
        f"{time.perf_counter() - start:.1f} s; wrote {args.out}/summary_{args.label}.json"
    )


if __name__ == "__main__":
    main()
