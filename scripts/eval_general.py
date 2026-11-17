"""General regression check: did Track B training disturb the model on ordinary chat data?

Usage:
    python scripts/eval_general.py --model Qwen/Qwen2.5-0.5B-Instruct \\
        --adapter checkpoints/trackb_full --out results/trackb_full/general_regression.json

On N held-out conversations from trl-lib/ultrafeedback_binarized (test split, the same
seeded selection tiny_rlhf.data.load_pairs makes), it compares the base model with the
base model plus the adapter on the tokens of the dataset's preferred response:
mean negative log-likelihood per token under each, their paired difference, and the
mean per-token KL(base || trained). All with 95% bootstrap intervals over conversations.

What this shows and does not show is spelled out in src/tiny_rlhf/trackb/general.py.
In short: it measures drift in next-token probabilities on text unlike the training
data. It does not judge generated text and proves nothing about capabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tiny_rlhf.data import load_pairs, pairs_fingerprint
from tiny_rlhf.trackb.evaluate import DEFAULT_RESAMPLES, load_model_for_eval
from tiny_rlhf.trackb.general import response_nll_and_kl, summarize_general
from tiny_rlhf.train_dpo import git_sha, library_versions, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True, help="directory of the saved LoRA adapter")
    parser.add_argument("--out", required=True, help="path of the JSON file to write")
    parser.add_argument("--dataset", default="trl-lib/ultrafeedback_binarized")
    parser.add_argument("--n", type=int, default=200, help="conversations to request")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    tokenizer, base = load_model_for_eval(args.model)
    _, trained = load_model_for_eval(args.model, args.adapter)
    base.to(device)
    trained.to(device)
    pairs = load_pairs(args.dataset, "test", args.n, args.seed, tokenizer, args.max_length)
    values = response_nll_and_kl(
        base,
        trained,
        pairs,
        pad_token_id=tokenizer.pad_token_id,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        amp=amp,
    )
    summary = summarize_general(values, args.resamples, args.seed)
    summary.update(
        model=args.model,
        adapter=args.adapter,
        dataset=args.dataset,
        split="test",
        n_requested=args.n,
        max_length=args.max_length,
        pairs_sha256=pairs_fingerprint(pairs, args.max_length),
        device=str(device),
        amp_fp16=amp,
        git_sha=git_sha(),
        versions=library_versions(),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    change = summary["nll_per_token_change"]
    kl = summary["kl_base_to_trained_per_token"]
    print(
        f"{summary['n_conversations']} conversations: NLL per token changed by "
        f"{change['difference']:+.4f} [{change['ci_low']:+.4f}, {change['ci_high']:+.4f}], "
        f"KL(base || trained) per token {kl['mean']:.5f}; wrote {out}"
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
