"""How far did a trained policy move from its reference? Mean per-token KL (M6).

For the run's own held-out pairs, at every position of the preferred response, compare
the next-token distribution of the reference model with that of the trained policy:

    KL(reference || policy) = sum over the vocabulary of p_ref * (log p_ref - log p_policy)

averaged over the response positions of each pair, then over pairs. It is 0 when the
two models predict identically and grows as the policy drifts. This is the quantity the
KL penalty in the RLHF objective limits and that beta controls in DPO, so it is the x
axis of the beta-sweep figure.

The KL is computed by tiny_rlhf.trackb.general.response_nll_and_kl, the same function
the Track B general regression check uses. Only the preferred responses are scored, and
only response positions (prompt positions are not measured).

Usage:
    python scripts/eval_drift.py --run results/sweep_beta_0.1
    python scripts/eval_drift.py --run results/dpo_full --adapter checkpoints/dpo_full
Reads:
    <run>/config.yaml and <run>/run_info.json, and the LoRA adapter saved by the run
    (default checkpoints/<run name>; lost when a Colab runtime is recycled, so run this
    in the same session as training)
Writes:
    <run>/drift.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from tiny_rlhf.data import load_pairs, pairs_fingerprint
from tiny_rlhf.trackb.evaluate import DEFAULT_RESAMPLES, load_model_for_eval
from tiny_rlhf.trackb.general import response_nll_and_kl, summarize_general
from tiny_rlhf.train_dpo import git_sha, library_versions, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="a run directory under results/")
    parser.add_argument("--adapter", help="LoRA adapter directory (default checkpoints/<run>)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    adapter = Path(args.adapter) if args.adapter else Path("checkpoints") / run_dir.name
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(
            f"no LoRA adapter at {adapter}. It is saved by training (save_adapter: true) and "
            "is not in git, so measure drift in the same session as the training run."
        )

    device = resolve_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    tokenizer, reference = load_model_for_eval(cfg["model_name"])
    _, policy = load_model_for_eval(cfg["model_name"], adapter)
    reference.to(device)
    policy.to(device)

    # The same held-out pairs the run evaluated on: same dataset, count, seed and length.
    pairs = load_pairs(
        cfg["dataset_name"], "test", cfg["n_eval"], cfg["seed"], tokenizer, cfg["max_length"]
    )
    fingerprint = pairs_fingerprint(pairs, cfg["max_length"])
    info_path = run_dir / "run_info.json"
    recorded = (
        json.loads(info_path.read_text()).get("eval_pairs_sha256") if info_path.exists() else None
    )
    if recorded is not None and recorded != fingerprint:
        raise SystemExit("held-out pairs differ from the ones the run recorded; refusing to score")

    values = response_nll_and_kl(
        reference,
        policy,
        pairs,
        pad_token_id=tokenizer.pad_token_id,
        max_length=cfg["max_length"],
        batch_size=args.batch_size,
        device=device,
        amp=amp,
    )
    general = summarize_general(values, args.resamples, cfg["seed"])
    summary = {
        "run_name": run_dir.name,
        "beta": cfg.get("beta"),
        "kl_reference_to_policy_per_token": general["kl_base_to_trained_per_token"],
        "nll_per_token_reference": general["nll_per_token_base"],
        "nll_per_token_policy": general["nll_per_token_trained"],
        "nll_per_token_change": general["nll_per_token_change"],
        "n_pairs": general["n_conversations"],
        "mean_response_tokens_scored": general["mean_response_tokens_scored"],
        "scored": "preferred responses of the run's held-out pairs, response positions only",
        "interval": general["interval"],
        "n_resamples": general["n_resamples"],
        "seed": general["seed"],
        "eval_pairs_sha256": fingerprint,
        "pairs_match_run": None if recorded is None else True,
        "adapter": str(adapter),
        "device": str(device),
        "amp_fp16": amp,
        "git_sha": git_sha(),
        "versions": library_versions(),
    }
    (run_dir / "drift.json").write_text(json.dumps(summary, indent=2) + "\n")
    kl = summary["kl_reference_to_policy_per_token"]
    print(
        f"{run_dir.name}: KL(reference || policy) per token {kl['mean']:.4g} "
        f"[{kl['ci_low']:.4g}, {kl['ci_high']:.4g}] over {summary['n_pairs']} held-out pairs; "
        f"wrote {run_dir / 'drift.json'}"
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
