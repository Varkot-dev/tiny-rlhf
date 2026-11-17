"""Where did the log-probability go? Per-pair analysis of a finished DPO run.

The first full run (results/dpo_full_t4) improved the loss and the reward margin while
the preferred responses became less likely. This script looks at that per held-out
pair, using only files the run saved plus a re-tokenization of the same held-out pairs
(checked against the run's fingerprint):

- how each pair's preferred and other response changed relative to the reference,
  in total and per scored token,
- which of the four "up/down" quadrants each pair falls in,
- whether the change depends on response length, and whether truncation matters,
- how concentrated the drop is (share of the total drop from the 10% largest drops).

Usage:
    python scripts/analyze_logp_change.py --run results/dpo_full_t4
Writes:
    results/<run>/logp_change.json
    assets/m1_logp_change_<run>.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def logp_change_stats(
    chosen_change: np.ndarray,
    rejected_change: np.ndarray,
    chosen_tokens: np.ndarray,
    rejected_tokens: np.ndarray,
    chosen_truncated: np.ndarray,
) -> dict[str, float | int]:
    """Summary of per-pair log-probability changes (policy minus reference, in nats).

    Args:
        chosen_change, rejected_change: [n] summed log-prob change of each response.
        chosen_tokens, rejected_tokens: [n] response tokens actually scored.
        chosen_truncated: [n] bool, the preferred response was cut by max_length.
    """
    n = len(chosen_change)
    per_tok_c = chosen_change / chosen_tokens
    per_tok_r = rejected_change / rejected_tokens
    drop_r = np.sort(np.clip(-rejected_change, 0, None))[::-1]
    top = max(1, int(round(0.1 * n)))
    return {
        "n_pairs": n,
        "mean_change_chosen": float(chosen_change.mean()),
        "mean_change_rejected": float(rejected_change.mean()),
        "mean_change_per_token_chosen": float(per_tok_c.mean()),
        "mean_change_per_token_rejected": float(per_tok_r.mean()),
        "median_change_per_token_chosen": float(np.median(per_tok_c)),
        "median_change_per_token_rejected": float(np.median(per_tok_r)),
        "frac_both_down": float(((chosen_change < 0) & (rejected_change < 0)).mean()),
        "frac_chosen_up_rejected_down": float(((chosen_change > 0) & (rejected_change < 0)).mean()),
        "frac_both_up": float(((chosen_change > 0) & (rejected_change > 0)).mean()),
        "frac_chosen_down_rejected_up": float(((chosen_change < 0) & (rejected_change > 0)).mean()),
        "preference_accuracy": float(((chosen_change - rejected_change) > 0).mean()),
        "corr_change_vs_tokens_chosen": float(np.corrcoef(chosen_change, chosen_tokens)[0, 1]),
        "corr_change_vs_tokens_rejected": float(
            np.corrcoef(rejected_change, rejected_tokens)[0, 1]
        ),
        "n_chosen_truncated": int(chosen_truncated.sum()),
        "mean_change_per_token_chosen_truncated": float(per_tok_c[chosen_truncated].mean())
        if chosen_truncated.any()
        else float("nan"),
        "mean_change_per_token_chosen_not_truncated": float(per_tok_c[~chosen_truncated].mean())
        if (~chosen_truncated).any()
        else float("nan"),
        "share_of_rejected_drop_from_top_10pct_pairs": float(drop_r[:top].sum() / drop_r.sum())
        if drop_r.sum() > 0
        else float("nan"),
    }


def load_run(run_dir: Path) -> dict[str, np.ndarray]:
    """Per-pair changes from the run's files and token counts from re-tokenizing."""
    import yaml
    from transformers import AutoTokenizer

    from tiny_rlhf.data import load_pairs, pairs_fingerprint

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    info = json.loads((run_dir / "run_info.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    pairs = load_pairs(
        cfg["dataset_name"], "test", cfg["n_eval"], cfg["seed"], tokenizer, cfg["max_length"]
    )
    if pairs_fingerprint(pairs, cfg["max_length"]) != info.get("eval_pairs_sha256"):
        raise SystemExit("re-tokenized held-out pairs do not match the run's fingerprint")

    with (run_dir / "eval_pairs_final.csv").open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(pairs):
        raise SystemExit(f"{len(rows)} per-pair rows but {len(pairs)} held-out pairs")

    limit = cfg["max_length"] if cfg["max_length"] is not None else float("inf")

    def scored(prompt: list[int], response: list[int]) -> int:
        return int(min(len(response), max(limit - len(prompt), 0)))

    return {
        "chosen_change": np.array(
            [float(r["chosen_logp"]) - float(r["ref_chosen_logp"]) for r in rows]
        ),
        "rejected_change": np.array(
            [float(r["rejected_logp"]) - float(r["ref_rejected_logp"]) for r in rows]
        ),
        "chosen_tokens": np.array([scored(p.prompt_ids, p.chosen_ids) for p in pairs]),
        "rejected_tokens": np.array([scored(p.prompt_ids, p.rejected_ids) for p in pairs]),
        "chosen_truncated": np.array(
            [scored(p.prompt_ids, p.chosen_ids) < len(p.chosen_ids) for p in pairs]
        ),
        "beta": cfg["beta"],
        "model_name": cfg["model_name"],
    }


def plot(data: dict, stats: dict, run_dir: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tiny_rlhf.plotting import SERIES, TEXT_PRIMARY, TEXT_SECONDARY, apply_style

    apply_style()
    dc, dr = data["chosen_change"], data["rejected_change"]
    fig, (ax_q, ax_h, ax_l) = plt.subplots(1, 3, figsize=(15, 4.8))

    # Panel 1: each pair, other response's change against preferred response's change.
    lim = float(np.percentile(np.abs(np.concatenate([dc, dr])), 98)) * 1.1
    # One colour here: blue and orange mean "preferred" and "other" in the next panels.
    # Which side of the dashed line a point falls on says whether the pair is ranked right.
    ax_q.scatter(dr, dc, s=14, color=TEXT_SECONDARY, alpha=0.8, zorder=3)
    ax_q.plot([-lim, lim], [-lim, lim], color=TEXT_SECONDARY, lw=0.8, ls="--")
    ax_q.axhline(0, color=TEXT_SECONDARY, lw=0.8)
    ax_q.axvline(0, color=TEXT_SECONDARY, lw=0.8)
    ax_q.set_xlim(-lim, lim)
    ax_q.set_ylim(-lim, lim)
    corners = [
        (
            0.03,
            0.97,
            "top",
            "left",
            f"preferred up, other down: {stats['frac_chosen_up_rejected_down']:.0%}",
        ),
        (0.03, 0.03, "bottom", "left", f"both down: {stats['frac_both_down']:.0%}"),
        (0.97, 0.97, "top", "right", f"both up: {stats['frac_both_up']:.0%}"),
        (
            0.97,
            0.03,
            "bottom",
            "right",
            f"preferred down, other up: {stats['frac_chosen_down_rejected_up']:.0%}",
        ),
    ]
    for x, y, va, ha, text in corners:
        ax_q.text(
            x, y, text, transform=ax_q.transAxes, va=va, ha=ha, color=TEXT_PRIMARY, fontsize=9
        )
    ax_q.set_title("Change per held-out pair (nats)")
    ax_q.set_xlabel("other response: log pi - log pi_ref")
    ax_q.set_ylabel("preferred response: log pi - log pi_ref")
    acc = stats["preference_accuracy"]
    ax_q.text(
        -0.55 * lim,
        0.6 * lim,
        f"above the dashed line:\nranked right ({acc:.0%})",
        ha="center",
        va="center",
        fontsize=9,
        color=TEXT_PRIMARY,
    )
    ax_q.text(
        0.6 * lim,
        -0.45 * lim,
        f"below the dashed line:\nranked wrong ({1 - acc:.0%})",
        ha="center",
        va="center",
        fontsize=9,
        color=TEXT_PRIMARY,
    )

    # Panel 2: change per scored token, both responses.
    pc = dc / data["chosen_tokens"]
    pr = dr / data["rejected_tokens"]
    lo, hi = np.percentile(np.concatenate([pc, pr]), [1, 99])
    bins = np.linspace(lo, hi, 40)
    ax_h.hist(pr, bins=bins, color=SERIES[1], alpha=0.75, label="other response")
    ax_h.hist(pc, bins=bins, color=SERIES[0], alpha=0.75, label="preferred response")
    ax_h.axvline(0, color=TEXT_SECONDARY, lw=0.8)
    ax_h.set_title("Change per scored token (nats)")
    ax_h.set_xlabel("(log pi - log pi_ref) / tokens scored")
    ax_h.set_ylabel("held-out responses")
    ax_h.text(
        0.03,
        0.97,
        f"median, preferred: {stats['median_change_per_token_chosen']:+.3f}\n"
        f"median, other: {stats['median_change_per_token_rejected']:+.3f}",
        transform=ax_h.transAxes,
        va="top",
        fontsize=9,
        color=TEXT_PRIMARY,
    )

    # Panel 3: does the total change grow with length?
    ax_l.scatter(data["chosen_tokens"], dc, s=12, color=SERIES[0], label="preferred response")
    ax_l.scatter(data["rejected_tokens"], dr, s=12, color=SERIES[1], label="other response")
    ax_l.axhline(0, color=TEXT_SECONDARY, lw=0.8)
    ax_l.set_ylim(-lim, lim)
    ax_l.set_title("Total change against response length")
    ax_l.set_xlabel("response tokens scored")
    ax_l.set_ylabel("log pi - log pi_ref (nats)")
    ax_l.text(
        0.97,
        0.97,
        f"correlation: preferred {stats['corr_change_vs_tokens_chosen']:+.2f}, "
        f"other {stats['corr_change_vs_tokens_rejected']:+.2f}",
        transform=ax_l.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        color=TEXT_PRIMARY,
    )

    fig.suptitle(
        f"Where the log-probability went: {data['model_name']}, beta = {data['beta']}, "
        f"{stats['n_pairs']} held-out pairs (axes clipped at the 98th percentile)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.02,
        f"Source: {run_dir.relative_to(ROOT)}/eval_pairs_final.csv and a re-tokenization of the "
        "same held-out pairs (fingerprint checked). Drawn by scripts/analyze_logp_change.py.",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    handles, labels = ax_l.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.67, 0.95))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run directory, e.g. results/dpo_full_t4")
    parser.add_argument("--out", help="PNG path (default assets/m1_logp_change_<run>.png)")
    args = parser.parse_args()
    run_dir = Path(args.run).resolve()
    data = load_run(run_dir)
    stats = logp_change_stats(
        data["chosen_change"],
        data["rejected_change"],
        data["chosen_tokens"],
        data["rejected_tokens"],
        data["chosen_truncated"],
    )
    (run_dir / "logp_change.json").write_text(json.dumps(stats, indent=2) + "\n")
    out = Path(args.out) if args.out else ROOT / "assets" / f"m1_logp_change_{run_dir.name}.png"
    plot(data, stats, run_dir, out)
    print(json.dumps(stats, indent=2))
    print(f"wrote {run_dir.relative_to(ROOT)}/logp_change.json and {out}")


if __name__ == "__main__":
    main()
