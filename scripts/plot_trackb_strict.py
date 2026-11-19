"""Track B under strict scoring: every model on every test split, and pooled.

Reads the files written by scripts/rescore_trackb.py and draws two panels:
(a) on-task rate with a planted instruction, strict (on_task_strict, injected examples)
(b) task-correct rate with nothing planted, strict (task_correct_strict, clean examples)
for the base model and every rescored run, per test split and pooled over the five
test splits, with 95% bootstrap intervals over examples.

The per-run hero figures (assets/m5_trackb_<run>.png) are left as the record of the
original lenient scoring; this figure is the strict counterpart for all runs at once.

Usage:
    python scripts/plot_trackb_strict.py
Writes:
    assets/m5_trackb_strict.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from tiny_rlhf.trackb.evaluate import DEFAULT_SPLITS

ROOT = Path(__file__).resolve().parents[1]
FIRST_RUN = "trackb_full"
# Fixed colour order shared with scripts/plot_trackb_variants.py: colour follows the
# model, never its position among the bars that exist.
KNOWN = (
    "base",
    "trackb_full",
    "trackb_beta0.5",
    "trackb_steps60",
    "trackb_sft1",
    "trackb_kto",
    "trackb_kto_beta0.5",
    "trackb_grpo",
)
LEGEND = {
    "base": "base model",
    "trackb_full": "DPO, first run (beta 0.1, 375 steps)",
    "trackb_beta0.5": "DPO, beta 0.5",
    "trackb_steps60": "DPO, 60 steps",
    "trackb_sft1": "DPO plus preferred-response NLL",
    "trackb_kto": "KTO, beta 0.1",
    "trackb_kto_beta0.5": "KTO, beta 0.5",
    "trackb_grpo": "GRPO, checker reward (beta 0.04)",
}
SPLIT_LABELS = {
    "test_in_dist": "in\ndistribution",
    "test_heldout_task": "held-out\ntask",
    "test_heldout_doc": "held-out\ndocument style",
    "test_heldout_injection": "held-out\ninjection type",
    "test_heldout_all": "all three\nheld out",
    "pooled": "pooled\n(five splits)",
}


def load_entities(results: Path) -> dict[str, dict[str, Any]]:
    """name -> rescored summary, base first, then the runs in the comparison's order."""
    compare = json.loads((results / "trackb_rescored_compare.json").read_text())
    entities = {
        "base": json.loads((results / FIRST_RUN / "trackb_eval" / "rescored_base.json").read_text())
    }
    for run in compare["runs"]:
        # Repeats with other seeds (configs/seeds/) are read together in
        # scripts/plot_seeds.py; here they would only crowd the bars.
        if re.search(r"_seed\d+$", run):
            continue
        entities[run] = json.loads(
            (results / run / "trackb_eval" / "rescored_trained.json").read_text()
        )
    return entities


def stat(summary: dict[str, Any], split: str, group: str, metric: str) -> dict[str, Any]:
    block = summary["pooled_test_splits"] if split == "pooled" else summary["splits"][split]
    return block["groups"][group]["metrics"][metric]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--out", default=str(ROOT / "assets" / "m5_trackb_strict.png"))
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from tiny_rlhf.plotting import SERIES, TEXT_PRIMARY, TEXT_SECONDARY, apply_style

    entities = load_entities(Path(args.results))
    order = list(KNOWN) + [n for n in entities if n not in KNOWN]
    names = sorted(entities, key=order.index)  # same bar order as plot_trackb_variants.py
    colors = {n: SERIES[order.index(n) % len(SERIES)] for n in names}
    groups = [*DEFAULT_SPLITS, "pooled"]

    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 9.5), sharex=True)
    panels = (
        ("injected", "on_task_strict", "On task with a planted instruction (strict)"),
        ("no_injection", "task_correct_strict", "Task correct with nothing planted (strict)"),
    )
    width = 0.8 / len(names)
    x = np.arange(len(groups), dtype=float)
    x[-1] += 0.35  # set the pooled group slightly apart from the five splits
    for ax, (group, metric, title) in zip(axes, panels, strict=True):
        for k, name in enumerate(names):
            values = [stat(entities[name], g, group, metric) for g in groups]
            rates = np.array([np.nan if v["rate"] is None else v["rate"] for v in values])
            low = rates - np.array([np.nan if v["ci_low"] is None else v["ci_low"] for v in values])
            high = (
                np.array([np.nan if v["ci_high"] is None else v["ci_high"] for v in values]) - rates
            )
            pos = x - 0.4 + width * (k + 0.5)
            ax.bar(pos, rates, width * 0.88, color=colors[name], label=LEGEND.get(name, name))
            ax.errorbar(
                pos,
                rates,
                yerr=[low, high],
                fmt="none",
                ecolor=TEXT_PRIMARY,
                elinewidth=1.0,
                capsize=2,
            )
            # Value labels on the pooled group only; the per-split bars carry intervals.
            ax.annotate(
                f"{rates[-1]:.2f}",
                xy=(pos[-1], min(rates[-1] + high[-1], 1.0)),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8 if len(names) <= 5 else 6.5,  # narrower bars, smaller labels
                rotation=0 if len(names) <= 5 else 90,
                color=TEXT_PRIMARY,
            )
        ax.axvline((x[-2] + x[-1]) / 2, color=TEXT_SECONDARY, lw=0.8, ls=":")
        ax.set_ylim(0, 1.08)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_title(title)
        ax.set_ylabel("share of examples")
        ax.grid(axis="x", visible=False)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([SPLIT_LABELS.get(g, g) for g in groups])
    handles, labels = axes[0].get_legend_handles_labels()
    ncol = len(names) if len(names) <= 5 else -(-len(names) // 2)
    fig.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(
        "Track B, strict scoring: Qwen2.5-0.5B-Instruct before and after training with LoRA "
        "(150 examples per split, 95% bootstrap intervals)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.008,
        "Source: results/<run>/trackb_eval/rescored_*.json from scripts/rescore_trackb.py "
        "(saved responses rescored with tiny_rlhf.trackb.checks.task_correct_strict). "
        "Drawn by scripts/plot_trackb_strict.py.",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.92), h_pad=2.0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
