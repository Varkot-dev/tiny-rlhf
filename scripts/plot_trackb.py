"""Hero figure: Track B before and after training.

Usage:
    python scripts/plot_trackb.py \\
        --base results/trackb_full/trackb_eval/summary_base.json \\
        --trained results/trackb_full/trackb_eval/summary_trained.json

Two panels, one measure each, the same five test files on the x axis:
    left   on-task rate on documents that contain a planted instruction
    right  task-correct rate on documents that contain none (did it get worse at its job?)
Bars are the base model and the trained model. Error bars are the 95% bootstrap
intervals stored in the summaries. Writes assets/m5_trackb_<run>.png, where <run> is
the directory two levels above the summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tiny_rlhf.plotting import SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, apply_style

ROOT = Path(__file__).resolve().parents[1]
SPLIT_LABELS = {
    "test_in_dist": "in\ndistribution",
    "test_heldout_task": "held-out\ntask",
    "test_heldout_doc": "held-out\ndocument style",
    "test_heldout_injection": "held-out\ninjection type",
    "test_heldout_all": "all three\nheld out",
}
PANELS = (
    ("injected", "on_task", "On-task rate, planted instruction in the document"),
    ("no_injection", "task_correct", "Task-correct rate, no planted instruction"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="summary_base.json")
    parser.add_argument("--trained", required=True, help="summary_trained.json")
    parser.add_argument("--out", help="default: assets/m5_trackb_<run>.png")
    return parser.parse_args()


def shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def draw_panel(ax, summaries: dict[str, dict], splits: list[str], group: str, metric: str) -> None:
    x = np.arange(len(splits))
    width = 0.36
    for k, (label, summary) in enumerate(summaries.items()):
        stats = [summary["splits"][s]["groups"][group]["metrics"][metric] for s in splits]
        rates = np.array([st["rate"] for st in stats])
        low = rates - np.array([st["ci_low"] for st in stats])
        high = np.array([st["ci_high"] for st in stats]) - rates
        # A 0.04 gap between the two bars of a pair keeps them visually separate.
        pos = x + (k - 0.5) * (width + 0.04)
        ax.bar(pos, rates, width, color=SERIES[k], label=label, zorder=2)
        ax.errorbar(
            pos,
            rates,
            yerr=[low, high],
            fmt="none",
            ecolor=TEXT_PRIMARY,
            elinewidth=1.2,
            capsize=3,
            zorder=3,
        )
        for p, rate, top in zip(pos, rates, rates + high, strict=True):
            ax.annotate(
                f"{rate:.2f}",
                xy=(p, min(top, 1.0)),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=TEXT_PRIMARY,
            )
    counts = [summaries[next(iter(summaries))]["splits"][s]["groups"][group]["n"] for s in splits]
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{SPLIT_LABELS.get(s, s)}\nn = {n}" for s, n in zip(splits, counts, strict=True)]
    )
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="x", visible=False)


def main() -> None:
    args = parse_args()
    base_path, trained_path = Path(args.base), Path(args.trained)
    base = json.loads(base_path.read_text())
    trained = json.loads(trained_path.read_text())
    splits = [s for s in SPLIT_LABELS if s in base["splits"] and s in trained["splits"]]
    if not splits:
        raise SystemExit("the summaries contain none of the five test files")
    summaries = {"base model": base, "after DPO on Track B": trained}

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (group, metric, title) in zip(axes, PANELS, strict=True):
        draw_panel(ax, summaries, splits, group, metric)
        ax.set_title(title)
    axes[0].set_ylabel("share of examples")
    axes[0].legend(loc="upper left", ncols=2, bbox_to_anchor=(0, 1.0))
    fig.suptitle(
        f"Staying on the user's task: {base.get('model', 'model')} before and after training",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.005,
        f"Source: {shown(base_path)} and {shown(trained_path)}. Error bars: 95% bootstrap "
        f"intervals over examples. Greedy decoding. Drawn by scripts/plot_trackb.py.",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95), w_pad=2)
    run = base_path.resolve().parent.parent.name
    out = Path(args.out) if args.out else ROOT / "assets" / f"m5_trackb_{run}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {shown(out)}")


if __name__ == "__main__":
    main()
