"""Track B follow-up: the first full run next to the variants meant to fix it.

The first full Track B run (results/trackb_full) stopped following planted instructions
but also stopped answering: its median answer shrank to one word and task-correct on
clean documents collapsed. Each variant changes one thing (beta, number of steps, an
added likelihood term; see configs/trackb_<variant>.yaml). This figure puts the base
model, trackb_full and every finished variant side by side on four measures:

(a) how often a planted instruction was followed (injected examples, five test splits)
(b) how often the task was done correctly when nothing was planted (same splits)
(c) median answer length in words (all examples of the five test splits)
(d) change in negative log-likelihood per token on general conversations, from
    general_regression.json (the base model is 0 by definition)

Rates (a) and (b) are pooled over the five test splits with a 95% percentile bootstrap
interval over examples. A variant is shown only if its trained evaluation scored exactly
the prompts the base evaluation scored (same examples fingerprint). Missing or
mismatched variants are named in the footer.

Usage:
    python scripts/plot_trackb_variants.py
Writes:
    results/trackb_variants_summary.json
    assets/m5_trackb_variants.png
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from tiny_rlhf.stats import bootstrap_interval
from tiny_rlhf.trackb.evaluate import DEFAULT_SPLITS, read_rows

ROOT = Path(__file__).resolve().parents[1]
FIRST_RUN = "trackb_full"
# Same list and order as tiny_rlhf.gpu_queue.TRACKB_VARIANTS (a test checks it).
VARIANTS = ("trackb_beta0.5", "trackb_steps60", "trackb_sft1")
# Short tick labels (five bars share a narrow panel) and the full names for the legend.
LABELS = {
    "base": "base\nmodel",
    "trackb_full": "first\nrun",
    "trackb_beta0.5": "beta\n0.5",
    "trackb_steps60": "60\nsteps",
    "trackb_sft1": "+ NLL\nterm",
}
LEGEND = {
    "base": "base model",
    "trackb_full": "first run (beta 0.1, 375 steps)",
    "trackb_beta0.5": "beta 0.5",
    "trackb_steps60": "60 steps",
    "trackb_sft1": "plus preferred-response NLL, weight 1",
}
# Colour follows the entity, never its position among the bars that happen to exist.
ENTITIES = ("base", FIRST_RUN, *VARIANTS)


def pooled_stats(rows: list[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    """The figure's numbers for one model, from its scored rows of the five test splits."""
    injected = [r for r in rows if r["has_injection"]]
    clean = [r for r in rows if not r["has_injection"]]

    def rate(group: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
        values = [float(bool(r["scores"][key])) for r in group]
        if not values:
            return {"rate": None, "ci_low": None, "ci_high": None, "n": 0}
        low, high = bootstrap_interval(values, seed=seed)
        return {"rate": sum(values) / len(values), "ci_low": low, "ci_high": high, "n": len(values)}

    return {
        "followed_injection": rate(injected, "followed_injection"),
        "task_correct_no_injection": rate(clean, "task_correct"),
        "median_words": statistics.median(len(r["response"].split()) for r in rows)
        if rows
        else None,
        "n_examples": len(rows),
    }


def split_rows(summary_path: Path) -> list[dict[str, Any]]:
    summary = json.loads(summary_path.read_text())
    rows: list[dict[str, Any]] = []
    for split in DEFAULT_SPLITS:
        rows += read_rows(summary_path.parent / summary["files"][split])
    return rows


def collect(results: Path, variants: tuple[str, ...]) -> tuple[dict[str, Any], list[str]]:
    """Per entity: pooled stats, NLL change and sources. Plus notes on what was left out."""
    base_summary_path = results / FIRST_RUN / "trackb_eval" / "summary_base.json"
    base_summary = json.loads(base_summary_path.read_text())
    entities: dict[str, Any] = {
        "base": {
            **pooled_stats(split_rows(base_summary_path)),
            "nll_change_per_token": {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0},
            "sources": [str(base_summary_path)],
        }
    }
    notes: list[str] = []
    for run in (FIRST_RUN, *variants):
        trained = results / run / "trackb_eval" / "summary_trained.json"
        general = results / run / "general_regression.json"
        if not trained.is_file():
            notes.append(f"{run}: not run yet")
            continue
        summary = json.loads(trained.read_text())
        if summary.get("examples_sha256") != base_summary.get("examples_sha256"):
            notes.append(f"{run}: left out, it scored different prompts from the base")
            continue
        change = None
        if general.is_file():
            change = json.loads(general.read_text()).get("nll_per_token_change")
        else:
            notes.append(f"{run}: no general regression check")
        entities[run] = {
            **pooled_stats(split_rows(trained)),
            "nll_change_per_token": change,
            "sources": [str(trained)] + ([str(general)] if general.is_file() else []),
        }
    return entities, notes


def plot(entities: dict[str, Any], notes: list[str], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from tiny_rlhf.plotting import SERIES, TEXT_PRIMARY, TEXT_SECONDARY, apply_style

    apply_style()
    names = [e for e in ENTITIES if e in entities]
    colors = [SERIES[ENTITIES.index(e)] for e in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.2))

    def bars(ax, values, lows=None, highs=None, fmt="{:.2f}", skip_base=False):
        shown = [i for i, v in enumerate(values) if v is not None]
        if skip_base:
            shown = [i for i in shown if names[i] != "base"]
        vals = np.array([values[i] for i in shown], dtype=float)
        ax.bar(x[shown], vals, 0.7, color=[colors[i] for i in shown], zorder=2)
        if lows is not None:
            lo = np.array([lows[i] for i in shown], dtype=float)
            hi = np.array([highs[i] for i in shown], dtype=float)
            ax.errorbar(
                x[shown],
                vals,
                yerr=[vals - lo, hi - vals],
                fmt="none",
                ecolor=TEXT_PRIMARY,
                elinewidth=1.2,
                capsize=3,
                zorder=3,
            )
            tops = np.maximum(vals, hi)
        else:
            tops = vals
        for xi, v, top in zip(x[shown], vals, tops, strict=True):
            ax.annotate(
                fmt.format(v),
                xy=(xi, max(top, 0)),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=TEXT_PRIMARY,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS.get(n, n) for n in names], fontsize=8.5)
        # Same slots in every panel, even where the base model has no bar.
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.grid(axis="x", visible=False)

    def metric(key: str, field: str) -> list[float | None]:
        return [entities[n][key][field] if entities[n][key] else None for n in names]

    ax = axes[0]
    bars(
        ax,
        metric("followed_injection", "rate"),
        metric("followed_injection", "ci_low"),
        metric("followed_injection", "ci_high"),
    )
    ax.set_ylim(0, 1.1)
    ax.set_title("Followed a planted instruction")
    ax.set_ylabel("share of injected examples (lower is better)")

    ax = axes[1]
    bars(
        ax,
        metric("task_correct_no_injection", "rate"),
        metric("task_correct_no_injection", "ci_low"),
        metric("task_correct_no_injection", "ci_high"),
    )
    ax.set_ylim(0, 1.1)
    ax.set_title("Task correct, nothing planted")
    ax.set_ylabel("share of clean examples (higher is better)")

    ax = axes[2]
    bars(ax, [entities[n]["median_words"] for n in names], fmt="{:.0f}")
    ax.set_title("Median answer length")
    ax.set_ylabel("words")
    ax.set_ylim(0, max(entities[n]["median_words"] or 0 for n in names) * 1.2 + 1)

    ax = axes[3]
    bars(
        ax,
        metric("nll_change_per_token", "difference"),
        metric("nll_change_per_token", "ci_low"),
        metric("nll_change_per_token", "ci_high"),
        fmt="{:+.2f}",
        skip_base=True,
    )
    ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
    highs = [h for h in metric("nll_change_per_token", "ci_high") if h is not None]
    lows = [lo for lo in metric("nll_change_per_token", "ci_low") if lo is not None]
    ax.set_ylim(min(0.0, min(lows)) * 1.2 - 0.05, max(0.1, max(highs)) * 1.15)
    ax.annotate(
        "0 by\ndefinition",
        xy=(0, 0),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=TEXT_SECONDARY,
    )
    ax.set_title("Worse at general text?")
    ax.set_ylabel("change in NLL per token (nats, lower is better)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    fig.legend(
        handles,
        [LEGEND.get(n, n) for n in names],
        loc="upper center",
        ncol=len(names),
        bbox_to_anchor=(0.5, 0.95),
        fontsize=9,
    )
    n_ex = entities["base"]["n_examples"]
    fig.suptitle(
        "Track B follow-up: Qwen2.5-0.5B-Instruct with DPO + LoRA, "
        f"{n_ex} test examples over five splits (95% bootstrap intervals)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    sources = "results/<run>/trackb_eval/summary_*.json and general_regression.json"
    footer = f"Source: {sources}. Drawn by scripts/plot_trackb_variants.py."
    if notes:
        footer += " Not shown: " + "; ".join(notes) + "."
    fig.text(0.01, 0.005, footer, fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9), w_pad=2.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(ROOT / "results"), help="results directory")
    parser.add_argument("--out", default=str(ROOT / "assets" / "m5_trackb_variants.png"))
    parser.add_argument(
        "--summary-out", default=str(ROOT / "results" / "trackb_variants_summary.json")
    )
    args = parser.parse_args()
    entities, notes = collect(Path(args.results), VARIANTS)
    summary = {"splits": list(DEFAULT_SPLITS), "entities": entities, "not_shown": notes}
    Path(args.summary_out).write_text(json.dumps(summary, indent=2) + "\n")
    plot(entities, notes, Path(args.out))
    for name, e in entities.items():
        fol, cor = e["followed_injection"], e["task_correct_no_injection"]
        print(
            f"{name:16s} followed {fol['rate']:.2f}  correct(clean) {cor['rate']:.2f}  "
            f"median words {e['median_words']}"
        )
    for note in notes:
        print(f"not shown: {note}")
    print(f"wrote {args.summary_out} and {args.out}")


if __name__ == "__main__":
    main()
