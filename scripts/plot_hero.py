"""The README's opening figure: every Track B run, behaviour against collateral change.

One point per training run: up is better at the user's task when a document carries a
planted instruction (pooled strict on-task rate, five test splits), left is less changed
on ordinary text (increase in negative log-likelihood per token on 183 general
conversations). DPO and KTO at beta 0.1 and 0.5 have three seeds each (seed 0 and the
repeats under configs/seeds/); GRPO has one. The base model sits at zero change.

Colour is the method (the same colours as the other Track B figures: DPO green, KTO
purple, GRPO red), marker shape repeats it for colour-blind readers, and fill is beta
(hollow 0.1, filled 0.5). Bars are 95% bootstrap intervals.

Usage:
    python scripts/plot_hero.py
Writes assets/hero_tradeoff.png and results/hero_values.json (every plotted value).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# (group in results/seeds_summary.json, method, beta)
SEED_GROUPS = (
    ("trackb_full", "DPO", 0.1),
    ("trackb_beta0.5", "DPO", 0.5),
    ("trackb_kto", "KTO", 0.1),
    ("trackb_kto_beta0.5", "KTO", 0.5),
)
GRPO_RUN = "trackb_grpo"
# Same entity colours as scripts/plot_trackb_variants.py (validated together, all pairs:
# the red-green pair is in the colour-blind warning band, hence the marker shapes).
METHOD_STYLE = {
    "DPO": {"color": "#1baf7a", "marker": "o"},
    "KTO": {"color": "#6250d6", "marker": "s"},
    "GRPO": {"color": "#e34948", "marker": "^"},
}


def collect(results: Path) -> dict[str, Any]:
    """Every plotted value, read from saved result files only."""
    seeds = json.loads((results / "seeds_summary.json").read_text())
    points = []
    for group, method, beta in SEED_GROUPS:
        for seed, stats in sorted(seeds["groups"][group]["seeds"].items()):
            points.append(
                {
                    "run": stats["run"],
                    "method": method,
                    "beta": beta,
                    "seed": int(seed),
                    "on_task": stats["on_task_strict"],
                    "drift": stats["nll_change_per_token"],
                    "median_words": stats["median_words"],
                }
            )
    rescored = json.loads(
        (results / GRPO_RUN / "trackb_eval" / "rescored_trained.json").read_text()
    )
    general = json.loads((results / GRPO_RUN / "general_regression.json").read_text())
    grpo_beta = json.loads((results / GRPO_RUN / "run_info.json").read_text())["trl_settings"][
        "beta"
    ]
    points.append(
        {
            "run": GRPO_RUN,
            "method": "GRPO",
            "beta": grpo_beta,
            "seed": 0,
            "on_task": rescored["pooled_test_splits"]["groups"]["injected"]["metrics"][
                "on_task_strict"
            ],
            "drift": general["nll_per_token_change"],
            "median_words": None,
        }
    )
    return {
        "base": {"on_task": seeds["base"]["on_task_strict"], "drift": 0.0},
        "points": points,
        "sources": [
            "results/seeds_summary.json",
            f"results/{GRPO_RUN}/trackb_eval/rescored_trained.json",
            f"results/{GRPO_RUN}/general_regression.json",
        ],
    }


def plot(values: dict[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from tiny_rlhf.plotting import SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, apply_style

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6.2))
    base = values["base"]["on_task"]["rate"]
    ax.axhline(base, color=TEXT_SECONDARY, lw=0.8, ls="--", zorder=1)
    ax.scatter(
        [0], [base], marker="D", s=70, color=TEXT_SECONDARY, edgecolor=SURFACE, lw=1.5, zorder=4
    )
    for p in values["points"]:
        style = METHOD_STYLE[p["method"]]
        x, y = p["drift"]["difference"], p["on_task"]["rate"]
        ax.errorbar(
            x,
            y,
            xerr=[[x - p["drift"]["ci_low"]], [p["drift"]["ci_high"] - x]],
            yerr=[[y - p["on_task"]["ci_low"]], [p["on_task"]["ci_high"] - y]],
            fmt="none",
            ecolor=style["color"],
            elinewidth=1,
            alpha=0.35,
            zorder=2,
        )
        filled = p["method"] == "GRPO" or p["beta"] >= 0.5
        ax.scatter(
            [x],
            [y],
            marker=style["marker"],
            s=80 if p["method"] == "GRPO" else 60,
            facecolor=style["color"] if filled else SURFACE,
            edgecolor=style["color"],
            lw=1.8,
            zorder=3,
        )

    def label(text: str, xy: tuple[float, float], at: tuple[float, float]) -> None:
        """Text at a fixed data position, with a thin leader line to the point."""
        ax.annotate(
            text,
            xy,
            xytext=at,
            textcoords="data",
            fontsize=9,
            color=TEXT_PRIMARY,
            va="center",
            arrowprops={"arrowstyle": "-", "color": TEXT_SECONDARY, "lw": 0.7, "shrinkB": 5},
        )

    by: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for p in values["points"]:
        by.setdefault((p["method"], p["beta"]), []).append(p)

    def xy(q: dict[str, Any]) -> tuple[float, float]:
        return q["drift"]["difference"], q["on_task"]["rate"]

    def centre(key: tuple[str, float]) -> tuple[float, float]:
        pts = [xy(q) for q in by[key]]
        return sum(a for a, _ in pts) / len(pts), sum(b for _, b in pts) / len(pts)

    (grpo,) = (q for q in values["points"] if q["method"] == "GRPO")
    label(
        "GRPO, checker reward (1 seed):\nas good as the best, least changed", xy(grpo), (0.0, 0.80)
    )
    label("KTO, beta 0.5: 3 of 3 seeds held", centre(("KTO", 0.5)), (0.42, 0.745))
    dpo05 = sorted(by[("DPO", 0.5)], key=lambda q: q["on_task"]["rate"])
    label("DPO, beta 0.5: seeds 0 and 1 held", xy(dpo05[-1]), (0.55, 0.705))
    worst = dpo05[0]
    label(
        f"DPO, beta 0.5, seed {worst['seed']}:\n"
        f"collapsed to {worst['median_words']:.0f}-word answers",
        xy(worst),
        (0.42, 0.385),
    )
    label("KTO, beta 0.1: 3 of 3 seeds held", centre(("KTO", 0.1)), (0.95, 0.58))
    label(
        "DPO, beta 0.1: collapsed on all 3 seeds\n(1-word answers)",
        centre(("DPO", 0.1)),
        (1.25, 0.455),
    )
    ax.annotate(
        f"base model: on task {base:.2f}",
        (0, base),
        xytext=(0.03, base + 0.012),
        textcoords="data",
        fontsize=9,
        color=TEXT_PRIMARY,
    )

    ax.set_xlim(left=-0.08)
    ax.set_ylim(0.15, 0.84)
    ax.set_xlabel(
        "change on ordinary text: negative log-likelihood per token, nats (left is better)"
    )
    ax.set_ylabel("on task with a planted instruction (strict, up is better)")
    ax.set_title(
        "Every Track B run: doing the user's task despite a planted instruction, against how "
        "much the model changed",
        loc="left",
    )
    handles = [
        Line2D([], [], ls="none", marker=s["marker"], ms=8, color=s["color"], label=m)
        for m, s in METHOD_STYLE.items()
    ] + [
        Line2D(
            [], [], ls="none", marker="o", ms=8, mfc=SURFACE, mec=TEXT_SECONDARY, label="beta 0.1"
        ),
        Line2D([], [], ls="none", marker="o", ms=8, color=TEXT_SECONDARY, label="beta 0.5"),
        Line2D([], [], ls="none", marker="D", ms=7, color=TEXT_SECONDARY, label="base model"),
    ]
    ax.legend(handles=handles, loc="upper right", ncols=2, frameon=False, fontsize=9)
    fig.text(
        0.01,
        -0.03,
        "Qwen2.5-0.5B-Instruct with LoRA; 750 Track B test examples over five splits; DPO and "
        "KTO three seeds each, GRPO one (beta 0.04). Bars: 95% bootstrap intervals.\nSource: "
        + ", ".join(values["sources"])
        + "; numbers in results/hero_values.json. Drawn by scripts/plot_hero.py.",
        fontsize=7.5,
        color=TEXT_SECONDARY,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--out", default=str(ROOT / "assets" / "hero_tradeoff.png"))
    parser.add_argument("--json-out", default=str(ROOT / "results" / "hero_values.json"))
    args = parser.parse_args()
    values = collect(Path(args.results))
    Path(args.json_out).write_text(json.dumps(values, indent=2) + "\n")
    plot(values, Path(args.out))
    print(f"wrote {args.out} and {args.json_out}")


if __name__ == "__main__":
    main()
