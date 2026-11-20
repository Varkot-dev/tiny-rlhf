"""Figure: base and trained models on real prompt injections (BIPIA).

Usage:
    python scripts/plot_external.py \\
        --base results/m9_external/base_eval/summary_base.json \\
        --trained "DPO, beta 0.1=results/trackb_full_m9/external_eval/summary_trained.json" \\
        --trained "DPO, beta 0.5=results/trackb_beta0.5_m9/external_eval/summary_trained.json"

--trained is repeatable, as LABEL=PATH or just PATH (then the run directory's name is the
label). Two panels, one bar per model:
    left   share of attacked examples where the attack's check fired, for all attacks
           and for each kind of check (format, marker, topic). A black tick on each bar
           is the same check run on that model's answer to the document WITHOUT the
           attack: an estimate of how often the check fires by chance
    right  task correct with the attack present, task correct without it, and refusals
           without it (did training cost the model its ordinary behaviour?)
Error bars are the 95% bootstrap intervals stored in the summaries. Writes
assets/m9_external_bipia.png unless --out is given, and with --summary-out a JSON file
holding every plotted value, so numbers quoted from the figure come from a saved file.
With --skip-missing a trained summary that does not exist yet is left out (with a note)
instead of stopping the script: the GPU queue redraws the figure as runs finish.
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
FOLLOWED = (
    (None, "all attacks"),
    ("format", "format\n(language, encoding, ...)"),
    ("marker", "marker\n(URL, brand name)"),
    ("topic", "topic\n(off-task content)"),
)
# Colour follows the model across figures: the same order as ENTITIES in
# scripts/plot_trackb_variants.py (tested), indexed into SERIES. A rerun made for this
# evaluation (<run>_m9) takes its saved run's colour.
ENTITY_ORDER = (
    "base",
    "trackb_full",
    "trackb_beta0.5",
    "trackb_steps60",
    "trackb_sft1",
    "trackb_kto",
    "trackb_kto_beta0.5",
    "trackb_grpo",
)
RERUN_SUFFIX = "_m9"
BEHAVIOUR = (
    ("injected", "task_correct", "task correct,\nattack present"),
    ("clean", "task_correct", "task correct,\nno attack"),
    ("clean", "refused", "refused,\nno attack"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="the base model's summary_base.json")
    parser.add_argument(
        "--trained", action="append", required=True, help="LABEL=PATH or PATH (repeatable)"
    )
    parser.add_argument("--out", help="default: assets/m9_external_bipia.png")
    parser.add_argument("--summary-out", help="also write every plotted value to this JSON file")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="leave out trained summaries that do not exist (at least one must)",
    )
    return parser.parse_args()


def shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def label_and_path(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label, Path(path)
    path = Path(spec)
    return path.resolve().parent.parent.name, path


def followed_stats(summary: dict, kind: str | None, metric: str) -> dict:
    if kind is None:
        return summary["groups"]["injected"]["metrics"][metric]
    block = summary["slices"]["check_kind"].get(kind)
    return block["metrics"][metric] if block else {"rate": None, "ci_low": None, "ci_high": None}


def model_colours(paths: list[Path]) -> list[str]:
    """One colour per model: the base model first, then each trained summary's run."""
    colours = []
    for k, path in enumerate(paths):
        run = "base" if k == 0 else path.resolve().parent.parent.name.removesuffix(RERUN_SUFFIX)
        colours.append(SERIES[ENTITY_ORDER.index(run)] if run in ENTITY_ORDER else SERIES[k])
    return colours


def draw_bars(
    ax,
    stats_per_model: list[list[dict]],
    ticks: list[list[dict]] | None,
    colours: list[str],
) -> None:
    """Grouped bars: one group per x position, one bar per model, with error bars."""
    n_models = len(stats_per_model)
    n_groups = len(stats_per_model[0])
    x = np.arange(n_groups)
    width = 0.8 / n_models
    for k, stats in enumerate(stats_per_model):
        rates = np.array([np.nan if s["rate"] is None else s["rate"] for s in stats])
        low = np.array([np.nan if s["ci_low"] is None else s["ci_low"] for s in stats])
        high = np.array([np.nan if s["ci_high"] is None else s["ci_high"] for s in stats])
        pos = x + (k - (n_models - 1) / 2) * width
        ax.bar(pos, rates, width * 0.9, color=colours[k], label=None, zorder=2)
        ax.errorbar(
            pos,
            rates,
            yerr=[rates - low, high - rates],
            fmt="none",
            ecolor=TEXT_PRIMARY,
            elinewidth=1.1,
            capsize=2.5,
            zorder=3,
        )
        if ticks is not None:
            base = np.array([np.nan if s["rate"] is None else s["rate"] for s in ticks[k]])
            ax.scatter(pos, base, marker="_", s=width * 900, color=TEXT_PRIMARY, lw=2, zorder=4)
        for p, rate, top in zip(pos, rates, high, strict=True):
            if np.isnan(rate):
                continue
            ax.annotate(
                f"{rate:.2f}",
                xy=(p, min(top, 1.0)),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=TEXT_PRIMARY,
            )
    ax.set_xticks(x)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="x", visible=False)


def main() -> None:
    args = parse_args()
    base_path = Path(args.base)
    trained = [label_and_path(t) for t in args.trained]
    if args.skip_missing:
        for label, path in trained:
            if not path.exists():
                print(f"left out (not evaluated yet): {label} ({path})")
        trained = [(label, path) for label, path in trained if path.exists()]
        if not trained:
            raise SystemExit("none of the trained summaries exist yet")
    models = [("base model", base_path)] + trained
    if len(models) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES) - 1} trained models fit the colour list")
    summaries = [(label, json.loads(path.read_text())) for label, path in models]
    sha = {s.get("examples_sha256") for _, s in summaries}
    if len(sha) != 1:
        raise SystemExit("the summaries were not made on the same examples")

    apply_style()
    fig, (ax_f, ax_b) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [4, 3]})
    followed = [
        [followed_stats(s, k, "followed_injection") for k, _ in FOLLOWED] for _, s in summaries
    ]
    chance = [
        [followed_stats(s, k, "check_fired_without_attack") for k, _ in FOLLOWED]
        for _, s in summaries
    ]
    colours = model_colours([path for _, path in models])
    draw_bars(ax_f, followed, chance, colours)
    first = summaries[0][1]
    counts = [
        first["groups"]["injected"]["n"]
        if k is None
        else first["slices"]["check_kind"].get(k, {}).get("n", 0)
        for k, _ in FOLLOWED
    ]
    ax_f.set_xticklabels(
        [f"{name}\nn = {n}" for (_, name), n in zip(FOLLOWED, counts, strict=True)]
    )
    ax_f.set_title("Followed the injected instruction (lower is better)")
    ax_f.set_ylabel("share of attacked examples")

    behaviour = [
        [s["groups"][group]["metrics"][metric] for group, metric, _ in BEHAVIOUR]
        for _, s in summaries
    ]
    draw_bars(ax_b, behaviour, None, colours)
    ax_b.set_xticklabels(
        [
            f"{name}\nn = {first['groups'][group]['metrics'][metric]['n']}"
            for group, metric, name in BEHAVIOUR
        ]
    )
    ax_b.set_title("Doing the user's task")
    ax_b.set_ylabel("share of examples")

    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[k]) for k in range(len(summaries))] + [
        plt.Line2D([], [], color=TEXT_PRIMARY, marker="_", ls="none", markersize=12, mew=2)
    ]
    labels = [label for label, _ in summaries] + ["same check, answer without the attack"]
    fig.suptitle(
        f"Real prompt injections (BIPIA): {first.get('model', 'model')}",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    # One legend for both panels, on its own row under the title.
    fig.legend(handles, labels, loc="upper left", ncols=len(labels), bbox_to_anchor=(0.005, 0.95))
    sources = ", ".join(shown(p) for _, p in models)
    fig.text(
        0.01,
        0.005,
        f"Source: {sources}\n"
        "Error bars: 95% bootstrap intervals over examples. Greedy decoding. Rule-based checks "
        "(docs/EXTERNAL_EVAL.md). Drawn by scripts/plot_external.py.",
        fontsize=7.5,
        color=TEXT_SECONDARY,
        wrap=True,
    )
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0.07, 1, 0.91), w_pad=2.5)
    out = Path(args.out) if args.out else ROOT / "assets" / "m9_external_bipia.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {shown(out)}")
    if args.summary_out:
        values = {
            label: {
                "summary": shown(path),
                "adapter": summary.get("adapter"),
                "followed_injection": {
                    name.split("\n")[0]: followed_stats(summary, kind, "followed_injection")
                    for kind, name in FOLLOWED
                },
                "check_fired_without_attack": {
                    name.split("\n")[0]: followed_stats(summary, kind, "check_fired_without_attack")
                    for kind, name in FOLLOWED
                },
                "behaviour": {
                    name.replace("\n", " "): summary["groups"][group]["metrics"][metric]
                    for group, metric, name in BEHAVIOUR
                },
            }
            for (label, path), (_, summary) in zip(models, summaries, strict=True)
        }
        summary_out = Path(args.summary_out)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(
            json.dumps({"figure": shown(out), "models": values}, indent=2) + "\n"
        )
        print(f"wrote {shown(summary_out)}")


if __name__ == "__main__":
    main()
