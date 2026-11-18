"""Track B across seeds: do the conclusions survive a change of random seed?

Every Track B run so far used seed 0. The `seeds` queue stage repeats the four runs the
conclusions rest on (DPO and KTO at beta 0.1 and 0.5) with seeds 1 and 2; the configs under
configs/seeds/ differ from the originals only in seed and run_name. This script puts the
three seeds of each run side by side.

Per run it pools the five test splits and scores every saved answer with the strict
checks, computed exactly as the strict rescoring does (tiny_rlhf.trackb.rescore:
load_rescored, then summarize_rows with RESCORE_METRICS): planted instructions followed,
on task with an instruction planted, task correct with nothing planted. It adds the median
answer length in words and the general-text drift from results/<run>/general_regression.json.
Each value carries its 95% bootstrap interval over examples; across seeds it reports the
mean, minimum and maximum. Runs that are missing, or that scored different prompts from the
base evaluation, are left out and named.

Usage:
    python scripts/plot_seeds.py
Writes:
    results/seeds_summary.json
    assets/seeds_trackb.png
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from tiny_rlhf.trackb.evaluate import DEFAULT_SPLITS, summarize_rows
from tiny_rlhf.trackb.rescore import RESCORE_METRICS, load_rescored

ROOT = Path(__file__).resolve().parents[1]
BASE_RUN = "trackb_full"
SEEDS = (0, 1, 2)  # seed 0 is the original run
# (original run, label). Colours match scripts/plot_trackb_variants.py and
# scripts/plot_trackb_strict.py, which assign SERIES by this fixed order of models:
# base, trackb_full, trackb_beta0.5, trackb_steps60, trackb_sft1, trackb_kto, trackb_kto_beta0.5.
GROUPS = (
    ("trackb_full", "DPO\nbeta 0.1"),
    ("trackb_kto", "KTO\nbeta 0.1"),
    ("trackb_beta0.5", "DPO\nbeta 0.5"),
    ("trackb_kto_beta0.5", "KTO\nbeta 0.5"),
)
COLOR_SLOT = {"trackb_full": 1, "trackb_beta0.5": 2, "trackb_kto": 5, "trackb_kto_beta0.5": 6}
# (key in the per-run stats, panel title, which way is better)
RATE_METRICS = (
    ("followed_injection", "Followed a planted instruction", "lower"),
    ("on_task_strict", "On task, instruction planted (strict)", "higher"),
    ("task_correct_strict", "Task correct, nothing planted (strict)", "higher"),
)
GROUP_OF = {
    "followed_injection": "injected",
    "on_task_strict": "injected",
    "task_correct_strict": "no_injection",
}


def run_name(run: str, seed: int) -> str:
    return run if seed == 0 else f"{run}_seed{seed}"


def pooled_stats(summary_path: Path, n_resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """The figure's numbers for one evaluation, pooled over the five test splits."""
    loaded = load_rescored(summary_path)
    rows = [r for split in DEFAULT_SPLITS for r in loaded["rows"].get(split, [])]
    pooled = summarize_rows(rows, n_resamples, seed, RESCORE_METRICS)
    stats: dict[str, Any] = {
        key: pooled["groups"].get(GROUP_OF[key], {}).get("metrics", {}).get(key)
        for key, _, _ in RATE_METRICS
    }
    stats["median_words"] = (
        float(statistics.median(len(r["response"].split()) for r in rows)) if rows else None
    )
    stats["n_examples"] = len(rows)
    stats["examples_sha256"] = loaded["summary"].get("examples_sha256")
    stats["lenient_mismatches_on_rescoring"] = loaded["mismatches"]
    return stats


def spread(values: list[float]) -> dict[str, Any]:
    """Mean, minimum and maximum of the per-seed values that exist."""
    if not values:
        return {"mean": None, "min": None, "max": None, "n_seeds": 0}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "n_seeds": len(values),
    }


def collect(results: Path, n_resamples: int = 2000) -> dict[str, Any]:
    """Every group's per-seed stats and their spread, plus the base model's stats."""
    base_path = results / BASE_RUN / "trackb_eval" / "summary_base.json"
    base = pooled_stats(base_path, n_resamples)
    out: dict[str, Any] = {
        "base": {**base, "source": str(base_path.relative_to(results.parent))},
        "scoring": "strict task correctness, pooled over the five test splits; see "
        "tiny_rlhf.trackb.rescore",
        "interval": "95% percentile bootstrap over examples",
        "n_resamples": n_resamples,
        "groups": {},
        "missing": [],
    }
    for run, label in GROUPS:
        seeds: dict[str, Any] = {}
        for seed in SEEDS:
            name = run_name(run, seed)
            trained = results / name / "trackb_eval" / "summary_trained.json"
            if not trained.is_file():
                out["missing"].append(f"{name}: not run yet")
                continue
            stats = pooled_stats(trained, n_resamples)
            if stats["examples_sha256"] != base["examples_sha256"]:
                out["missing"].append(f"{name}: scored different prompts from the base")
                continue
            general = results / name / "general_regression.json"
            stats["nll_change_per_token"] = (
                json.loads(general.read_text()).get("nll_per_token_change")
                if general.is_file()
                else None
            )
            stats["run"] = name
            stats["seed"] = seed
            seeds[str(seed)] = stats
        summary: dict[str, Any] = {}
        for key, _, _ in RATE_METRICS:
            summary[key] = spread([s[key]["rate"] for s in seeds.values() if s.get(key)])
        summary["median_words"] = spread(
            [s["median_words"] for s in seeds.values() if s["median_words"] is not None]
        )
        summary["nll_change_per_token"] = spread(
            [
                s["nll_change_per_token"]["difference"]
                for s in seeds.values()
                if s.get("nll_change_per_token")
            ]
        )
        out["groups"][run] = {"label": label.replace("\n", ", "), "seeds": seeds, **summary}
    return out


def plot(data: dict[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from tiny_rlhf.plotting import SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, apply_style

    apply_style()
    panels = [
        *[(key, title, better, "rate") for key, title, better in RATE_METRICS],
        ("median_words", "Median answer length", "", "words"),
        ("nll_change_per_token", "Worse at general text?", "lower", "nll"),
    ]
    fig, grid = plt.subplots(2, 3, figsize=(16, 9.5))
    axes = list(grid.flat[: len(panels)])
    legend_ax = grid.flat[len(panels)]
    legend_ax.axis("off")
    offsets = {0: -0.22, 1: 0.0, 2: 0.22}
    markers = {0: "o", 1: "s", 2: "^"}
    x_positions = range(len(GROUPS))
    for ax, (key, title, better, kind) in zip(axes, panels, strict=True):
        for x, (run, _) in zip(x_positions, GROUPS, strict=True):
            group = data["groups"][run]
            color = SERIES[COLOR_SLOT[run]]
            for seed_key, stats in group["seeds"].items():
                seed = int(seed_key)
                if kind == "rate":
                    stat = stats.get(key)
                    if not stat or stat["rate"] is None:
                        continue
                    y, low, high = stat["rate"], stat["ci_low"], stat["ci_high"]
                elif kind == "words":
                    y, low, high = stats["median_words"], None, None
                else:
                    change = stats.get("nll_change_per_token")
                    if not change:
                        continue
                    y, low, high = change["difference"], change["ci_low"], change["ci_high"]
                yerr = None if low is None else [[y - low], [high - y]]
                ax.errorbar(
                    [x + offsets[seed]],
                    [y],
                    yerr=yerr,
                    fmt=markers[seed],
                    color=color,
                    ecolor=color,
                    elinewidth=1.3,
                    capsize=2.5,
                    markersize=7,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.2,
                    zorder=3,
                )
            mean = group[key]["mean"]
            if mean is not None:
                ax.plot([x - 0.34, x + 0.34], [mean, mean], color=TEXT_PRIMARY, lw=2, zorder=4)
                if kind == "words":
                    text = f"{mean:.0f}"
                elif kind == "nll":
                    text = f"{mean:+.2f}"
                else:
                    text = f"{mean:.2f}"
                # Means in a row along the top of the panel, clear of the points.
                ax.text(
                    x,
                    0.97,
                    f"mean {text}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    color=TEXT_PRIMARY,
                )
        base = data["base"]
        if kind == "rate" and base.get(key):
            ax.axhline(base[key]["rate"], color=TEXT_SECONDARY, lw=1, ls=(0, (4, 3)), zorder=1)
        elif kind == "words" and base.get("median_words") is not None:
            ax.axhline(base["median_words"], color=TEXT_SECONDARY, lw=1, ls=(0, (4, 3)), zorder=1)
        elif kind == "nll":
            ax.axhline(0.0, color=TEXT_SECONDARY, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(list(x_positions), [label for _, label in GROUPS])
        ax.set_xlim(-0.6, len(GROUPS) - 0.4)
        suffix = f"\n({better} is better)" if better else "\n"
        ax.set_title(title + suffix, fontsize=10.5)
        if kind == "rate":
            ax.set_ylim(0, 1.12)
            ax.set_ylabel("share of examples")
        elif kind == "words":
            # Headroom so the base line sits below the row of mean labels.
            tallest = [base.get("median_words") or 0.0] + [
                s["median_words"]
                for run, _ in GROUPS
                for s in data["groups"][run]["seeds"].values()
                if s["median_words"] is not None
            ]
            ax.set_ylim(0, max(tallest) * 1.3 + 1)
            ax.set_ylabel("words")
        else:
            ax.margins(y=0.18)
            ax.set_ylabel("change in NLL per token (nats)")
        ax.grid(axis="x", visible=False)

    handles = [
        Line2D([], [], marker=markers[s], ls="", color=TEXT_SECONDARY, markersize=7, label=label)
        for s, label in ((0, "seed 0 (original run)"), (1, "seed 1"), (2, "seed 2"))
    ]
    handles += [
        Line2D([], [], color=TEXT_PRIMARY, lw=2, label="mean of the seeds"),
        Line2D([], [], color=TEXT_SECONDARY, lw=1, ls=(0, (4, 3)), label="base model"),
    ]
    legend_ax.legend(handles=handles, loc="center", frameon=False, fontsize=10)
    fig.suptitle(
        "Track B across seeds: each point is one training run, bars are 95% bootstrap intervals",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    missing = "; ".join(data["missing"]) if data["missing"] else "none"
    fig.text(
        0.01,
        -0.04,
        "Pooled over the five test splits (750 examples), strict scoring, as "
        "tiny_rlhf.trackb.rescore. Seeds change LoRA initialisation and data order only.\n"
        f"Not shown: {missing}. Source: results/<run>/trackb_eval and general_regression.json; "
        "numbers in results/seeds_summary.json; drawn by scripts/plot_seeds.py.",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--out", default=str(ROOT / "assets" / "seeds_trackb.png"))
    parser.add_argument("--summary-out", default=None, help="default: <results>/seeds_summary.json")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    results = Path(args.results)
    data = collect(results, args.resamples)
    summary_out = Path(args.summary_out) if args.summary_out else results / "seeds_summary.json"
    summary_out.write_text(json.dumps(data, indent=2) + "\n")
    for run, _ in GROUPS:
        group = data["groups"][run]
        parts = []
        for key in ("followed_injection", "on_task_strict", "task_correct_strict"):
            g = group[key]
            if g["mean"] is not None:
                parts.append(f"{key} {g['mean']:.3f} [{g['min']:.3f}, {g['max']:.3f}]")
        print(f"{run:20} seeds {group['on_task_strict']['n_seeds']}  " + "  ".join(parts))
    plot(data, Path(args.out))
    print(f"wrote {summary_out} and {args.out}")


if __name__ == "__main__":
    main()
