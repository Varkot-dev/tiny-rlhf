"""Figure for one ablation sweep: final held-out numbers against the swept setting (M6).

Three panels, one measure each, sharing the x axis (the swept value):
    1. preference accuracy on held-out pairs
    2. reward margin on held-out pairs
    3. change in the preferred response's log-probability relative to the reference
       (below zero means the preferred response became less likely)

Every point is the final evaluation of one run, with a 95% bootstrap interval over the
held-out pairs. Runs that have not finished are left out and named in the footer.

Usage:
    python scripts/plot_sweep.py --sweep lora_rank
    python scripts/plot_sweep.py --sweep configs/sweeps/data_size.yaml
Reads:
    configs/sweeps/<name>.yaml and results/<run>/eval_pairs_final.csv for each run
Writes:
    assets/m6_sweep_<name>.png unless --out is given
    results/sweeps/<name>/summary.json (the plotted numbers)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tiny_rlhf.plotting import SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, apply_style
from tiny_rlhf.sweeps import Sweep, load_sweep, sweep_summary

ROOT = Path(__file__).resolve().parents[1]
PANELS = (
    ("accuracy", "Preference accuracy", "share of held-out pairs with margin > 0"),
    ("reward_margin", "Reward margin", "chosen reward - rejected reward"),
    ("chosen_logp_change", "Preferred response log-probability", "change vs reference (nats)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", required=True, help="sweep name or path to its YAML file")
    parser.add_argument("--root", default=str(ROOT), help="repo root holding results/")
    parser.add_argument("--out", help="output PNG path")
    parser.add_argument("--resamples", type=int, default=2000)
    return parser.parse_args()


def resolve_sweep(arg: str) -> Sweep:
    path = Path(arg)
    if not path.exists():
        path = ROOT / "configs" / "sweeps" / f"{arg}.yaml"
    return load_sweep(path)


def x_positions(sweep: Sweep, runs: list[dict]) -> tuple[list[float], bool]:
    """x coordinate per run and whether the axis is categorical.

    Numeric values are used as they are. If the sweep asks for a categorical axis, or a
    value is not a number, runs are placed by their value's rank, so two runs with the
    same value (for example 30% noise with two different losses) share an x position.
    """
    numeric = all(
        isinstance(r["value"], (int, float)) and not isinstance(r["value"], bool) for r in runs
    )
    if numeric and sweep.x_scale != "categorical":
        return [float(r["value"]) for r in runs], False
    order = []
    for r in runs:
        if r["value"] not in order:
            order.append(r["value"])
    return [float(order.index(r["value"])) for r in runs], True


def make_figure(sweep: Sweep, summary: dict, out_path: Path, root: Path) -> None:
    runs = summary["runs"]
    xs, categorical = x_positions(sweep, runs)
    series_names = []
    for r in runs:
        name = r["series"] or "runs"
        if name not in series_names:
            series_names.append(name)
    # A second series at the same x is nudged sideways so the intervals do not overlap.
    nudge = 0.16 if categorical and len(series_names) > 1 else 0.0

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True)
    for ax, (key, title, ylabel) in zip(axes, PANELS, strict=True):
        for s_index, name in enumerate(series_names):
            color = SERIES[s_index]
            idx = [i for i, r in enumerate(runs) if (r["series"] or "runs") == name]
            px = [xs[i] + nudge * (s_index - (len(series_names) - 1) / 2) for i in idx]
            mean = [runs[i][key]["mean"] for i in idx]
            low = [runs[i][key]["mean"] - runs[i][key]["ci_low"] for i in idx]
            high = [runs[i][key]["ci_high"] - runs[i][key]["mean"] for i in idx]
            if len(idx) > 1:
                ax.plot(px, mean, color=color, linewidth=2, zorder=2)
            ax.errorbar(
                px,
                mean,
                yerr=[low, high],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.5,
                capsize=3,
                markersize=8,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                label=name,
                zorder=3,
            )
            for x, m, lo, hi in zip(px, mean, low, high, strict=True):
                # The value goes just past the end of its own error bar: above it for the
                # first series, below it for the second. Next to the point it collided with
                # the connecting line and with neighbouring points' labels.
                above = s_index == 0
                ax.annotate(
                    f"{m:.3f}" if abs(m) < 10 else f"{m:.1f}",
                    xy=(x, m + hi if above else m - lo),
                    xytext=(0, 4 if above else -4),
                    ha="center",
                    va="bottom" if above else "top",
                    textcoords="offset points",
                    fontsize=8.5,
                    color=TEXT_PRIMARY,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"},
                    zorder=4,
                )
        if key == "accuracy":
            ax.set_ylim(0, 1.02)
            ax.axhline(0.5, color=TEXT_SECONDARY, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
        else:
            ax.axhline(0.0, color=TEXT_SECONDARY, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(sweep.x_label or sweep.field)
        if categorical:
            order = []
            for r in runs:
                if r["value"] not in order:
                    order.append(r["value"])
            ax.set_xticks(range(len(order)), [str(v) for v in order])
            ax.set_xlim(-0.5, len(order) - 0.5)
        elif sweep.x_scale == "log":
            ax.set_xscale("log")
            ax.set_xticks(sorted(set(xs)), [f"{v:g}" for v in sorted(set(xs))])
            ax.minorticks_off()
        ax.margins(x=0.12, y=0.18)
    if len(series_names) > 1:
        axes[0].legend(loc="lower left", fontsize=8.5)

    n_pairs = sorted({r["n_pairs"] for r in runs})
    fig.suptitle(
        f"Sweep: {sweep.description or sweep.name}",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )
    footer = (
        f"Final held-out evaluation of each run ({', '.join(str(n) for n in n_pairs)} pairs), "
        "bars are 95% bootstrap intervals. "
        f"Source: results/<run>/eval_pairs_final.csv for {', '.join(r['run_name'] for r in runs)}; "
        "numbers in " + f"results/sweeps/{sweep.name}/summary.json; drawn by scripts/plot_sweep.py."
    )
    if summary["missing_runs"]:
        footer += " Not finished and not shown: " + ", ".join(summary["missing_runs"]) + "."
    fig.text(
        0.01, 0.005, footer, fontsize=7.5, color=TEXT_SECONDARY, ha="left", va="bottom", wrap=True
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95), w_pad=2.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    sweep = resolve_sweep(args.sweep)
    summary = sweep_summary(sweep, root, n_resamples=args.resamples)
    if not summary["runs"]:
        raise SystemExit(f"no finished runs for sweep {sweep.name}: {summary['missing_runs']}")
    summary_path = root / "results" / "sweeps" / sweep.name / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    out_path = Path(args.out) if args.out else root / "assets" / f"m6_sweep_{sweep.name}.png"
    make_figure(sweep, summary, out_path, root)
    for r in summary["runs"]:
        acc = r["accuracy"]
        print(
            f"{r['run_name']:>28}  accuracy {acc['mean']:.3f} [{acc['ci_low']:.3f}, "
            f"{acc['ci_high']:.3f}]  margin {r['reward_margin']['mean']:+.4f}  "
            f"chosen logp change {r['chosen_logp_change']['mean']:+.3f}"
        )
    print(f"wrote {summary_path} and {out_path}")


if __name__ == "__main__":
    main()
