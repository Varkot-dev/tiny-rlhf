"""M6 figure: the beta sweep. Drift from the reference against preference accuracy.

Left panel: one point per beta. x is how far the trained policy moved from the
reference, the mean per-token KL(reference || policy) on held-out preferred responses
(scripts/eval_drift.py). y is preference accuracy on the same held-out pairs. Bars are
95% bootstrap intervals in both directions.

Right panel: for each beta, the change in the summed log-probability of the preferred
response and of the rejected response relative to the reference. This is the check
from docs/PLAN.md section 2: a run can widen the margin by pushing both responses down.

Usage:
    python scripts/plot_beta_sweep.py
    python scripts/plot_beta_sweep.py --sweep configs/sweeps/beta.yaml --out assets/x.png
Reads:
    results/<run>/eval_pairs_final.csv and results/<run>/drift.json for each run
Writes:
    assets/m6_beta_sweep.png unless --out is given
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
from tiny_rlhf.sweeps import load_sweep, sweep_summary

ROOT = Path(__file__).resolve().parents[1]
DRIFT_KEY = "kl_reference_to_policy_per_token"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", default=str(ROOT / "configs" / "sweeps" / "beta.yaml"))
    parser.add_argument("--root", default=str(ROOT), help="repo root holding results/")
    parser.add_argument("--out", help="output PNG path")
    parser.add_argument("--resamples", type=int, default=2000)
    return parser.parse_args()


def bars(entry: dict) -> tuple[float, list[list[float]]]:
    """(mean, [[distance down], [distance up]]) for matplotlib's errorbar."""
    return entry["mean"], [[entry["mean"] - entry["ci_low"]], [entry["ci_high"] - entry["mean"]]]


def make_figure(summary: dict, runs: list[dict], out_path: Path) -> None:
    apply_style()
    fig, (ax_main, ax_logp) = plt.subplots(
        1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1]}
    )

    # One series (the sweep), so one colour; each point is named directly by its beta.
    for r in runs:
        kl, kl_err = bars(r["drift"][DRIFT_KEY])
        acc, acc_err = bars(r["accuracy"])
        ax_main.errorbar(
            [kl],
            [acc],
            xerr=kl_err,
            yerr=acc_err,
            fmt="o",
            color=SERIES[0],
            ecolor=SERIES[0],
            elinewidth=1.5,
            capsize=3,
            markersize=9,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            zorder=3,
        )
        ax_main.annotate(
            f"beta = {r['value']:g}",
            xy=(kl, acc),
            xytext=(10, 8),
            textcoords="offset points",
            fontsize=9.5,
            color=TEXT_PRIMARY,
        )
    ordered = sorted(runs, key=lambda r: r["value"])
    ax_main.plot(
        [r["drift"][DRIFT_KEY]["mean"] for r in ordered],
        [r["accuracy"]["mean"] for r in ordered],
        color=SERIES[0],
        linewidth=1.2,
        alpha=0.45,
        zorder=2,
    )
    ax_main.set_title("Drift from the reference vs preference accuracy")
    ax_main.set_xlabel("mean per-token KL(reference || policy) on held-out responses (nats)")
    ax_main.set_ylabel("share of held-out pairs with margin > 0")
    ax_main.set_xlim(left=0)
    ax_main.margins(x=0.25, y=0.25)

    labels = [f"{r['value']:g}" for r in ordered]
    positions = list(range(len(ordered)))
    for offset, key, name, color in (
        (-0.12, "chosen_logp_change", "preferred response", SERIES[0]),
        (0.12, "rejected_logp_change", "rejected response", SERIES[1]),
    ):
        means = [r[key]["mean"] for r in ordered]
        ax_logp.errorbar(
            [p + offset for p in positions],
            means,
            yerr=[
                [r[key]["mean"] - r[key]["ci_low"] for r in ordered],
                [r[key]["ci_high"] - r[key]["mean"] for r in ordered],
            ],
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
        ax_logp.annotate(
            name,
            xy=(positions[-1] + offset, means[-1]),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT_PRIMARY,
        )
    ax_logp.axhline(0.0, color=TEXT_SECONDARY, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax_logp.set_xticks(positions, labels)
    ax_logp.set_xlim(-0.6, len(positions) - 0.4 + 1.0)
    ax_logp.set_title("Did the responses get more or less likely?")
    ax_logp.set_xlabel("beta")
    ax_logp.set_ylabel("change in log pi(y|x) vs reference (nats)")
    ax_logp.legend(loc="best", fontsize=8.5)
    ax_logp.margins(y=0.2)

    n_pairs = sorted({r["n_pairs"] for r in runs})
    fig.suptitle(
        "beta sweep: how far the policy moves, and what that buys",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )
    footer = (
        f"Final held-out evaluation of each run ({', '.join(str(n) for n in n_pairs)} pairs); bars "
        "are 95% bootstrap intervals. Source: eval_pairs_final.csv and drift.json under "
        + ", ".join(f"results/{r['run_name']}" for r in ordered)
        + f"; numbers in results/sweeps/{summary['sweep']}/summary.json; "
        "drawn by scripts/plot_beta_sweep.py."
    )
    if summary["missing_runs"] or summary.get("runs_without_drift"):
        footer += (
            " Not shown: "
            + ", ".join(summary["missing_runs"] + summary.get("runs_without_drift", []))
            + "."
        )
    fig.text(
        0.01, 0.005, footer, fontsize=7.5, color=TEXT_SECONDARY, ha="left", va="bottom", wrap=True
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94), w_pad=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    sweep = load_sweep(args.sweep)
    summary = sweep_summary(sweep, root, n_resamples=args.resamples)
    runs = [r for r in summary["runs"] if "drift" in r]
    summary["runs_without_drift"] = [r["run_name"] for r in summary["runs"] if "drift" not in r]
    if not runs:
        raise SystemExit(
            "no run of this sweep has both eval_pairs_final.csv and drift.json. Run "
            "scripts/run_sweep.py, then scripts/eval_drift.py --run results/<run> for each run."
        )
    summary_path = root / "results" / "sweeps" / sweep.name / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    out_path = Path(args.out) if args.out else root / "assets" / "m6_beta_sweep.png"
    make_figure(summary, runs, out_path)
    for r in sorted(runs, key=lambda r: r["value"]):
        print(
            f"beta {r['value']:g}: KL {r['drift'][DRIFT_KEY]['mean']:.5f}, accuracy "
            f"{r['accuracy']['mean']:.3f}, "
            f"chosen logp change {r['chosen_logp_change']['mean']:+.3f}"
        )
    print(f"wrote {summary_path} and {out_path}")


if __name__ == "__main__":
    main()
