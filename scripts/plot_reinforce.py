"""Figure for the toy RL experiment (stretch item).

Reads results/<run>/{curves.csv, grad_variance.json, summary.json} written by
scripts/train_reinforce.py and draws three panels:
    1. expected reward against sequences sampled, for REINFORCE with each baseline
       (mean over seeds, band from the lowest to the highest seed), with the expected
       reward of the closed-form optimum as a dashed line
    2. spread of the gradient estimate per baseline at three points in training
    3. KL(pi || pi*) over training for REINFORCE (leave-one-out baseline) and for DPO
       with Bradley-Terry labels and with always-correct labels

Usage:
    python scripts/plot_reinforce.py --run results/reinforce_toy
Writes:
    assets/m_stretch_<run_name>.png
    results/<run_name>/plot_data.json   the numbers that were drawn
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tiny_rlhf.plotting import SERIES, TEXT_PRIMARY, TEXT_SECONDARY, apply_style

ROOT = Path(__file__).resolve().parents[1]
BASELINES = {
    "reinforce_none": ("no baseline", SERIES[0]),
    "reinforce_running_mean": ("running mean", SERIES[1]),
    "reinforce_leave_one_out": ("leave-one-out mean", SERIES[2]),
}
DISTANCE_METHODS = {
    "reinforce_leave_one_out": ("REINFORCE, leave-one-out", SERIES[2]),
    "dpo_bradley_terry": ("DPO, Bradley-Terry labels", SERIES[3]),
    "dpo_hard": ("DPO, always-correct labels", SERIES[4]),
}


def load_curves(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """method -> {"x": sequences seen [n_steps], metric: [n_seeds, n_steps]}."""
    by_method: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with path.open() as f:
        for row in csv.DictReader(f):
            by_method[row["method"]][int(row["seed"])].append(row)
    out = {}
    for method, seeds in by_method.items():
        first = next(iter(seeds.values()))
        out[method] = {"x": np.array([int(r["sequences_seen"]) for r in first])}
        for metric in ("expected_reward", "kl_from_ref", "kl_to_opt"):
            out[method][metric] = np.array(
                [[float(r[metric]) for r in rows] for rows in seeds.values()]
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="results/<run_name> directory")
    parser.add_argument("--out", help="output PNG (default assets/m_stretch_<run_name>.png)")
    args = parser.parse_args()
    run_dir = Path(args.run)
    out_path = Path(args.out) if args.out else ROOT / "assets" / f"m_stretch_{run_dir.name}.png"

    curves = load_curves(run_dir / "curves.csv")
    grad_variance = json.loads((run_dir / "grad_variance.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    task = summary["task"]
    n_seeds = next(iter(curves.values()))["expected_reward"].shape[0]
    plot_data: dict = {"expected_reward": {}, "grad_variance": grad_variance, "kl_to_opt": {}}

    apply_style()
    fig, (ax_reward, ax_var, ax_dist) = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # Panel 1: learning curves per baseline.
    for method, (label, color) in BASELINES.items():
        x, y = curves[method]["x"], curves[method]["expected_reward"]
        mean, low, high = y.mean(0), y.min(0), y.max(0)
        ax_reward.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
        ax_reward.plot(x, mean, color=color, label=label)
        plot_data["expected_reward"][method] = {
            "sequences_seen": x.tolist(),
            "mean": mean.tolist(),
            "min": low.tolist(),
            "max": high.tolist(),
        }
    optimum = task["optimal_expected_reward"]
    ax_reward.axhline(optimum, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax_reward.annotate(
        f"closed-form optimum: {optimum:.2f}",
        xy=(0, optimum),
        xytext=(4, 5),
        textcoords="offset points",
        color=TEXT_PRIMARY,
        fontsize=9,
    )
    ax_reward.set_title("REINFORCE: expected reward by baseline")
    ax_reward.set_xlabel(
        "sequences sampled from the policy\n"
        f"(line: mean of {n_seeds} seeds, band: lowest to highest)"
    )
    ax_reward.set_ylabel("exact expected reward")
    ax_reward.set_ylim(top=optimum + 0.45)
    ax_reward.legend(loc="lower right")

    # Panel 2: gradient variance, grouped by where in training it was measured.
    steps = sorted({g["at_step"] for g in grad_variance})
    width = 0.26
    for i, (method, (label, color)) in enumerate(BASELINES.items()):
        name = method.removeprefix("reinforce_")
        values = [
            next(
                g["variance"] for g in grad_variance if g["at_step"] == s and g["baseline"] == name
            )
            for s in steps
        ]
        positions = np.arange(len(steps)) + (i - 1) * width
        ax_var.bar(positions, values, width=width - 0.03, color=color, label=label)
        for xpos, value in zip(positions, values, strict=True):
            ax_var.annotate(
                f"{value:.2g}",
                xy=(xpos, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=90,
                color=TEXT_PRIMARY,
                fontsize=8,
            )
    ax_var.set_yscale("log")
    ax_var.set_xticks(np.arange(len(steps)), [f"policy at step {s}" for s in steps])
    ax_var.set_title("Spread of the gradient estimate (log scale)")
    ax_var.set_ylabel("trace of covariance across batches")
    ax_var.grid(axis="x", visible=False)
    low, high = ax_var.get_ylim()
    ax_var.set_ylim(low / 3, high * 300)  # headroom for the value labels and the legend
    ax_var.legend(loc="upper center", ncols=3, fontsize=8, columnspacing=1.0, handlelength=1.2)

    # Panel 3: distance to the closed-form optimum, REINFORCE against DPO.
    for method, (label, color) in DISTANCE_METHODS.items():
        x, y = curves[method]["x"], curves[method]["kl_to_opt"]
        mean, low, high = y.mean(0), y.min(0), y.max(0)
        ax_dist.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
        ax_dist.plot(x, mean, color=color, label=label)
        ax_dist.annotate(
            label.split(",")[0] if method.startswith("reinforce") else label.split(", ")[1],
            xy=(x[-1], mean[-1]),
            xytext=(5, {"dpo_hard": 7, "dpo_bradley_terry": -7}.get(method, 0)),
            textcoords="offset points",
            va="center",
            color=TEXT_PRIMARY,
            fontsize=8,
        )
        plot_data["kl_to_opt"][method] = {
            "sequences_seen": x.tolist(),
            "mean": mean.tolist(),
            "min": low.tolist(),
            "max": high.tolist(),
        }
    ax_dist.set_yscale("log")
    ax_dist.set_title("Distance to the closed-form optimum (log scale)")
    ax_dist.set_xlabel("sequences seen (REINFORCE: sampled; DPO: in labelled pairs)")
    ax_dist.set_ylabel("KL(policy || optimum), exact")
    ax_dist.set_xlim(right=x[-1] * 1.32)
    ax_dist.legend(loc="center right", fontsize=8)

    try:
        shown = run_dir.resolve().relative_to(ROOT)
    except ValueError:
        shown = run_dir
    fig.suptitle(
        f"Toy RL task: {task['n_sequences']} possible sequences, "
        f"policy with {task['n_parameters']} parameters, exact optimum by enumeration",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.05,
        f"Source: {shown}/curves.csv, grad_variance.json and summary.json, "
        "drawn by scripts/plot_reinforce.py. A toy: nothing here is a result about "
        "language models.",
        color=TEXT_SECONDARY,
        fontsize=8,
    )
    fig.tight_layout(w_pad=2.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    (run_dir / "plot_data.json").write_text(json.dumps(plot_data, indent=2) + "\n")
    print(f"wrote {out_path} and {run_dir / 'plot_data.json'}")


if __name__ == "__main__":
    main()
