"""M1 figure: how one DPO training run went.

Reads a run directory written by the training loop and draws four panels that share
the optimizer-step axis:
    1. loss, with the log 2 starting value marked
    2. reward margin
    3. preference accuracy
    4. mean log-probability of the chosen and the rejected responses on held-out pairs

Panels 1 to 3 show per-step training values (thin), their rolling mean, and the
held-out evaluation values. Panel 4 is the check from docs/PLAN.md section 2: DPO can
widen the margin while making both responses less likely.

Usage:
    python scripts/plot_training.py --run results/<run_name>
    python scripts/plot_training.py --run results/<run_name> --out assets/some_name.png
Reads:
    <run>/train.csv, <run>/eval.csv, <run>/config.yaml
Writes:
    assets/m1_training_<run_name>.png unless --out is given
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb

from tiny_rlhf.plotting import SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, apply_style

ROOT = Path(__file__).resolve().parents[1]
TRAIN_COLOR, EVAL_COLOR = SERIES[0], SERIES[1]
CHOSEN_COLOR, REJECTED_COLOR = SERIES[0], SERIES[1]


def read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    """Read a numeric CSV into one float array per column. Missing file gives no columns."""
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out: dict[str, np.ndarray] = {}
    for key in rows[0]:
        values = []
        for row in rows:
            try:
                values.append(float(row[key]))
            except (TypeError, ValueError):
                values.append(math.nan)
        out[key] = np.array(values, dtype=float)
    return out


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over up to `window` points, so the first points are still defined."""
    if len(values) == 0 or window <= 1:
        return values.copy()
    padded = np.concatenate([[0.0], np.cumsum(np.nan_to_num(values))])
    idx = np.arange(1, len(values) + 1)
    start = np.maximum(idx - window, 0)
    return (padded[idx] - padded[start]) / (idx - start)


def lighten(color: str, amount: float = 0.6) -> tuple[float, float, float]:
    """Mix a colour toward the chart surface, for the raw per-step trace."""
    rgb, surface = np.array(to_rgb(color)), np.array(to_rgb(SURFACE))
    return tuple(rgb + (surface - rgb) * amount)


def plot_metric(
    ax: Axes,
    train: dict[str, np.ndarray],
    evals: dict[str, np.ndarray],
    key: str,
    window: int,
) -> None:
    """Per-step training values, their rolling mean, and held-out values for one metric."""
    if key in train and len(train[key]):
        steps, values = train["step"], train[key]
        ax.plot(steps, values, color=lighten(TRAIN_COLOR), lw=0.8, label="train, each step")
        if len(values) > 1 and window > 1:
            ax.plot(
                steps,
                rolling_mean(values, window),
                color=TRAIN_COLOR,
                lw=2,
                label=f"train, mean of last {window} steps",
            )
    if key in evals and len(evals[key]):
        ax.plot(
            evals["step"],
            evals[key],
            color=EVAL_COLOR,
            lw=2,
            marker="o",
            ms=6,
            mec=SURFACE,
            mew=1.5,
            label="held-out pairs",
            zorder=4,
        )


def describe_run(config: dict, train: dict[str, np.ndarray], evals: dict[str, np.ndarray]) -> str:
    """One title line built from the saved config, with fallbacks from the CSVs."""
    parts = []
    if config.get("model_name"):
        parts.append(str(config["model_name"]))
    if config.get("beta") is not None:
        parts.append(f"beta = {config['beta']}")
    if config.get("n_train") is not None:
        parts.append(f"{config['n_train']} training pairs")
    n_eval = config.get("n_eval")
    if n_eval is None and "n_pairs" in evals and len(evals["n_pairs"]):
        n_eval = int(evals["n_pairs"][0])
    if n_eval is not None:
        parts.append(f"{n_eval} held-out pairs")
    if "step" in train and len(train["step"]):
        parts.append(f"{int(train['step'][-1])} optimizer steps")
    return "DPO training run: " + ", ".join(parts) if parts else "DPO training run"


def make_figure(run_dir: Path, out_path: Path) -> None:
    train = read_csv_columns(run_dir / "train.csv")
    evals = read_csv_columns(run_dir / "eval.csv")
    if not train and not evals:
        raise SystemExit(f"no train.csv or eval.csv with rows found in {run_dir}")
    config_path = run_dir / "config.yaml"
    config = (yaml.safe_load(config_path.read_text()) or {}) if config_path.exists() else {}

    n_steps = len(train.get("step", []))
    window = max(1, min(25, n_steps // 5)) if n_steps else 1

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
    ax_loss, ax_margin, ax_acc, ax_logp = axes.flat

    plot_metric(ax_loss, train, evals, "loss", window)
    log2 = math.log(2.0)
    ax_loss.axhline(log2, color=TEXT_SECONDARY, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax_loss.annotate(
        "log 2",
        xy=(1.0, log2),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        color=TEXT_SECONDARY,
        fontsize=9,
    )
    ax_loss.set_title("Loss (starts at log 2 = 0.693)")
    ax_loss.set_ylabel("DPO loss")

    plot_metric(ax_margin, train, evals, "reward_margin", window)
    ax_margin.set_title("Reward margin")
    ax_margin.set_ylabel("chosen reward - rejected reward")

    plot_metric(ax_acc, train, evals, "accuracy", window)
    ax_acc.set_ylim(-0.02, 1.02)
    ax_acc.set_title("Preference accuracy")
    ax_acc.set_ylabel("share of pairs with margin > 0")

    for key, color, name in (
        ("chosen_logp", CHOSEN_COLOR, "chosen"),
        ("rejected_logp", REJECTED_COLOR, "rejected"),
    ):
        if key in evals and len(evals[key]):
            ax_logp.plot(
                evals["step"],
                evals[key],
                color=color,
                lw=2,
                marker="o",
                ms=6,
                mec=SURFACE,
                mew=1.5,
                label=f"{name} response",
            )
    labels = [
        (evals[key][-1], name)
        for key, name in (("chosen_logp", "chosen"), ("rejected_logp", "rejected"))
        if key in evals and len(evals[key])
    ]
    if labels:
        # Direct labels to the right of the last point, nudged apart if the two nearly coincide.
        lo, hi = ax_logp.get_ylim()
        close = len(labels) == 2 and abs(labels[0][0] - labels[1][0]) < 0.06 * (hi - lo)
        top = max(labels)
        for value, name in labels:
            nudge = (6 if (value, name) == top else -6) if close else 0
            ax_logp.annotate(
                name,
                xy=(evals["step"][-1], value),
                xytext=(9, nudge),
                textcoords="offset points",
                ha="left",
                va="center",
                color=TEXT_PRIMARY,
                fontsize=9,
                annotation_clip=False,
            )
    ax_logp.set_title("Mean log-probability on held-out pairs")
    ax_logp.set_ylabel("log pi(y|x), summed over response tokens")

    for ax in (ax_loss, ax_margin, ax_acc, ax_logp):
        ax.margins(x=0.03)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=8.5)
    for ax in (ax_acc, ax_logp):
        ax.set_xlabel("optimizer step")
    all_steps = np.concatenate([train.get("step", np.array([])), evals.get("step", np.array([]))])
    if len(all_steps) and all_steps.max() == all_steps.min():
        ax_loss.set_xlim(all_steps.min() - 1, all_steps.max() + 1)

    fig.suptitle(describe_run(config, train, evals), x=0.012, ha="left", fontsize=12, weight="bold")
    try:
        shown = run_dir.resolve().relative_to(ROOT)
    except ValueError:
        shown = run_dir
    fig.text(
        0.012,
        0.004,
        f"Source: {shown}/train.csv and eval.csv, drawn by scripts/plot_training.py",
        color=TEXT_SECONDARY,
        fontsize=8,
        ha="left",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97), w_pad=2.5, h_pad=1.8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="run directory under results/")
    parser.add_argument("--out", type=Path, default=None, help="output PNG path")
    args = parser.parse_args()
    run_dir: Path = args.run
    if not run_dir.is_dir():
        raise SystemExit(f"run directory not found: {run_dir}")
    out = args.out or ROOT / "assets" / f"m1_training_{run_dir.resolve().name}.png"
    make_figure(run_dir, out)


if __name__ == "__main__":
    main()
