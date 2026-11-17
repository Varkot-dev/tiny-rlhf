"""M2 figure: the hand-written DPO loop and TRL's DPOTrainer on the same run, overlaid.

Reads two run directories that were trained from the same config, one by
scripts/train_dpo.py and one by scripts/train_trl.py, and draws four panels that share
the optimizer-step axis:
    1. loss, with the log 2 starting value marked
    2. reward margin
    3. preference accuracy
    4. mean log-probability of the preferred response on held-out pairs

The hand-written run is a solid line and TRL is a dashed line on top of it, so where
the two agree the dashes sit on the solid line. Each panel states the largest absolute
difference between the two runs. The same numbers go to parity_summary.json.

Usage:
    python scripts/plot_parity.py --ours results/parity_tiny --trl results/parity_tiny_trl
Reads:
    <run>/train.csv, <run>/eval.csv, <run>/config.yaml, <run>/run_info.json from both
Writes:
    assets/m2_parity_<ours run name>.png unless --out is given
    <ours>/parity_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from tiny_rlhf.plotting import SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, apply_style

ROOT = Path(__file__).resolve().parents[1]
OURS_COLOR, TRL_COLOR = SERIES[0], SERIES[1]
METRICS = (
    "loss",
    "reward_margin",
    "accuracy",
    "chosen_logp",
    "rejected_logp",
    "chosen_reward",
    "rejected_reward",
)


def read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    """Read a numeric CSV into one float array per column. Blank cells become nan."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for key in rows[0]:
        values = []
        for row in rows:
            try:
                values.append(float(row[key]))
            except (TypeError, ValueError):
                values.append(math.nan)
        out[key] = np.array(values)
    return out


def align(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> tuple[dict, dict, bool]:
    """Keep only the rows whose step appears in both runs. Also says if any were dropped."""
    steps = np.intersect1d(a["step"], b["step"])
    same = len(steps) == len(a["step"]) == len(b["step"])
    pick_a, pick_b = np.isin(a["step"], steps), np.isin(b["step"], steps)
    return {k: v[pick_a] for k, v in a.items()}, {k: v[pick_b] for k, v in b.items()}, same


def differences(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Largest and mean absolute difference per metric, over aligned rows."""
    out = {}
    for key in METRICS:
        gap = np.abs(a[key] - b[key])
        out[key] = {
            "max_abs_diff": float(np.nanmax(gap)),
            "mean_abs_diff": float(np.nanmean(gap)),
            "at_step": int(a["step"][int(np.nanargmax(gap))]),
        }
    return out


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def shown_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def summarize(ours_dir: Path, trl_dir: Path) -> tuple[dict, dict]:
    """Compare the two runs. Returns (summary for the JSON file, aligned data to plot)."""
    data = {}
    summary: dict = {"ours": shown_path(ours_dir), "trl": shown_path(trl_dir)}
    for name in ("train", "eval"):
        ours, trl, same_steps = align(
            read_csv_columns(ours_dir / f"{name}.csv"), read_csv_columns(trl_dir / f"{name}.csv")
        )
        data[name] = (ours, trl)
        summary[name] = {
            "rows_compared": int(len(ours["step"])),
            "both_runs_have_the_same_steps": same_steps,
            "metrics": differences(ours, trl),
        }
    ours_info, trl_info = (
        load_json(ours_dir / "run_info.json"),
        load_json(trl_dir / "run_info.json"),
    )
    summary["same_train_order"] = trl_info.get("train_order") == "sequential"
    summary["tokenization_matches"] = trl_info.get("tokenization_matches_handwritten")
    summary["eval_batches_equal_size"] = trl_info.get("eval_batches_equal_size")
    for key in ("device", "amp_fp16", "n_train_pairs", "n_eval_pairs", "wall_time_s", "git_sha"):
        summary[key] = {"ours": ours_info.get(key), "trl": trl_info.get(key)}
    summary["versions"] = trl_info.get("versions")
    return summary, data


def difference_note(ax: Axes, summary: dict, key: str, kinds: tuple[str, ...]) -> None:
    """Above the panel, right-aligned: the largest gap between the two runs."""
    labels = {"train": "train", "eval": "held-out"}
    parts = [f"{labels[k]} {summary[k]['metrics'][key]['max_abs_diff']:.1e}" for k in kinds]
    ax.text(
        1.0,
        1.025,
        "max gap: " + ", ".join(parts),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=TEXT_SECONDARY,
    )


def plot_pair(ax: Axes, data: dict, key: str, train: bool = True, heldout: bool = True) -> None:
    """Hand-written solid underneath, TRL dashed on top; held-out values as markers."""
    if train:
        ours, trl = data["train"]
        ax.plot(ours["step"], ours[key], color=OURS_COLOR, lw=2.4, zorder=2)
        ax.plot(trl["step"], trl[key], color=TRL_COLOR, lw=1.3, ls=(0, (3, 2)), zorder=3)
    if heldout:
        ours, trl = data["eval"]
        if not train:
            ax.plot(ours["step"], ours[key], color=OURS_COLOR, lw=2.4, zorder=2)
            ax.plot(trl["step"], trl[key], color=TRL_COLOR, lw=1.3, ls=(0, (3, 2)), zorder=3)
        ax.plot(ours["step"], ours[key], "o", ms=9, color=OURS_COLOR, mec=SURFACE, mew=2, zorder=4)
        ax.plot(trl["step"], trl[key], "x", ms=6, color=TRL_COLOR, mew=1.8, zorder=5)


def make_figure(ours_dir: Path, trl_dir: Path, out_path: Path) -> dict:
    summary, data = summarize(ours_dir, trl_dir)
    config = yaml.safe_load((ours_dir / "config.yaml").read_text()) or {}

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True)
    (ax_loss, ax_margin), (ax_acc, ax_logp) = axes

    plot_pair(ax_loss, data, "loss")
    ax_loss.axhline(math.log(2.0), color=TEXT_SECONDARY, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax_loss.annotate(
        "log 2",
        xy=(1.0, math.log(2.0)),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=TEXT_SECONDARY,
    )
    ax_loss.set_title("Loss")
    ax_loss.set_ylabel("DPO loss")
    difference_note(ax_loss, summary, "loss", ("train", "eval"))

    plot_pair(ax_margin, data, "reward_margin")
    ax_margin.set_title("Reward margin")
    ax_margin.set_ylabel("chosen reward - rejected reward")
    difference_note(ax_margin, summary, "reward_margin", ("train", "eval"))

    plot_pair(ax_acc, data, "accuracy")
    ax_acc.set_ylim(-0.03, 1.03)
    ax_acc.set_title("Preference accuracy")
    ax_acc.set_ylabel("share of pairs with margin > 0")
    ax_acc.set_xlabel("optimizer step")
    difference_note(ax_acc, summary, "accuracy", ("train", "eval"))

    plot_pair(ax_logp, data, "chosen_logp", train=False)
    ax_logp.set_title("Preferred log-prob, held-out")
    ax_logp.set_ylabel("mean log pi(y_w|x)")
    ax_logp.set_xlabel("optimizer step")
    ax_logp.ticklabel_format(axis="y", useOffset=False)
    ax_logp.xaxis.set_major_locator(MaxNLocator(integer=True))
    difference_note(ax_logp, summary, "chosen_logp", ("eval",))

    handles = [
        Line2D([], [], color=OURS_COLOR, lw=2.4, label="hand-written, each training step"),
        Line2D([], [], color=TRL_COLOR, lw=1.3, ls=(0, (3, 2)), label="TRL, each training step"),
        Line2D(
            [], [], color=OURS_COLOR, marker="o", ms=9, mec=SURFACE, mew=2, lw=0,
            label="hand-written, held-out pairs",
        ),
        Line2D(
            [], [], color=TRL_COLOR, marker="x", ms=6, mew=1.8, lw=0, label="TRL, held-out pairs"
        ),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.945), fontsize=9)

    pairs = summary["n_train_pairs"]["ours"] or config.get("n_train")
    steps = summary["train"]["rows_compared"]
    fig.suptitle(
        f"Hand-written DPO vs TRL DPOTrainer: {config.get('model_name', 'model')}, "
        f"beta = {config.get('beta')}, {pairs} training pairs, {steps} optimizer steps",
        fontsize=12,
        fontweight="bold",
        color=TEXT_PRIMARY,
        x=0.01,
        ha="left",
    )
    order = "same batch order" if summary["same_train_order"] else "different batch order"
    fig.text(
        0.01,
        0.005,
        f"Source: {summary['ours']} (hand-written) and {summary['trl']} (TRL), {order}.\n"
        "Max gap = largest absolute difference between the two runs at the same step. "
        "Drawn by scripts/plot_parity.py.",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.91), h_pad=2.5, w_pad=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ours", required=True, type=Path, help="run dir from train_dpo.py")
    parser.add_argument("--trl", required=True, type=Path, help="run dir from train_trl.py")
    parser.add_argument(
        "--out", type=Path, help="figure path (default: assets/m2_parity_<run>.png)"
    )
    args = parser.parse_args()

    out = args.out or ROOT / "assets" / f"m2_parity_{args.ours.resolve().name}.png"
    summary = make_figure(args.ours, args.trl, out)
    summary_path = args.ours / "parity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"wrote {shown_path(out)} and {shown_path(summary_path)}")
    for name in ("train", "eval"):
        print(f"{name}: {summary[name]['rows_compared']} rows compared")
        for key, d in summary[name]["metrics"].items():
            print(f"  {key:16s} max |diff| {d['max_abs_diff']:.3e}   mean {d['mean_abs_diff']:.3e}")
    if not summary["same_train_order"]:
        print(
            "note: the two runs did not use the same batch order, so step-level gaps are expected"
        )


if __name__ == "__main__":
    main()
