"""M8: GRPO training curves, one point per optimizer step, from results/<run>/train.csv.

Four panels:
(a) mean reward of the step's sampled answers (the checker reward, -1.25 to 1), with the
    share of answers the strict check judged correct and the share that followed the
    planted instruction, so a rising reward can be read as "more right" or "less obeying"
(b) share of prompts whose answers all got the same reward. Those groups have zero
    advantage and produce no gradient: high means the prompts are too easy or too hard
    for the current policy
(c) KL to the base model (TRL's token mean of the k3 estimate; absent when beta is 0)
(d) mean answer length in tokens, with the share that hit the length limit

Per-step values are noisy (4 prompts per step), so each panel also draws a moving average
over `--window` steps; the raw points stay visible.

Usage:
    python scripts/plot_grpo_training.py --run results/trackb_grpo
Writes:
    assets/m8_grpo_training.png
    results/grpo_training_summary.json (first and last window averages of every column)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRPO_SLOT = 7  # colour slot of trackb_grpo in scripts/plot_trackb_variants.py ENTITIES


def read_train_csv(path: Path) -> dict[str, list[float | None]]:
    """Columns of a GRPO train.csv as lists of floats (None where TRL logged nothing)."""
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    return {
        key: [float(r[key]) if r[key] not in ("", None) else None for r in rows] for key in rows[0]
    }


def moving_average(values: list[float | None], window: int) -> list[float | None]:
    """Trailing mean over the last `window` values that exist (None where none do)."""
    out: list[float | None] = []
    for i in range(len(values)):
        recent = [v for v in values[max(0, i - window + 1) : i + 1] if v is not None]
        out.append(sum(recent) / len(recent) if recent else None)
    return out


def window_summary(columns: dict[str, list[float | None]], window: int) -> dict[str, dict]:
    """Mean of each column over the first and the last `window` steps."""

    def mean(values: list[float | None]) -> float | None:
        present = [v for v in values if v is not None]
        return sum(present) / len(present) if present else None

    return {
        key: {"first": mean(values[:window]), "last": mean(values[-window:])}
        for key, values in columns.items()
        if key != "step"
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=str(ROOT / "results" / "trackb_grpo"))
    parser.add_argument("--out", default=str(ROOT / "assets" / "m8_grpo_training.png"))
    parser.add_argument("--json-out", default=str(ROOT / "results" / "grpo_training_summary.json"))
    parser.add_argument("--window", type=int, default=20, help="moving-average window, steps")
    args = parser.parse_args()

    run = Path(args.run)
    columns = read_train_csv(run / "train.csv")
    info = json.loads((run / "run_info.json").read_text())
    summary = {
        "run": run.name,
        "optimizer_steps": info["optimizer_steps"],
        "stopped_by_time_limit": info.get("stopped_by_time_limit"),
        "window": args.window,
        "columns": window_summary(columns, args.window),
        "source": f"results/{run.name}/train.csv",
    }
    Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tiny_rlhf.plotting import SERIES, TEXT_PRIMARY, TEXT_SECONDARY, apply_style

    apply_style()
    steps = columns["step"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
    color = SERIES[GRPO_SLOT]

    def draw(ax, key: str, line_color: str, label: str | None = None, raw: bool = True) -> None:
        values = columns.get(key) or []
        pts = [(s, v) for s, v in zip(steps, values, strict=True) if v is not None]
        if not pts:
            return
        if raw:
            ax.scatter(*zip(*pts, strict=True), s=8, color=line_color, alpha=0.3, lw=0)
        smooth = [
            (s, v)
            for s, v in zip(steps, moving_average(values, args.window), strict=True)
            if v is not None
        ]
        ax.plot(*zip(*smooth, strict=True), color=line_color, label=label)

    ax = axes[0]
    draw(ax, "reward_mean", color, "mean reward")
    draw(ax, "rate_task_correct_strict", SERIES[2], "share correct (strict)", raw=False)
    draw(ax, "rate_followed_injection", SERIES[3], "share obeying the planted text", raw=False)
    ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
    ax.set_title("(a) Reward of the sampled answers")
    ax.set_ylabel("reward / share of answers")
    ax.legend(loc="lower right", fontsize=8.5)

    ax = axes[1]
    draw(ax, "frac_groups_all_equal", color)
    ax.set_ylim(0, 1.02)
    ax.set_title("(b) Prompts whose answers all tie")
    ax.set_ylabel("share of groups (no gradient)")

    ax = axes[2]
    if any(v is not None for v in columns.get("kl", [])):
        draw(ax, "kl", color)
    else:
        ax.text(0.5, 0.5, "beta = 0: no KL term", ha="center", transform=ax.transAxes)
    ax.set_title("(c) KL to the base model")
    ax.set_ylabel("token mean of the k3 estimate (nats)")

    ax = axes[3]
    draw(ax, "completion_length", color, "mean length")
    ax.set_title("(d) Answer length")
    ax.set_ylabel("tokens")
    twin = ax.twinx()
    draw(twin, "truncated_share", SERIES[1], "share at the limit", raw=False)
    twin.set_ylim(0, 1.02)
    twin.set_ylabel("share that hit the length limit", color=SERIES[1])
    twin.grid(False)
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="upper right", fontsize=8.5)

    for ax in axes:
        ax.set_xlabel("optimizer step")
    stop = " (stopped by the time limit)" if info.get("stopped_by_time_limit") else ""
    fig.suptitle(
        f"GRPO on Track B ({run.name}): {info['optimizer_steps']} steps{stop}, "
        f"{info['prompts_per_step']} prompts x {info['trl_settings']['num_generations']} "
        f"answers per step, beta {info['trl_settings']['beta']}",
        x=0.01,
        ha="left",
        fontweight="bold",
        color=TEXT_PRIMARY,
    )
    fig.text(
        0.01,
        0.01,
        f"Source: results/{run.name}/train.csv (TRL's logs and the reward's per-check shares). "
        f"Points: each step; lines: trailing mean over {args.window} steps. "
        "Drawn by scripts/plot_grpo_training.py.",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92), w_pad=2.5)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {args.json_out} and {out}")


if __name__ == "__main__":
    main()
