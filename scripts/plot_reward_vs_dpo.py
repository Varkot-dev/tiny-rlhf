"""M3 figure: does the explicit reward model rank responses the way DPO's implicit reward does?

Joins the per-pair held-out rows of a DPO run and a reward-model run on pair_index.
Both runs must have scored the same held-out pairs in the same order (same dataset,
n_eval, seed, max_length and tokenizer); the script refuses to continue if the pair
counts differ or if both runs recorded a fingerprint of their pairs and the two differ.

    explicit reward  r(x, y)                        from the reward model
    implicit reward  beta * (log pi(y|x) - log pi_ref(y|x))   from the DPO run

The two rewards live on different scales, so compare rankings and signs, not values.

Usage:
    python scripts/plot_reward_vs_dpo.py --dpo results/dpo_tiny --reward results/reward_tiny
Writes:
    results/<reward run>/reward_vs_dpo.json
    assets/m3_reward_vs_dpo_<reward run>.png   (or --out)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tiny_rlhf.plotting import SERIES, TEXT_PRIMARY, TEXT_SECONDARY, apply_style

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0


def read_pairs(run_dir: Path) -> dict[str, np.ndarray]:
    path = run_dir / "eval_pairs_final.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}: finish the run before plotting")
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    columns = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    order = np.argsort(columns["pair_index"])
    return {k: v[order] for k, v in columns.items()}


def read_info(run_dir: Path) -> dict:
    path = run_dir / "run_info.json"
    return json.loads(path.read_text()) if path.exists() else {}


def ranks(x: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, with tied values sharing the mean of their ranks."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    out = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        out[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman rank correlation: the Pearson correlation of the two rank vectors."""
    ra, rb = ranks(a), ranks(b)
    if len(a) < 2 or ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def bootstrap_mean_ci(values: np.ndarray) -> list[float]:
    """Percentile bootstrap 95% interval for a mean, resampling pairs with replacement."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    index = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[index].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dpo", required=True, type=Path, help="DPO run directory")
    parser.add_argument("--reward", required=True, type=Path, help="reward-model run directory")
    parser.add_argument("--out", type=Path, help="output PNG path")
    args = parser.parse_args()

    dpo, reward = read_pairs(args.dpo), read_pairs(args.reward)
    dpo_info, reward_info = read_info(args.dpo), read_info(args.reward)

    if len(dpo["pair_index"]) != len(reward["pair_index"]) or not np.array_equal(
        dpo["pair_index"], reward["pair_index"]
    ):
        raise SystemExit(
            f"the two runs scored different pairs: {len(dpo['pair_index'])} rows in the DPO run, "
            f"{len(reward['pair_index'])} in the reward run. Use the same n_eval, seed and "
            "max_length for both."
        )
    hash_dpo, hash_reward = dpo_info.get("eval_pairs_sha256"), reward_info.get("eval_pairs_sha256")
    if hash_dpo and hash_reward and hash_dpo != hash_reward:
        raise SystemExit(
            "the two runs recorded different fingerprints for their held-out pairs, so "
            "pair_index does not refer to the same pairs. Refusing to compare."
        )
    alignment = "verified by fingerprint" if hash_dpo and hash_reward else "not verified"
    if alignment == "not verified":
        print(
            "warning: at least one run has no eval_pairs_sha256 in run_info.json; the pairs "
            "are assumed to match because the counts match"
        )

    explicit_margin = reward["reward_margin"]
    implicit_margin = dpo["reward_margin"]
    explicit_right = explicit_margin > 0  # ties count as wrong, as everywhere in this repo
    implicit_right = implicit_margin > 0
    agree = explicit_right == implicit_right
    n = len(agree)
    quadrants = {
        "both_right": int((explicit_right & implicit_right).sum()),
        "both_wrong": int((~explicit_right & ~implicit_right).sum()),
        "only_reward_model_right": int((explicit_right & ~implicit_right).sum()),
        "only_dpo_right": int((~explicit_right & implicit_right).sum()),
    }
    explicit_scores = np.concatenate([reward["chosen_reward"], reward["rejected_reward"]])
    implicit_scores = np.concatenate([dpo["chosen_reward"], dpo["rejected_reward"]])

    summary = {
        "dpo_run": relative(args.dpo),
        "reward_run": relative(args.reward),
        "n_pairs": n,
        "pair_alignment": alignment,
        "reward_model_accuracy": float(explicit_right.mean()),
        "reward_model_accuracy_ci95": bootstrap_mean_ci(explicit_right.astype(float)),
        "dpo_accuracy": float(implicit_right.mean()),
        "dpo_accuracy_ci95": bootstrap_mean_ci(implicit_right.astype(float)),
        "agreement_rate": float(agree.mean()),
        "agreement_rate_ci95": bootstrap_mean_ci(agree.astype(float)),
        "quadrant_counts": quadrants,
        "spearman_margins": spearman(explicit_margin, implicit_margin),
        "spearman_response_scores": spearman(explicit_scores, implicit_scores),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "tie_rule": "a margin of exactly 0 counts as wrong",
    }
    summary_path = args.reward / "reward_vs_dpo.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    apply_style()
    fig, (ax_margin, ax_score) = plt.subplots(1, 2, figsize=(11.5, 5.2))

    # Panel (a): one dot per held-out pair.
    ax_margin.axhline(0, color=TEXT_SECONDARY, lw=0.8, zorder=1)
    ax_margin.axvline(0, color=TEXT_SECONDARY, lw=0.8, zorder=1)
    ax_margin.scatter(
        explicit_margin, implicit_margin, s=26, color=SERIES[0], edgecolor="white", lw=0.6, zorder=3
    )
    corner_labels = (
        (0.98, 0.98, "right", "top", f"both right: {quadrants['both_right']}"),
        (0.02, 0.02, "left", "bottom", f"both wrong: {quadrants['both_wrong']}"),
        (
            0.98,
            0.02,
            "right",
            "bottom",
            f"only reward model right: {quadrants['only_reward_model_right']}",
        ),
        (0.02, 0.98, "left", "top", f"only DPO right: {quadrants['only_dpo_right']}"),
    )
    for x, y, ha, va, text in corner_labels:
        ax_margin.text(
            x, y, text, transform=ax_margin.transAxes, ha=ha, va=va, fontsize=9, color=TEXT_PRIMARY
        )
    # Symmetric limits keep the zero lines in the middle so the four quadrants read evenly.
    for values, setter in (
        (explicit_margin, ax_margin.set_xlim),
        (implicit_margin, ax_margin.set_ylim),
    ):
        reach = float(np.abs(values).max()) * 1.25 or 1.0
        setter(-reach, reach)
    ax_margin.set_title(f"Margin per held-out pair ({n} pairs)")
    ax_margin.set_xlabel("reward model: r(chosen) - r(rejected)")
    ax_margin.set_ylabel("DPO implicit reward: chosen - rejected")

    # Panel (b): one dot per held-out response.
    half = len(reward["chosen_reward"])
    for label, color, sl in (
        ("chosen response", SERIES[0], slice(0, half)),
        ("rejected response", SERIES[1], slice(half, None)),
    ):
        ax_score.scatter(
            explicit_scores[sl],
            implicit_scores[sl],
            s=26,
            color=color,
            edgecolor="white",
            lw=0.6,
            label=label,
            zorder=3,
        )
    ax_score.axhline(0, color=TEXT_SECONDARY, lw=0.8, zorder=1)
    ax_score.set_title(f"Score per held-out response ({2 * n} responses)")
    ax_score.set_xlabel("reward model score r(x, y)")
    ax_score.set_ylabel("DPO implicit reward beta * (log pi - log pi_ref)")
    ax_score.legend(loc="best")

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    low, high = summary["agreement_rate_ci95"]
    fig.suptitle(
        "Reward model vs DPO implicit reward: they prefer the same response on "
        f"{int(agree.sum())} of {n} pairs "
        f"({agree.mean():.2f}, 95% interval {low:.2f} to {high:.2f})",
        fontsize=11,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.text(
        0.01,
        0.005,
        f"Spearman of margins {fmt(summary['spearman_margins'])}, of response scores "
        f"{fmt(summary['spearman_response_scores'])}. Pair alignment {alignment}.\n"
        f"Source: {relative(args.dpo)} and\n{relative(args.reward)}, "
        "drawn by scripts/plot_reward_vs_dpo.py",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95), w_pad=3)

    out = args.out or ROOT / "assets" / f"m3_reward_vs_dpo_{args.reward.resolve().name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {relative(summary_path)} and {relative(out)}")


if __name__ == "__main__":
    main()
