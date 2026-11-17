"""M7: DPO and KTO training curves on the same held-out Track B pairs.

Both loops score the same 300 validation pairs at the same steps, so their curves can be
put on one set of axes. Three panels, one measure each:

(a) preference accuracy: share of held-out pairs where the policy raised the good answer
    more (relative to the reference) than the bad one
(b) mean log-probability of the good (preferred) answer: the warning sign in the first
    DPO run, where it fell from -33 to -59 while accuracy was already 0.99
(c) the DPO loss of the held-out pairs at the run's beta. For DPO runs this is their own
    loss; KTO runs log it as `dpo_loss` (their training loss is the KTO loss, which starts
    at 0.5 instead of log 2 and is not comparable)

DPO runs are solid lines, KTO runs dashed. Colours match scripts/plot_trackb_variants.py.
Runs that have not finished are named in the footer.

Usage:
    python scripts/plot_kto_training.py
Writes:
    assets/m7_training_curves.png
    results/kto_training_curves.json (the plotted numbers)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# name -> (method, legend label, colour slot shared with plot_trackb_variants.ENTITIES)
RUNS = {
    "trackb_full": ("DPO", "DPO, beta 0.1", 1),
    "trackb_beta0.5": ("DPO", "DPO, beta 0.5", 2),
    "trackb_kto": ("KTO", "KTO, beta 0.1", 5),
    "trackb_kto_beta0.5": ("KTO", "KTO, beta 0.5", 6),
}


def read_curves(results: Path) -> tuple[dict[str, dict[str, list[float]]], list[str]]:
    """run -> {step, accuracy, chosen_logp, dpo_loss} from its eval.csv; plus what is missing."""
    curves, missing = {}, []
    for run, (method, _, _) in RUNS.items():
        path = results / run / "eval.csv"
        if not (results / run / "run_info.json").is_file() or not path.is_file():
            missing.append(run)
            continue
        with path.open() as f:
            rows = list(csv.DictReader(f))
        loss_key = "loss" if method == "DPO" else "dpo_loss"
        curves[run] = {
            "step": [int(r["step"]) for r in rows],
            "accuracy": [float(r["accuracy"]) for r in rows],
            "chosen_logp": [float(r["chosen_logp"]) for r in rows],
            "dpo_loss": [float(r[loss_key]) for r in rows],
        }
    return curves, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--out", default=str(ROOT / "assets" / "m7_training_curves.png"))
    parser.add_argument("--json-out", default=str(ROOT / "results" / "kto_training_curves.json"))
    args = parser.parse_args()
    curves, missing = read_curves(Path(args.results))
    if not curves:
        raise SystemExit("no finished run to plot")
    Path(args.json_out).write_text(
        json.dumps({"curves": curves, "missing": missing}, indent=2) + "\n"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tiny_rlhf.plotting import SERIES, TEXT_PRIMARY, TEXT_SECONDARY, apply_style

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    panels = (
        ("accuracy", "Preference accuracy, held-out pairs", "share of pairs ranked right"),
        ("chosen_logp", "Log-probability of the good answer", "mean log pi(y_w|x), nats"),
        ("dpo_loss", "DPO loss of the held-out pairs (at each run's beta)", "-log sigmoid(margin)"),
    )
    for ax, (key, title, ylabel) in zip(axes, panels, strict=True):
        for run, data in curves.items():
            method, label, slot = RUNS[run]
            ax.plot(
                data["step"],
                data[key],
                color=SERIES[slot],
                ls="-" if method == "DPO" else (0, (5, 3)),
                marker="o",
                ms=5,
                mec="white",
                mew=1,
                label=label,
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("optimizer step")
    axes[0].set_ylim(0, 1.05)
    log2 = math.log(2.0)
    axes[2].axhline(log2, color=TEXT_SECONDARY, lw=0.9, ls=(0, (4, 3)), zorder=1)
    axes[2].annotate(
        "log 2 (step 0)",
        xy=(1.0, log2),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        color=TEXT_SECONDARY,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.94))
    fig.suptitle(
        "DPO (solid) and KTO (dashed) on Track B: the same 3,000 training pairs, scored on the "
        "same 300 held-out pairs",
        x=0.01,
        ha="left",
        fontweight="bold",
        color=TEXT_PRIMARY,
    )
    footer = (
        "Source: results/<run>/eval.csv (KTO runs: the dpo_loss column). "
        "Drawn by scripts/plot_kto_training.py."
    )
    if missing:
        footer += " Not run yet: " + ", ".join(missing) + "."
    fig.text(0.01, 0.01, footer, fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.04, 1, 0.88), w_pad=2.5)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {args.json_out} and {out}")


if __name__ == "__main__":
    main()
