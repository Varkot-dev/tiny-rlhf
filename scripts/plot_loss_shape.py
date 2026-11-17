"""M0 figure: the shape of the DPO loss and of the push it gives each pair.

Evaluates tiny_rlhf.dpo_loss on a grid of log-ratio differences
    u = (log pi(y_w|x) - log pi_ref(y_w|x)) - (log pi(y_l|x) - log pi_ref(y_l|x))
for three values of beta. The gradient panel comes from autograd on that same
function, not from a formula typed in again here.

No model and no data are involved: this figure shows the loss function, not a result.

Usage:
    python scripts/plot_loss_shape.py
Writes:
    results/m0_loss_shape.csv
    assets/m0_loss_shape.png
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tiny_rlhf import dpo_loss
from tiny_rlhf.plotting import SERIES, TEXT_PRIMARY, TEXT_SECONDARY, apply_style

ROOT = Path(__file__).resolve().parents[1]
BETAS = (0.01, 0.1, 0.5)
U_MIN, U_MAX, N_POINTS = -40.0, 40.0, 321


def loss_and_push(u: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-point loss and -d loss / d log pi(y_w|x), for a grid of differences u."""
    policy_chosen = u.clone().requires_grad_(True)
    zeros = torch.zeros_like(u)
    losses = torch.stack(
        [
            dpo_loss(policy_chosen[i : i + 1], zeros[:1], zeros[:1], zeros[:1], beta=beta).loss
            for i in range(len(u))
        ]
    )
    (grad,) = torch.autograd.grad(losses.sum(), policy_chosen)
    return losses.detach(), -grad


def main() -> None:
    u = torch.linspace(U_MIN, U_MAX, N_POINTS, dtype=torch.float64)
    curves = {beta: loss_and_push(u, beta) for beta in BETAS}

    results_path = ROOT / "results" / "m0_loss_shape.csv"
    results_path.parent.mkdir(exist_ok=True)
    with results_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["beta", "u_nats", "loss", "push_on_chosen_logprob"])
        for beta, (loss, push) in curves.items():
            for ui, li, pi in zip(u.tolist(), loss.tolist(), push.tolist(), strict=True):
                writer.writerow([beta, f"{ui:.4f}", f"{li:.10f}", f"{pi:.10f}"])

    apply_style()
    fig, (ax_loss, ax_push) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for color, (beta, (loss, push)) in zip(SERIES, curves.items(), strict=False):
        ax_loss.plot(u, loss, color=color, label=f"beta = {beta}")
        ax_push.plot(u, push, color=color, label=f"beta = {beta}")
        ax_push.annotate(
            f"beta = {beta}",
            xy=(U_MIN, push[0].item()),
            xytext=(4, 5),
            textcoords="offset points",
            color=TEXT_PRIMARY,
            fontsize=9,
        )

    log2 = math.log(2.0)
    ax_loss.plot([0], [log2], "o", ms=8, color=TEXT_PRIMARY, mec="white", mew=2, zorder=5)
    ax_loss.annotate(
        "step 0: policy = reference\nloss = log 2 = 0.693 for every beta",
        xy=(0, log2),
        xytext=(4, 3.2),
        color=TEXT_PRIMARY,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": TEXT_SECONDARY, "lw": 0.8},
    )
    ax_loss.set_ylim(0, 6)
    ax_loss.set_title("DPO loss per pair")
    ax_loss.set_xlabel(
        "u: how much more the policy favours the preferred response\nthan the reference did (nats)"
    )
    ax_loss.set_ylabel("loss = -log sigmoid(beta * u)")
    ax_loss.legend(loc="upper right")

    ax_push.set_title("How hard a pair pushes on the preferred log-probability")
    ax_push.set_xlabel("u (nats)")
    ax_push.set_ylabel("-d loss / d log pi(y_w|x) = beta * sigmoid(-beta * u)")
    ax_push.set_ylim(0, 0.55)

    fig.tight_layout(w_pad=3)
    out = ROOT / "assets" / "m0_loss_shape.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    print(f"wrote {results_path.relative_to(ROOT)} and {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
