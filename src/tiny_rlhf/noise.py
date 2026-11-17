"""Label noise for the M6 ablation: flip a share of the training pairs, then train on them.

Two pieces:

flip_pairs        swap chosen and rejected for a seeded random subset of training pairs,
                  which simulates annotators who picked the wrong response
robust_dpo_loss   a DPO loss corrected for a known flip rate. It is what TRL's
                  `label_smoothing` option does when `loss_type="robust"` (trl 1.18.0;
                  the default sigmoid loss ignores `label_smoothing` entirely)

Held-out pairs are never flipped: the question is how training on noisy labels changes
behaviour on clean ones.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import torch.nn.functional as F
from torch import Tensor, nn

from tiny_rlhf.data import TokenizedPair
from tiny_rlhf.losses import DPOLossOutput


def flip_at(pairs: Sequence[TokenizedPair], indices: Sequence[int]) -> list[TokenizedPair]:
    """Copy of `pairs` with chosen and rejected swapped at the given positions."""
    flip = set(indices)
    if flip and (min(flip) < 0 or max(flip) >= len(pairs)):
        raise ValueError("flip index out of range")
    return [
        TokenizedPair(p.prompt_ids, p.rejected_ids, p.chosen_ids) if i in flip else p
        for i, p in enumerate(pairs)
    ]


def flip_pairs(
    pairs: Sequence[TokenizedPair], fraction: float, seed: int
) -> tuple[list[TokenizedPair], list[int]]:
    """Swap chosen and rejected in round(fraction * n) randomly picked pairs.

    Args:
        pairs: training pairs. The input list is not modified.
        fraction: share of pairs to flip, between 0 and 1.
        seed: seed for the choice of pairs, so a run can be repeated exactly.

    Returns:
        (new list of pairs in the same order, sorted indices of the flipped pairs).
        An exact count is flipped, not each pair independently with probability
        `fraction`, so a 10% run really has 10% wrong labels.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}")
    n_flip = round(fraction * len(pairs))
    indices = sorted(random.Random(seed).sample(range(len(pairs)), n_flip))
    return flip_at(pairs, indices), indices


def indices_fingerprint(indices: Sequence[int]) -> str:
    """sha256 of the flipped indices, short enough to record with a run."""
    return hashlib.sha256(json.dumps(list(indices)).encode()).hexdigest()


def robust_dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    ref_chosen_logps: Tensor,
    ref_rejected_logps: Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> DPOLossOutput:
    """DPO loss corrected for labels that are wrong with a known probability.

    Derivation. Write u for the usual bracket,
        u = (log pi(y_w|x) - log pi_ref(y_w|x)) - (log pi(y_l|x) - log pi_ref(y_l|x)),
    and L(u) = -log sigmoid(beta * u) for the ordinary DPO loss of a pair in the order
    we were given. If the pair had been recorded the other way round, its loss would be
    L(-u).

    Suppose each label is flipped independently with probability e < 1/2. We only see
    the noisy order, but we want the loss we would have had with the true order. Look
    for weights a, b such that the noisy loss a*L(u) + b*L(-u) has the right average:
        true order is the one we see, probability 1-e:   a*L(u)  + b*L(-u)
        true order is the opposite,   probability e:     a*L(-u) + b*L(u)   (u changes sign)
    The average over the noise, written for the true order with bracket u, is
        (1-e)*(a*L(u) + b*L(-u)) + e*(a*L(-u) + b*L(u)).
    For this to equal L(u) for every u we need
        (1-e)*a + e*b = 1     and     (1-e)*b + e*a = 0,
    which gives a = (1-e)/(1-2e) and b = -e/(1-2e). So

        loss = [ (1-e) * L(u) - e * L(-u) ] / (1 - 2e)

    With e = 0 this is the ordinary DPO loss. At u = 0 it is log 2 for every e, so the
    step-0 check still holds.

    Two things follow from the minus sign and are worth knowing before using it:
    - The loss is not bounded below. For large u it behaves like
      -e * beta * u / (1 - 2e), so it can go negative and never stops rewarding a
      wider margin, unlike ordinary DPO whose push fades to zero.
    - Gradients are scaled by 1/(1-2e): 1.25 times at e = 0.1, 2.5 times at e = 0.3.

    This is the "robust" loss of Chowdhury et al. 2024 ("Provably Robust DPO") as
    implemented in trl 1.18.0, where `label_smoothing` is e.

    Args:
        policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps:
            [batch] summed response log-probabilities, as for tiny_rlhf.losses.dpo_loss.
        beta: as in DPO.
        label_smoothing: the assumed flip probability e, in [0, 0.5).

    Returns:
        DPOLossOutput with the same logging fields as dpo_loss.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError(f"label_smoothing must be in [0, 0.5), got {label_smoothing}")

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps.detach())
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps.detach())
    scaled_u = chosen_rewards - rejected_rewards  # beta * u
    as_given = -F.logsigmoid(scaled_u)  # L(u)
    as_flipped = -F.logsigmoid(-scaled_u)  # L(-u)
    per_pair = ((1 - label_smoothing) * as_given - label_smoothing * as_flipped) / (
        1 - 2 * label_smoothing
    )

    margin = scaled_u.detach()
    return DPOLossOutput(
        loss=per_pair.mean(),
        chosen_rewards=chosen_rewards.detach(),
        rejected_rewards=rejected_rewards.detach(),
        reward_margin=margin,
        accuracy=(margin > 0).float().mean(),
    )


@contextmanager
def training_loss(loss_fn: Callable[..., DPOLossOutput]) -> Iterator[None]:
    """Make tiny_rlhf.train_dpo use `loss_fn` in place of dpo_loss inside the block.

    train_dpo.train has no loss argument, so the function it looks up is swapped for
    the duration of one run and put back afterwards. Both the training steps and the
    held-out evaluations inside the block use `loss_fn`.
    """
    from tiny_rlhf import train_dpo

    original = train_dpo.dpo_loss
    train_dpo.dpo_loss = loss_fn
    try:
        yield
    finally:
        train_dpo.dpo_loss = original


def train_with_label_noise(
    cfg: Any,
    policy: nn.Module,
    reference: nn.Module,
    train_pairs: list[TokenizedPair],
    eval_pairs: list[TokenizedPair],
    pad_token_id: int,
    out_dir: str | Path,
    noise_fraction: float,
    noise_seed: int,
    label_smoothing: float = 0.0,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Flip a share of the training pairs, train, and record what was done.

    With label_smoothing 0 the ordinary DPO loss is used. With label_smoothing > 0 the
    robust loss is used for training and for the `loss` column of both CSV logs, so
    loss values are comparable only between runs with the same label_smoothing. Reward
    margin, preference accuracy and log-probabilities do not depend on the loss and
    stay comparable across all runs.
    """
    from tiny_rlhf.train_dpo import train

    noisy_pairs, flipped = flip_pairs(train_pairs, noise_fraction, noise_seed)
    out_dir = Path(out_dir)
    if label_smoothing > 0:
        with training_loss(partial(robust_dpo_loss, label_smoothing=label_smoothing)):
            info = train(
                cfg,
                policy,
                reference,
                noisy_pairs,
                eval_pairs,
                pad_token_id,
                out_dir,
                checkpoint_dir,
            )
    else:
        info = train(
            cfg, policy, reference, noisy_pairs, eval_pairs, pad_token_id, out_dir, checkpoint_dir
        )

    info.update(
        {
            "label_noise": {
                "fraction_requested": noise_fraction,
                "n_flipped": len(flipped),
                "n_train_pairs": len(train_pairs),
                "fraction_flipped": len(flipped) / len(train_pairs),
                "seed": noise_seed,
                "flipped_indices_sha256": indices_fingerprint(flipped),
                "heldout_pairs_flipped": 0,
            },
            "loss_type": "robust" if label_smoothing > 0 else "sigmoid",
            "label_smoothing": label_smoothing,
        }
    )
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info
