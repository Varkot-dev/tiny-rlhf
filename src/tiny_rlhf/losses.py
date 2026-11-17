"""The three building blocks of the project, in plain PyTorch.

sequence_logprob    log pi(y | x): the log-probability a causal LM gives a response
bradley_terry_loss  the reward-model loss on (preferred, other) pairs
dpo_loss            the DPO loss, which is Bradley-Terry with the reward
                    r(x, y) = beta * (log pi(y|x) - log pi_ref(y|x))

Conventions used everywhere in this repo
- "chosen" is the preferred response y_w, "rejected" is the other response y_l.
- All log-probabilities are natural logs, summed over response tokens only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


def sequence_logprob(logits: Tensor, input_ids: Tensor, response_mask: Tensor) -> Tensor:
    """Sum of log-probabilities of the response tokens, one number per sequence.

    A causal language model reads tokens 0..t and its output at position t is a
    distribution over token t+1. So the score for the token sitting at position t
    comes from the logits at position t-1. This function does that one-step shift
    internally; callers pass logits and input_ids exactly as the model saw them.

    Args:
        logits: [batch, seq_len, vocab] raw model outputs for the full sequence
            (prompt followed by response, then padding).
        input_ids: [batch, seq_len] the token ids that were fed to the model.
        response_mask: [batch, seq_len] true (or 1) at positions whose token is
            part of the response, false (or 0) for prompt and padding positions.

    Returns:
        [batch] tensor: sum over response positions t of
        log softmax(logits[:, t-1])[input_ids[:, t]].

    Raises:
        ValueError: on shape mismatch, or if position 0 is marked as response.
            The first token has no context, so the model never predicts it.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be [batch, seq_len, vocab], got {tuple(logits.shape)}")
    if input_ids.shape != logits.shape[:2] or response_mask.shape != input_ids.shape:
        raise ValueError(
            "shape mismatch: logits "
            f"{tuple(logits.shape)}, input_ids {tuple(input_ids.shape)}, "
            f"response_mask {tuple(response_mask.shape)}"
        )
    mask = response_mask.bool()
    if mask[:, 0].any():
        raise ValueError(
            "response_mask marks position 0, but a causal LM has no prediction for the "
            "first token. The response must come after at least one prompt token."
        )

    # logits[:, t-1] predicts input_ids[:, t]: drop the last logit and the first token.
    pred_logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    target_mask = mask[:, 1:]

    # Keep only the response positions before doing anything expensive. A full
    # log_softmax over [batch, seq_len, vocab] in float32 is several GB for a 150k
    # vocabulary; this touches the same numbers and never builds that tensor. Prompt
    # and padding positions are not merely zeroed, they are never read.
    rows = pred_logits[target_mask]  # [n_response_tokens, vocab]
    row_targets = targets[target_mask]  # [n_response_tokens]
    # Do the arithmetic in float32 when the logits arrive in half precision (float64
    # is kept). This cannot undo rounding already present in the logits: under fp16
    # autocast a summed log-probability carries noise of a few hundredths of a nat.
    if rows.dtype in (torch.float16, torch.bfloat16):
        rows = rows.float()
    # log softmax(z)[k] = z[k] - log(sum_j exp(z[j]))
    token_logps = rows.gather(dim=-1, index=row_targets.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps - torch.logsumexp(rows, dim=-1)

    # Add each token's log-probability to the total of the sequence it belongs to.
    sequence_index = target_mask.nonzero(as_tuple=True)[0]
    totals = torch.zeros(logits.shape[0], dtype=token_logps.dtype, device=logits.device)
    return totals.index_add(0, sequence_index, token_logps)


def bradley_terry_loss(reward_chosen: Tensor, reward_rejected: Tensor) -> Tensor:
    """Negative log-likelihood of the preferences under the Bradley-Terry model.

    Model: P(chosen beats rejected) = sigmoid(r_chosen - r_rejected).
    Loss:  -log sigmoid(r_chosen - r_rejected), averaged over the batch.

    Only the difference of the two rewards enters, so adding any constant (or any
    function of the prompt alone) to both rewards leaves the loss unchanged.

    Args:
        reward_chosen: [batch] scalar reward for each preferred response.
        reward_rejected: [batch] scalar reward for each other response.

    Returns:
        Scalar mean loss.
    """
    if reward_chosen.shape != reward_rejected.shape:
        raise ValueError(
            f"shape mismatch: {tuple(reward_chosen.shape)} vs {tuple(reward_rejected.shape)}"
        )
    # logsigmoid is stable for large |x|; log(sigmoid(x)) underflows to -inf.
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()


@dataclass(frozen=True)
class DPOLossOutput:
    """DPO loss plus the quantities worth logging every step.

    loss             scalar, has gradients
    chosen_rewards   [batch] beta * (log pi(y_w|x) - log pi_ref(y_w|x)), detached
    rejected_rewards [batch] beta * (log pi(y_l|x) - log pi_ref(y_l|x)), detached
    reward_margin    [batch] chosen_rewards - rejected_rewards, detached.
                     This is the argument of the sigmoid.
    accuracy         scalar, fraction of pairs with reward_margin > 0, detached.
                     Ties count as wrong, so it is 0 at step 0 where pi equals pi_ref.
    """

    loss: Tensor
    chosen_rewards: Tensor
    rejected_rewards: Tensor
    reward_margin: Tensor
    accuracy: Tensor


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    ref_chosen_logps: Tensor,
    ref_rejected_logps: Tensor,
    beta: float = 0.1,
) -> DPOLossOutput:
    """The DPO loss (Rafailov et al., 2023), sigmoid form.

    loss = -log sigmoid( beta * [ (log pi(y_w|x) - log pi_ref(y_w|x))
                                - (log pi(y_l|x) - log pi_ref(y_l|x)) ] )

    This is exactly bradley_terry_loss with the implicit reward
    r(x, y) = beta * (log pi(y|x) - log pi_ref(y|x)).

    Args:
        policy_chosen_logps: [batch] log pi(y_w|x) from the model being trained.
        policy_rejected_logps: [batch] log pi(y_l|x) from the model being trained.
        ref_chosen_logps: [batch] log pi_ref(y_w|x) from the frozen reference.
        ref_rejected_logps: [batch] log pi_ref(y_l|x) from the frozen reference.
        beta: strength of the pull toward the reference. Must be positive.

    Returns:
        DPOLossOutput. Gradients flow only through `loss`, and only into the policy
        log-probabilities: the reference terms are detached here as a safety net.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    shapes = {
        tuple(t.shape)
        for t in (policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps)
    }
    if len(shapes) != 1:
        raise ValueError(f"all four log-probability tensors must share a shape, got {shapes}")

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps.detach())
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps.detach())
    loss = bradley_terry_loss(chosen_rewards, rejected_rewards)

    margin = (chosen_rewards - rejected_rewards).detach()
    return DPOLossOutput(
        loss=loss,
        chosen_rewards=chosen_rewards.detach(),
        rejected_rewards=rejected_rewards.detach(),
        reward_margin=margin,
        accuracy=(margin > 0).float().mean(),
    )
