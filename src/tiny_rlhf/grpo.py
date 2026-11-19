"""GRPO (group relative policy optimization), written by hand.

This module holds the two pieces of GRPO that are not plumbing: the group-relative
advantage and the clipped policy-gradient loss with a KL penalty. Generation, batching and
the optimizer come from TRL's GRPOTrainer (see tiny_rlhf.train_grpo); the tests in
tests/test_grpo.py check that these functions reproduce TRL 1.18's numbers and gradients.

The idea in one paragraph. For each prompt x, sample a group of G answers y_1 ... y_G from
the current policy and score each with a reward r_i (here the Track B checker, see
tiny_rlhf.grpo_reward). The advantage of answer i is how much better it did than its own
group:

    A_i = (r_i - mean(r_1 ... r_G)) / (std(r_1 ... r_G) + 1e-4)

Then raise the log-probability of every token of answer i in proportion to A_i. No value
network and no learned reward model: the group is the baseline.

Worked examples (G = 4; std is the sample standard deviation, divided by G - 1, as in TRL):

    rewards 1, 0, 0, 0     mean 0.25, std 0.5   advantages +1.4997, -0.4999, -0.4999, -0.4999
    rewards 1, 1, 1, 1     mean 1,    std 0     advantages 0, 0, 0, 0   (the group teaches nothing)
    rewards 1, -1, 0, 0    mean 0,    std 0.8165  advantages +1.2246, -1.2246, 0, 0

The first case is one success among four: the success is pushed up three times as hard as
each failure is pushed down, and the four advantages sum to zero. The second case is why
the share of all-equal groups is logged: those prompts are too easy (or too hard) for the
current policy and cost a generation without producing any gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

SCALE_MODES = ("group", "batch", "none")
LOSS_TYPES = ("grpo", "bnpo", "dr_grpo", "dapo")
STD_EPS = 1e-4  # TRL adds this to the standard deviation before dividing


def nanstd(x: Tensor, dim: int | None = None) -> Tensor:
    """Sample standard deviation (divided by count - 1) that ignores NaN entries.

    Same definition as TRL's trl.trainer.utils.nanstd: NaN when fewer than two values are
    present along `dim`.
    """
    mean = torch.nanmean(x, dim=dim, keepdim=True)
    var = torch.nanmean((x - mean) ** 2, dim=dim, keepdim=True)
    count = torch.sum(~torch.isnan(x), dim=dim, keepdim=True)
    correction = torch.where(
        count > 1, count / (count - 1), torch.full_like(count, float("nan"), dtype=x.dtype)
    )
    out = torch.sqrt(var * correction)
    return out.squeeze(dim) if dim is not None else out.reshape(())


def group_advantages(
    rewards: Tensor, group_size: int, scale: str = "group"
) -> tuple[Tensor, Tensor]:
    """Group-relative advantages, exactly as TRL 1.18's GRPOTrainer computes them.

    Args:
        rewards: [N] one reward per completion, N = number of prompts * group_size, with the
            group_size completions of each prompt next to each other (TRL's order). A NaN
            reward marks a completion no reward function could score.
        group_size: completions per prompt (TRL's num_generations).
        scale: what the centred reward is divided by.
            "group": the standard deviation of its own group (the GRPO paper's choice).
            "batch": the standard deviation over all N rewards.
            "none":  nothing; the advantage is r_i - group mean (Dr. GRPO's choice).

    Returns:
        (advantages [N], all_equal [N] bool): all_equal marks completions whose group (or
        whole batch, for scale="batch") has zero spread, so their advantage is 0. Its mean
        is TRL's logged frac_reward_zero_std.

    Math, for one group with rewards r_1 ... r_G:
        mean = (r_1 + ... + r_G) / G
        std  = sqrt( sum over i of (r_i - mean)^2 / (G - 1) )
        A_i  = (r_i - mean) / (std + 1e-4)            (scale "group")
    The 1e-4 keeps an all-equal group at A = 0 / 1e-4 = 0 instead of 0 / 0.
    NaN rewards are left out of the mean and std, and their own advantage is set to 0.

    Connection to REINFORCE with a baseline (docs/REINFORCE.md section 2): r_i - mean is
    exactly (G - 1) / G times r_i minus the leave-one-out mean of the other G - 1 rewards,
    because r_i - (r_i + S) / G = ((G - 1) / G) * (r_i - S / (G - 1)) with S the sum of the
    others. So the group mean is the unbiased leave-one-out baseline, shrunk by a constant.
    Dividing by the group's std is the part with no REINFORCE counterpart: it gives a
    prompt whose answers barely differ the same total push as one whose answers differ a
    lot. The Dr. GRPO paper (Liu et al., 2025, "Understanding R1-Zero-Like Training")
    argues this over-weights prompts that are almost always solved or almost never solved;
    scale="none" turns it off.
    """
    if scale not in SCALE_MODES:
        raise ValueError(f"scale must be one of {SCALE_MODES}, got {scale!r}")
    if group_size < 1 or rewards.numel() % group_size:
        raise ValueError(f"{rewards.numel()} rewards do not split into groups of {group_size}")
    rewards = rewards.float()
    grouped = rewards.view(-1, group_size)
    mean = torch.nanmean(grouped, dim=1).repeat_interleave(group_size)
    if scale == "batch":
        std = (
            nanstd(rewards).expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
        )
    elif group_size > 1:
        std = nanstd(grouped, dim=1).repeat_interleave(group_size)
    else:
        std = torch.zeros_like(rewards)
    advantages = rewards - mean
    if scale != "none":
        advantages = advantages / (std + STD_EPS)
    all_equal = torch.isclose(std, torch.zeros_like(std))
    return torch.nan_to_num(advantages, nan=0.0), all_equal


def completion_logps(
    model: nn.Module,
    prompt_ids: Tensor,
    prompt_mask: Tensor,
    completion_ids: Tensor,
    completion_mask: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """log pi(y_t | x, y_<t) for every completion token, at the sampling temperature.

    Prompts are left-padded and completions right-padded, as TRL builds them. Returns
    [B, T] with T = completion_ids.shape[1]; positions where completion_mask is 0 hold
    meaningless values and must be masked by the caller.

    The temperature matters: TRL samples from softmax(logits / temperature) and scores the
    same tempered distribution in the loss, so the policy being trained is the tempered
    one. At temperature 1 this is the model's own distribution.
    """
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # The logit at position p predicts token p + 1, so the completion tokens, which occupy
    # positions P ... P + T - 1, are predicted by positions P - 1 ... P + T - 2.
    n_prompt = prompt_ids.shape[1]
    logits = logits[:, n_prompt - 1 : -1].float() / temperature
    logps = torch.log_softmax(logits, dim=-1)
    return logps.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)


@dataclass
class GRPOLossOutput:
    """The loss to call backward on, and what TRL logs alongside it (all detached)."""

    loss: Tensor
    per_token_loss: Tensor  # [B, T] before masking and averaging
    kl: Tensor | None  # token mean of the KL estimate, None when beta is 0
    clip_ratio: Tensor  # share of completion tokens whose ratio was clipped


def grpo_loss(
    logps: Tensor,
    advantages: Tensor,
    mask: Tensor,
    old_logps: Tensor | None = None,
    ref_logps: Tensor | None = None,
    beta: float = 0.0,
    epsilon: float = 0.2,
    loss_type: str = "dapo",
    max_completion_length: int | None = None,
) -> GRPOLossOutput:
    """The GRPO loss of one whole generation batch (TRL 1.18, token-level importance ratio).

    Args:
        logps: [B, T] log pi_theta of each completion token, with gradient
            (completion_logps). B = prompts * G completions, T = longest completion.
        advantages: [B] group_advantages of each completion.
        mask: [B, T] 1 on completion tokens (up to and including the end token), 0 on padding.
        old_logps: [B, T] log-probs under the policy that sampled the batch. None means
            "this is the only optimizer step on this batch" (TRL's num_iterations = 1), where
            the sampling policy IS the current one, so old_logps = logps.detach().
        ref_logps: [B, T] log-probs under the reference model; needed when beta != 0.
        beta: weight of the KL penalty to the reference. TRL's default is 0.
        epsilon: clipping range of the ratio, [1 - epsilon, 1 + epsilon].
        loss_type: how per-token losses are averaged; see below.
        max_completion_length: the generation limit; needed for "dr_grpo".

    Per token t of completion i, with ratio rho = exp(logps - old_logps):

        policy term   -min( rho * A_i, clip(rho, 1 - eps, 1 + eps) * A_i )
        KL term       beta * rho * k3,   k3 = exp(ref - logp) - (ref - logp) - 1

    and the loss averages the sum of the two over tokens:

        "grpo"     mean over completions of (mean over that completion's tokens)
        "bnpo"     mean over every token of the batch
        "dapo"     mean over every token of the generation batch (TRL's default)
        "dr_grpo"  sum over every token / (B * max_completion_length)

    For one batch processed in one piece, "bnpo" and "dapo" agree. They differ when TRL
    splits a generation batch into gradient-accumulation micro-batches: "bnpo" averages
    within each micro-batch, "dapo" divides every micro-batch by the token count of the
    whole generation batch, so the accumulated gradient is this function's. ("grpo" and
    "dr_grpo" are divided by the number of micro-batches instead, which also recovers this
    function's value when micro-batches hold equal numbers of completions.)

    Why this is REINFORCE with a baseline (docs/REINFORCE.md). With one step per batch,
    old_logps = logps.detach(), so rho equals 1 in value and the clip never binds (1 lies
    inside [0.8, 1.2]). The policy term is then -A_i * rho, and since the gradient of rho is
    rho * grad logps = grad logps, the gradient of the loss is

        - (1 / N) * sum over i, t of A_i * grad log pi(y_it | x, y_i,<t)
      = - (1 / N) * sum over i of A_i * grad log pi(y_i | x)

    with N the number of tokens ("dapo"). That is the REINFORCE estimate of
    tiny_rlhf.reinforce.reinforce_loss, with return r_i, baseline = group mean, an extra
    division by the group's std, and a division by the token count instead of the number of
    samples. The ratio and the clip only start to matter when the same batch is reused for
    several optimizer steps (num_iterations > 1), which is PPO's trick for squeezing more
    out of each generation; this project does not use it.

    Why the KL term is multiplied by rho. k3 is Schulman's estimator of KL(pi || pi_ref)
    from samples of pi: its expectation is the KL and it is never negative. But the samples
    come from pi, which itself depends on theta, and "average k3 over samples, then
    differentiate" forgets that. Multiplying by rho (1 in value) adds the missing
    score-function term k3 * grad log pi. The gradient of rho * k3 at rho = 1 is

        k3 * grad logp + (1 - exp(ref - logp)) * grad logp = (logp - ref) * grad logp

    because k3 + 1 - exp(ref - logp) = -(ref - logp). So each token is pushed down in
    proportion to log pi - log pi_ref: REINFORCE on the KL-penalised return of
    docs/REINFORCE.md section 3, applied token by token. (Token by token means it ignores
    how a token changes the KL of the tokens after it; it pulls each next-token distribution
    towards the reference at the prefixes the policy visits.) Without rho (TRL's
    use_bias_correction_kl=False) the gradient would be (1 - pi_ref / pi) * grad logp, whose
    expectation is the gradient of the reverse divergence KL(pi_ref || pi).
    """
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {LOSS_TYPES}, got {loss_type!r}")
    if beta != 0.0 and ref_logps is None:
        raise ValueError("beta != 0 needs ref_logps")
    if loss_type == "dr_grpo" and max_completion_length is None:
        raise ValueError("loss_type 'dr_grpo' needs max_completion_length")
    mask = mask.to(logps.dtype)
    if old_logps is None:
        old_logps = logps.detach()
    adv = advantages.to(logps.dtype).unsqueeze(1)

    ratio = torch.exp(logps - old_logps)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    per_token = -torch.min(ratio * adv, clipped * adv)

    kl = None
    if beta != 0.0:
        diff = ref_logps - logps
        per_token_kl = (torch.exp(diff) - diff - 1) * ratio
        per_token = per_token + beta * per_token_kl
        kl = ((per_token_kl * mask).sum() / mask.sum().clamp(min=1.0)).detach()

    masked = per_token * mask
    if loss_type == "grpo":
        loss = (masked.sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    elif loss_type == "dr_grpo":
        loss = masked.sum() / (per_token.shape[0] * max_completion_length)
    else:  # "bnpo" and "dapo" on a whole generation batch
        loss = masked.sum() / mask.sum().clamp(min=1.0)

    is_clipped = ((ratio < 1 - epsilon) & (adv < 0)) | ((ratio > 1 + epsilon) & (adv > 0))
    clip_ratio = ((is_clipped.to(mask.dtype) * mask).sum() / mask.sum().clamp(min=1.0)).detach()
    return GRPOLossOutput(loss=loss, per_token_loss=per_token, kl=kl, clip_ratio=clip_ratio)
