"""Tests for the M0 building blocks.

The three "done when" criteria from docs/PLAN.md each have a section:
1. the DPO loss is log 2 when the policy equals the reference
2. gradients raise the preferred response and lower the other
3. masking matches a hand-computed example
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn

from tiny_rlhf.losses import bradley_terry_loss, dpo_loss, sequence_logprob

LOG2 = math.log(2.0)
SEED = 0


def logits_from_probs(probs: list[list[float]]) -> torch.Tensor:
    """Logits whose softmax is exactly the given rows of probabilities. Shape [1, T, V]."""
    return torch.tensor(probs, dtype=torch.float64).log().unsqueeze(0)


class BigramLM(nn.Module):
    """Smallest possible causal LM: the next-token logits depend on the current token only."""

    def __init__(self, vocab: int = 11, dim: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.out = nn.Linear(dim, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.out(self.emb(input_ids))


def make_pair_batch() -> dict[str, torch.Tensor]:
    """Two prompts, each with a chosen and a rejected response, padded to length 6.

    Token 0 is padding. Each row is prompt (2 tokens) + response + padding.
    """
    chosen = torch.tensor([[3, 4, 5, 6, 7, 0], [8, 9, 1, 2, 0, 0]])
    rejected = torch.tensor([[3, 4, 9, 9, 0, 0], [8, 9, 6, 5, 4, 3]])
    chosen_mask = torch.tensor([[0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0]]).bool()
    rejected_mask = torch.tensor([[0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1]]).bool()
    return {
        "chosen": chosen,
        "rejected": rejected,
        "chosen_mask": chosen_mask,
        "rejected_mask": rejected_mask,
    }


def pair_logps(model: nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    chosen = sequence_logprob(model(batch["chosen"]), batch["chosen"], batch["chosen_mask"])
    rejected = sequence_logprob(model(batch["rejected"]), batch["rejected"], batch["rejected_mask"])
    return chosen, rejected


# --------------------------------------------------------------------------------------
# sequence_logprob: masking and the one-position shift
# --------------------------------------------------------------------------------------

# One sequence, vocabulary of 3, length 5:
#   position   0        1        2          3          4
#   token      2        0        1          2          0
#   role       prompt   prompt   response   response   padding
#
# The logits at position t are the model's distribution over the token at t+1.
#   token at position 2 (id 1) is predicted at position 1: P = 0.50
#   token at position 3 (id 2) is predicted at position 2: P = 0.25
# So log pi(y|x) = log 0.50 + log 0.25 = log 0.125 = -2.0794...
HAND_IDS = torch.tensor([[2, 0, 1, 2, 0]])
HAND_MASK = torch.tensor([[0, 0, 1, 1, 0]])
HAND_PROBS = [
    [0.20, 0.30, 0.50],  # position 0 predicts position 1 (prompt): must be ignored
    [0.25, 0.50, 0.25],  # position 1 predicts position 2: P(token 1) = 0.50
    [0.50, 0.25, 0.25],  # position 2 predicts position 3: P(token 2) = 0.25
    [0.10, 0.10, 0.80],  # position 3 predicts position 4 (padding): must be ignored
    [0.70, 0.20, 0.10],  # position 4 predicts nothing we have: must be ignored
]
HAND_EXPECTED = math.log(0.5) + math.log(0.25)


def test_sequence_logprob_matches_hand_computed_example() -> None:
    got = sequence_logprob(logits_from_probs(HAND_PROBS), HAND_IDS, HAND_MASK)
    assert got.shape == (1,)
    assert got.item() == pytest.approx(HAND_EXPECTED, abs=1e-12)
    assert got.item() == pytest.approx(math.log(0.125), abs=1e-12)


def test_sequence_logprob_would_catch_a_missing_shift() -> None:
    # The classic bug reads the token at position t from the logits at position t.
    # On the hand example that gives log 0.25 + log 0.80, which is not the right answer,
    # so the example above really does distinguish shifted from unshifted indexing.
    unshifted = math.log(HAND_PROBS[2][1]) + math.log(HAND_PROBS[3][2])
    assert unshifted != pytest.approx(HAND_EXPECTED, abs=1e-3)


def test_prompt_and_padding_logits_do_not_affect_the_result() -> None:
    torch.manual_seed(SEED)
    base = logits_from_probs(HAND_PROBS)
    changed = base.clone()
    # Positions 0, 3 and 4 only ever predict prompt or padding tokens here.
    changed[:, [0, 3, 4], :] = torch.randn(1, 3, 3, dtype=torch.float64) * 5
    assert torch.equal(
        sequence_logprob(base, HAND_IDS, HAND_MASK),
        sequence_logprob(changed, HAND_IDS, HAND_MASK),
    )


def test_padding_token_ids_do_not_affect_the_result() -> None:
    logits = logits_from_probs(HAND_PROBS)
    other_pad = HAND_IDS.clone()
    other_pad[0, 4] = 1
    assert torch.equal(
        sequence_logprob(logits, HAND_IDS, HAND_MASK),
        sequence_logprob(logits, other_pad, HAND_MASK),
    )


def test_padding_a_sequence_leaves_its_logprob_unchanged() -> None:
    torch.manual_seed(SEED)
    model = BigramLM().double()
    short = torch.tensor([[3, 4, 5, 6]])
    short_mask = torch.tensor([[0, 0, 1, 1]])
    padded = torch.tensor([[3, 4, 5, 6, 0, 0, 0]])
    padded_mask = torch.tensor([[0, 0, 1, 1, 0, 0, 0]])
    a = sequence_logprob(model(short), short, short_mask)
    b = sequence_logprob(model(padded), padded, padded_mask)
    assert a.item() == pytest.approx(b.item(), abs=1e-12)


def test_masked_position_with_zero_probability_does_not_poison_the_sum() -> None:
    # The prompt token at position 1 gets probability 0 (logit -inf). It is masked,
    # so the answer must still be finite. Multiplying by the mask would give nan.
    logits = logits_from_probs(HAND_PROBS)
    logits[0, 0, 0] = float("-inf")
    got = sequence_logprob(logits, HAND_IDS, HAND_MASK)
    assert got.item() == pytest.approx(HAND_EXPECTED, abs=1e-12)


def test_sequence_logprob_is_per_row_in_a_batch() -> None:
    torch.manual_seed(SEED)
    model = BigramLM().double()
    batch = make_pair_batch()
    together = sequence_logprob(model(batch["chosen"]), batch["chosen"], batch["chosen_mask"])
    for i in range(2):
        ids = batch["chosen"][i : i + 1]
        alone = sequence_logprob(model(ids), ids, batch["chosen_mask"][i : i + 1])
        assert together[i].item() == pytest.approx(alone.item(), abs=1e-12)


def test_half_precision_logits_are_scored_in_float32() -> None:
    logits = logits_from_probs(HAND_PROBS).to(torch.bfloat16)
    got = sequence_logprob(logits, HAND_IDS, HAND_MASK)
    assert got.dtype == torch.float32
    assert got.item() == pytest.approx(HAND_EXPECTED, abs=2e-2)  # bfloat16 input rounding


def test_sequence_logprob_rejects_response_at_position_zero() -> None:
    logits = logits_from_probs(HAND_PROBS)
    with pytest.raises(ValueError, match="position 0"):
        sequence_logprob(logits, HAND_IDS, torch.tensor([[1, 1, 1, 1, 0]]))


def test_sequence_logprob_rejects_mismatched_shapes() -> None:
    logits = logits_from_probs(HAND_PROBS)
    with pytest.raises(ValueError, match="shape mismatch"):
        sequence_logprob(logits, HAND_IDS[:, :4], HAND_MASK[:, :4])


# --------------------------------------------------------------------------------------
# bradley_terry_loss
# --------------------------------------------------------------------------------------


def test_bradley_terry_is_log2_for_equal_rewards() -> None:
    r = torch.tensor([0.3, -4.0, 12.5], dtype=torch.float64)
    assert bradley_terry_loss(r, r).item() == pytest.approx(LOG2, abs=1e-12)


def test_bradley_terry_hand_computed_values() -> None:
    # difference +1: -log sigmoid(1)  = log(1 + e^-1) = 0.31326...
    # difference -1: -log sigmoid(-1) = log(1 + e^+1) = 1.31326...
    one = torch.tensor([1.0], dtype=torch.float64)
    zero = torch.tensor([0.0], dtype=torch.float64)
    assert bradley_terry_loss(one, zero).item() == pytest.approx(
        math.log1p(math.exp(-1)), abs=1e-12
    )
    assert bradley_terry_loss(zero, one).item() == pytest.approx(math.log1p(math.exp(1)), abs=1e-12)
    assert bradley_terry_loss(one, zero).item() == pytest.approx(0.3132616875, abs=1e-9)


def test_bradley_terry_only_depends_on_the_difference() -> None:
    torch.manual_seed(SEED)
    r_w = torch.randn(16, dtype=torch.float64)
    r_l = torch.randn(16, dtype=torch.float64)
    shift = torch.randn(16, dtype=torch.float64) * 100  # a different constant per prompt
    assert bradley_terry_loss(r_w + shift, r_l + shift).item() == pytest.approx(
        bradley_terry_loss(r_w, r_l).item(), abs=1e-9
    )


def test_bradley_terry_gradient_signs_and_values() -> None:
    r_w = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64, requires_grad=True)
    r_l = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64, requires_grad=True)
    bradley_terry_loss(r_w, r_l).backward()
    # d/d r_w of -log sigmoid(r_w - r_l) is -sigmoid(-(r_w - r_l)); the mean divides by 3.
    expected = -torch.sigmoid(-(r_w - r_l)).detach() / 3
    assert torch.allclose(r_w.grad, expected, atol=1e-12)
    assert torch.allclose(r_l.grad, -expected, atol=1e-12)
    assert (r_w.grad < 0).all() and (r_l.grad > 0).all()


def test_bradley_terry_is_finite_for_extreme_differences() -> None:
    big = torch.tensor([1000.0, -1000.0])
    loss = bradley_terry_loss(big, -big)
    assert torch.isfinite(loss)
    # First pair: essentially 0. Second pair: essentially the difference itself, 2000.
    assert loss.item() == pytest.approx(1000.0, rel=1e-6)


# --------------------------------------------------------------------------------------
# dpo_loss: criterion 1, log 2 when the policy equals the reference
# --------------------------------------------------------------------------------------


def test_dpo_is_log2_when_policy_equals_reference() -> None:
    torch.manual_seed(SEED)
    chosen = -torch.rand(8, dtype=torch.float64) * 50
    rejected = -torch.rand(8, dtype=torch.float64) * 50
    for beta in (0.01, 0.1, 0.5, 5.0):
        out = dpo_loss(chosen, rejected, chosen, rejected, beta=beta)
        assert out.loss.item() == pytest.approx(LOG2, abs=1e-12)
        assert torch.equal(out.reward_margin, torch.zeros_like(chosen))
        assert out.accuracy.item() == 0.0  # ties are not counted as correct


def test_dpo_is_log2_end_to_end_with_a_copied_model() -> None:
    torch.manual_seed(SEED)
    reference = BigramLM().double()
    policy = copy.deepcopy(reference)
    batch = make_pair_batch()
    with torch.no_grad():
        ref_c, ref_r = pair_logps(reference, batch)
    pol_c, pol_r = pair_logps(policy, batch)
    out = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
    assert out.loss.item() == pytest.approx(LOG2, abs=1e-12)


# --------------------------------------------------------------------------------------
# dpo_loss: criterion 2, gradients raise the preferred response and lower the other
# --------------------------------------------------------------------------------------


def test_dpo_gradient_signs_and_exact_values() -> None:
    beta = 0.25
    pol_c = torch.tensor([-10.0, -3.0, -7.0], dtype=torch.float64, requires_grad=True)
    pol_r = torch.tensor([-9.0, -8.0, -7.5], dtype=torch.float64, requires_grad=True)
    ref_c = torch.tensor([-11.0, -2.0, -7.0], dtype=torch.float64)
    ref_r = torch.tensor([-9.5, -4.0, -7.5], dtype=torch.float64)
    dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=beta).loss.backward()

    # With u = (pol_c - ref_c) - (pol_r - ref_r):
    #   d loss / d pol_c = -beta * sigmoid(-beta * u) / batch
    #   d loss / d pol_r = +beta * sigmoid(-beta * u) / batch
    u = (pol_c - ref_c) - (pol_r - ref_r)
    weight = beta * torch.sigmoid(-beta * u).detach() / 3
    assert torch.allclose(pol_c.grad, -weight, atol=1e-12)
    assert torch.allclose(pol_r.grad, weight, atol=1e-12)
    # Gradient descent moves against the gradient: chosen log-prob up, rejected down.
    assert (pol_c.grad < 0).all()
    assert (pol_r.grad > 0).all()


def test_dpo_gradient_at_step_zero_is_beta_over_two() -> None:
    # At pi = pi_ref, u = 0 and sigmoid(0) = 1/2, so every pair pushes with weight beta/2.
    beta = 0.1
    logp = torch.tensor([-5.0], dtype=torch.float64, requires_grad=True)
    other = torch.tensor([-6.0], dtype=torch.float64, requires_grad=True)
    dpo_loss(logp, other, logp.detach(), other.detach(), beta=beta).loss.backward()
    assert logp.grad.item() == pytest.approx(-beta / 2, abs=1e-12)
    assert other.grad.item() == pytest.approx(beta / 2, abs=1e-12)


def test_no_gradient_reaches_the_reference() -> None:
    pol = torch.tensor([-5.0], requires_grad=True)
    ref_c = torch.tensor([-4.0], requires_grad=True)
    ref_r = torch.tensor([-6.0], requires_grad=True)
    dpo_loss(pol, pol - 1, ref_c, ref_r).loss.backward()
    assert ref_c.grad is None and ref_r.grad is None


def test_one_sgd_step_raises_chosen_and_lowers_rejected_end_to_end() -> None:
    torch.manual_seed(SEED)
    reference = BigramLM().double()
    policy = copy.deepcopy(reference)
    batch = make_pair_batch()
    with torch.no_grad():
        ref_c, ref_r = pair_logps(reference, batch)

    pol_c, pol_r = pair_logps(policy, batch)
    before = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
    before.loss.backward()
    with torch.no_grad():
        for p in policy.parameters():
            p -= 0.05 * p.grad

    with torch.no_grad():
        new_c, new_r = pair_logps(policy, batch)
        after = dpo_loss(new_c, new_r, ref_c, ref_r, beta=0.1)
    assert after.loss.item() < before.loss.item() == pytest.approx(LOG2, abs=1e-12)
    assert (after.reward_margin > 0).all()
    assert after.accuracy.item() == 1.0
    # In this toy the chosen and rejected responses share no (current, next) token
    # transitions, so the model can move each one freely: chosen up, rejected down.
    assert (new_c > ref_c).all()
    assert (new_r < ref_r).all()


# --------------------------------------------------------------------------------------
# dpo_loss: structure
# --------------------------------------------------------------------------------------


def test_dpo_is_bradley_terry_on_the_implicit_reward() -> None:
    torch.manual_seed(SEED)
    beta = 0.3
    pol_c, pol_r, ref_c, ref_r = (-torch.rand(4, 12, dtype=torch.float64) * 30).unbind(0)
    out = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=beta)
    r_chosen = beta * (pol_c - ref_c)
    r_rejected = beta * (pol_r - ref_r)
    assert out.loss.item() == pytest.approx(
        bradley_terry_loss(r_chosen, r_rejected).item(), abs=1e-12
    )
    assert torch.allclose(out.chosen_rewards, r_chosen)
    assert torch.allclose(out.rejected_rewards, r_rejected)
    assert torch.allclose(out.reward_margin, r_chosen - r_rejected)


def test_dpo_hand_computed_value() -> None:
    # Policy moved chosen up by 2 nats and rejected down by 3 nats relative to the
    # reference, so the bracket is 2 - (-3) = 5. With beta = 0.1 the sigmoid sees 0.5.
    # loss = log(1 + e^-0.5) = 0.474077...
    out = dpo_loss(
        torch.tensor([-8.0], dtype=torch.float64),
        torch.tensor([-13.0], dtype=torch.float64),
        torch.tensor([-10.0], dtype=torch.float64),
        torch.tensor([-10.0], dtype=torch.float64),
        beta=0.1,
    )
    assert out.reward_margin.item() == pytest.approx(0.5, abs=1e-12)
    assert out.loss.item() == pytest.approx(math.log1p(math.exp(-0.5)), abs=1e-12)
    assert out.loss.item() == pytest.approx(0.4740769842, abs=1e-9)
    assert out.accuracy.item() == 1.0


def test_dpo_loss_can_fall_while_the_chosen_response_gets_less_likely() -> None:
    # The failure mode from docs/PLAN.md section 2: both responses lose probability,
    # the rejected one loses more, and the loss still improves below log 2.
    ref = torch.tensor([-10.0], dtype=torch.float64)
    out = dpo_loss(ref - 4.0, ref - 20.0, ref, ref, beta=0.1)
    assert out.chosen_rewards.item() < 0  # the preferred response is now LESS likely
    assert out.loss.item() < LOG2  # and the loss says things got better
    assert out.accuracy.item() == 1.0


def test_dpo_accuracy_counts_pairs_with_positive_margin() -> None:
    zeros = torch.zeros(4)
    pol_c = torch.tensor([1.0, -1.0, 2.0, 0.0])
    out = dpo_loss(pol_c, zeros, zeros, zeros, beta=1.0)
    assert out.accuracy.item() == pytest.approx(0.5)


def test_dpo_rejects_bad_inputs() -> None:
    x = torch.zeros(3)
    with pytest.raises(ValueError, match="beta"):
        dpo_loss(x, x, x, x, beta=0.0)
    with pytest.raises(ValueError, match="share a shape"):
        dpo_loss(x, x, x, torch.zeros(4))
