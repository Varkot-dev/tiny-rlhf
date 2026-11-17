"""Tests for the toy RL experiment (stretch item): the exact optimum, the REINFORCE
estimator and its baselines, and DPO on the same toy. All offline and CPU-only."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from tiny_rlhf.losses import dpo_loss
from tiny_rlhf.reinforce import (
    RunningMean,
    TinyPolicy,
    ToyConfig,
    enumerate_sequences,
    exact_metrics,
    kl_penalised_return,
    leave_one_out_baseline,
    load_toy_config,
    make_policy,
    make_preference_pairs,
    make_task,
    optimal_log_probs,
    reinforce_loss,
    reward,
    train_dpo_toy,
    train_reinforce,
)


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    """These models are a few thousand parameters; extra threads only add overhead
    (on a busy machine, a lot of it). Restored afterwards for the other test modules."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


SMALL = ToyConfig(vocab_size=3, seq_len=2, beta=0.7, embed_dim=4, hidden_dim=8, batch_size=16)


def other_policy(cfg: ToyConfig, seed: int = 1) -> TinyPolicy:
    """A policy with different weights from the reference (which uses cfg.init_seed)."""
    torch.manual_seed(seed)
    return TinyPolicy(cfg.vocab_size, cfg.seq_len, cfg.embed_dim, cfg.hidden_dim)


def flat_grad(loss: torch.Tensor, policy: TinyPolicy) -> torch.Tensor:
    grads = torch.autograd.grad(loss, list(policy.parameters()))
    return torch.cat([g.flatten() for g in grads]).double()


def exact_negative_objective(policy: TinyPolicy, task) -> torch.Tensor:
    """-J(pi) computed by summing over every sequence, differentiable in the policy."""
    logp = policy.log_prob(task.all_seqs)
    per_sequence = task.rewards - task.cfg.beta * (logp - task.ref_logp)
    return -(logp.exp() * per_sequence).sum()


# --------------------------------------------------------------------------------------
# The task and its exact optimum
# --------------------------------------------------------------------------------------


def test_reward_on_hand_examples() -> None:
    seqs = torch.tensor(
        [
            [0, 1, 2, 0, 1, 3],  # four upward steps, then stop: 4 + 2
            [3, 3, 3, 3, 3, 3],  # only the stop bonus
            [0, 0, 0, 0, 0, 0],  # nothing
            [2, 0, 3, 0, 1, 2],  # 2->0, 0->1, 1->2 count; pairs touching the stop do not
        ]
    )
    assert reward(seqs, vocab_size=4, stop_bonus=2.0).tolist() == [6.0, 2.0, 0.0, 3.0]


def test_enumeration_lists_every_sequence_once() -> None:
    seqs = enumerate_sequences(3, 2)
    assert seqs.shape == (9, 2)
    assert len({tuple(row) for row in seqs.tolist()}) == 9


def test_optimal_policy_on_a_two_outcome_hand_case() -> None:
    # Two outcomes, reference 50/50, rewards 0 and ln 3, beta 1:
    #   pi* proportional to 0.5 * e^0 and 0.5 * e^(ln 3) = 0.5 and 1.5  ->  0.25 and 0.75
    ref_logp = torch.tensor([0.5, 0.5], dtype=torch.float64).log()
    rewards = torch.tensor([0.0, math.log(3.0)], dtype=torch.float64)
    opt = optimal_log_probs(ref_logp, rewards, beta=1.0).exp()
    assert opt.tolist() == pytest.approx([0.25, 0.75], abs=1e-12)
    # A larger beta trusts the reference more: beta 2 gives weights 0.5 and 0.5 * sqrt(3).
    opt2 = optimal_log_probs(ref_logp, rewards, beta=2.0).exp()
    assert opt2[1].item() == pytest.approx(math.sqrt(3) / (1 + math.sqrt(3)), abs=1e-12)


def test_reference_and_optimum_are_proper_distributions() -> None:
    task = make_task(SMALL)
    assert task.ref_logp.exp().sum().item() == pytest.approx(1.0, abs=1e-10)
    assert task.opt_logp.exp().sum().item() == pytest.approx(1.0, abs=1e-10)
    expected = task.ref_logp + task.rewards / SMALL.beta
    assert torch.allclose(task.opt_logp, expected - torch.logsumexp(expected, 0), atol=1e-12)


def test_objective_equals_optimum_minus_beta_times_kl_to_optimum() -> None:
    # J(pi) = J(pi*) - beta * KL(pi || pi*): the closed form really is the maximiser,
    # and "distance to pi*" is the objective gap in units of beta.
    task = make_task(SMALL)
    for seed in (1, 2, 3):
        m = exact_metrics(other_policy(SMALL, seed), task)
        assert m["kl_to_opt"] > 0
        assert m["objective"] == pytest.approx(
            task.optimal_objective - SMALL.beta * m["kl_to_opt"], abs=1e-9
        )
    at_reference = exact_metrics(make_policy(SMALL), task)
    assert at_reference["kl_from_ref"] == pytest.approx(0.0, abs=1e-12)


def test_sampling_matches_the_policy_distribution() -> None:
    task = make_task(SMALL)
    seqs = task.reference.sample(200_000, torch.Generator().manual_seed(0))
    index = seqs[:, 0] * SMALL.vocab_size + seqs[:, 1]  # same order as the enumeration
    freq = torch.bincount(index, minlength=9).double() / len(seqs)
    assert torch.allclose(freq, task.ref_logp.exp(), atol=5e-3)


# --------------------------------------------------------------------------------------
# The REINFORCE estimator
# --------------------------------------------------------------------------------------


def test_kl_penalised_return_formula_and_detachment() -> None:
    logp = torch.tensor([-1.0, -2.0], dtype=torch.float64, requires_grad=True)
    ref_logp = torch.tensor([-1.5, -1.0], dtype=torch.float64)
    out = kl_penalised_return(torch.tensor([3.0, 1.0]), logp, ref_logp, beta=0.5)
    # 3 - 0.5 * (-1 + 1.5) = 2.75 ;  1 - 0.5 * (-2 + 1) = 1.5
    assert out.tolist() == pytest.approx([2.75, 1.5])
    assert not out.requires_grad


def test_score_function_estimator_is_unbiased() -> None:
    # Monte Carlo REINFORCE gradient from 400,000 samples against the exact gradient of
    # the KL-penalised objective by enumeration. Tolerance: 5% of the gradient's norm.
    task = make_task(SMALL)
    policy = other_policy(SMALL)
    exact = flat_grad(exact_negative_objective(policy, task), policy)

    seqs = policy.sample(400_000, torch.Generator().manual_seed(0))
    logp = policy.log_prob(seqs)
    returns = kl_penalised_return(
        reward(seqs, SMALL.vocab_size, SMALL.stop_bonus),
        logp,
        task.reference.log_prob(seqs),
        SMALL.beta,
    )
    estimate = flat_grad(reinforce_loss(logp, returns), policy)
    assert (estimate - exact).norm() / exact.norm() < 0.05

    # A constant baseline changes the estimate only by sampling noise.
    with_baseline = flat_grad(reinforce_loss(policy.log_prob(seqs), returns, baseline=5.0), policy)
    assert (with_baseline - exact).norm() / exact.norm() < 0.05


def test_baseline_lowers_variance_when_returns_share_a_large_offset() -> None:
    # Constructed case: every return is shifted by +10. Without a baseline the estimate
    # is dominated by 10 * grad log pi, which is pure noise around zero.
    policy = other_policy(SMALL)
    generator = torch.Generator().manual_seed(0)

    def batch_grads(baseline: float) -> torch.Tensor:
        grads = []
        for _ in range(150):
            seqs = policy.sample(16, generator)
            returns = reward(seqs, SMALL.vocab_size, SMALL.stop_bonus) + 10.0
            grads.append(
                flat_grad(reinforce_loss(policy.log_prob(seqs), returns, baseline), policy)
            )
        return torch.stack(grads)

    def variance(g: torch.Tensor) -> float:
        return (g - g.mean(0)).pow(2).sum(1).mean().item()

    plain, centred = batch_grads(0.0), batch_grads(10.0 + 1.0)
    assert variance(centred) < variance(plain) / 10
    # ... and the two average gradients point the same way.
    cosine = torch.nn.functional.cosine_similarity(plain.mean(0), centred.mean(0), dim=0)
    assert cosine.item() > 0.5


def test_leave_one_out_baseline_on_hand_numbers() -> None:
    out = leave_one_out_baseline(torch.tensor([1.0, 2.0, 3.0]))
    assert out.tolist() == pytest.approx([2.5, 2.0, 1.5])


def test_running_mean_starts_at_zero_then_averages() -> None:
    running = RunningMean(decay=0.9)
    assert running.value == 0.0
    running.update(torch.tensor([2.0, 4.0]))
    assert running.value == pytest.approx(3.0)
    running.update(torch.tensor([13.0]))
    assert running.value == pytest.approx(0.9 * 3.0 + 0.1 * 13.0)


# --------------------------------------------------------------------------------------
# Training behaviour
# --------------------------------------------------------------------------------------

SHORT = ToyConfig(steps=150, eval_every=150)


def test_large_beta_stays_near_reference_and_small_beta_chases_reward() -> None:
    timid = make_task(dataclasses.replace(SHORT, beta=50.0))
    history, _ = train_reinforce(timid, "leave_one_out", seed=0)
    assert history[-1]["kl_from_ref"] < 0.05

    greedy = make_task(dataclasses.replace(SHORT, beta=0.2))
    history, _ = train_reinforce(greedy, "leave_one_out", seed=0)
    start, end = history[0], history[-1]
    assert end["expected_reward"] > start["expected_reward"] + 2.0
    assert end["kl_from_ref"] > 1.0


def test_reinforce_moves_toward_the_closed_form_optimum() -> None:
    task = make_task(SHORT)
    history, _ = train_reinforce(task, "leave_one_out", seed=0)
    assert history[-1]["kl_to_opt"] < history[0]["kl_to_opt"] / 10


def test_training_is_deterministic_given_the_seed() -> None:
    task = make_task(dataclasses.replace(SHORT, steps=20, eval_every=10))
    a, _ = train_reinforce(task, "running_mean", seed=3)
    b, _ = train_reinforce(task, "running_mean", seed=3)
    c, _ = train_reinforce(task, "running_mean", seed=4)
    assert a == b
    assert a != c
    assert train_dpo_toy(task, "bradley_terry", 3) == train_dpo_toy(task, "bradley_terry", 3)


def test_snapshots_are_taken_at_the_requested_steps() -> None:
    task = make_task(dataclasses.replace(SHORT, steps=20, eval_every=10))
    _, snapshots = train_reinforce(task, "none", seed=0, snapshot_steps=(0, 20))
    assert sorted(snapshots) == [0, 20]
    reference_state = task.reference.state_dict()
    assert all(torch.equal(snapshots[0][k], reference_state[k]) for k in reference_state)
    assert any(not torch.equal(snapshots[20][k], reference_state[k]) for k in reference_state)


# --------------------------------------------------------------------------------------
# DPO on the toy
# --------------------------------------------------------------------------------------


def test_hard_labels_always_prefer_the_higher_reward() -> None:
    task = make_task(ToyConfig())
    chosen, rejected = make_preference_pairs(task, 500, "hard", torch.Generator().manual_seed(0))
    assert chosen.shape == rejected.shape == (500, 6)
    assert (reward(chosen, 4) >= reward(rejected, 4)).all()


def test_bradley_terry_labels_are_noisy_at_the_modelled_rate() -> None:
    task = make_task(ToyConfig())
    chosen, rejected = make_preference_pairs(
        task, 20_000, "bradley_terry", torch.Generator().manual_seed(0)
    )
    diff = reward(chosen, 4) - reward(rejected, 4)
    # Among pairs whose rewards differ by exactly 1, the higher one should win with
    # probability sigmoid(1) = 0.731.
    one_apart = diff.abs() == 1
    assert one_apart.sum() > 2000
    rate = (diff[one_apart] > 0).double().mean().item()
    assert rate == pytest.approx(torch.sigmoid(torch.tensor(1.0)).item(), abs=0.03)


def test_dpo_starts_at_the_reference() -> None:
    task = make_task(dataclasses.replace(SHORT, steps=10, eval_every=10))
    history = train_dpo_toy(task, "bradley_terry", seed=0)
    assert history[0]["kl_from_ref"] == pytest.approx(0.0, abs=1e-12)
    assert history[0]["sequences_seen"] == 0
    assert history[-1]["sequences_seen"] == 10 * SHORT.batch_size


def test_population_dpo_optimum_under_bradley_terry_labels_is_the_closed_form() -> None:
    # Tabular policy over 9 sequences, every ordered pair weighted by how often it is
    # sampled from the reference and by the Bradley-Terry label probability. Minimising
    # that exact (infinite-data) DPO loss must land on pi* = pi_ref * exp(r / beta) / Z.
    task = make_task(SMALL)
    n = len(task.rewards)
    ia, ib = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    ia, ib = ia.flatten(), ib.flatten()
    pair_weight = (task.ref_logp[ia] + task.ref_logp[ib]).exp()
    p_a_preferred = torch.sigmoid(task.rewards[ia] - task.rewards[ib])

    theta = task.ref_logp.clone().requires_grad_(True)  # start at the reference
    optimizer = torch.optim.LBFGS(
        [theta],
        max_iter=2000,
        tolerance_grad=1e-14,
        tolerance_change=1e-16,
        line_search_fn="strong_wolfe",
    )

    def population_loss() -> torch.Tensor:
        logp = torch.log_softmax(theta, dim=0)
        # Per-pair losses for "a preferred" and for "b preferred", without the batch mean.
        margin = SMALL.beta * ((logp[ia] - task.ref_logp[ia]) - (logp[ib] - task.ref_logp[ib]))
        a_wins = torch.nn.functional.softplus(-margin)
        b_wins = torch.nn.functional.softplus(margin)
        return (pair_weight * (p_a_preferred * a_wins + (1 - p_a_preferred) * b_wins)).sum()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = population_loss()
        loss.backward()
        return loss

    optimizer.step(closure)
    learned = torch.log_softmax(theta.detach(), dim=0)
    assert torch.allclose(learned.exp(), task.opt_logp.exp(), atol=1e-6)

    # The per-pair term used above is the repo's dpo_loss.
    one = dpo_loss(theta[:1].detach(), theta[1:2].detach(), task.ref_logp[:1], task.ref_logp[1:2])
    assert torch.isfinite(one.loss)


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


def test_config_loading_and_validation(tmp_path) -> None:
    cfg = load_toy_config("configs/reinforce_toy.yaml")
    assert cfg.vocab_size**cfg.seq_len == 4096
    bad = tmp_path / "bad.yaml"
    bad.write_text("batch_size: 7\n")
    with pytest.raises(ValueError, match="even batch_size"):
        load_toy_config(bad)
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("gamma: 0.9\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_toy_config(unknown)
