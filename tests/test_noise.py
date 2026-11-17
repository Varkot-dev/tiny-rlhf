"""Tests for label noise: flipping pairs, the robust DPO loss, and the KL drift measure."""

from __future__ import annotations

import copy
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from tiny_rlhf import train_dpo
from tiny_rlhf.data import TokenizedPair
from tiny_rlhf.losses import dpo_loss
from tiny_rlhf.noise import (
    flip_at,
    flip_pairs,
    indices_fingerprint,
    robust_dpo_loss,
    train_with_label_noise,
    training_loss,
)
from tiny_rlhf.train_dpo import DPOConfig

LOG2 = math.log(2.0)
PAD = 0
VOCAB = 50


def make_pairs(n: int, seed: int = 0) -> list[TokenizedPair]:
    g = torch.Generator().manual_seed(seed)

    def ids(length: int) -> list[int]:
        return torch.randint(3, VOCAB, (length,), generator=g).tolist()

    return [TokenizedPair(ids(3 + i % 3), ids(2 + i % 4), ids(2 + (i + 2) % 4)) for i in range(n)]


def tiny_model(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=VOCAB,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            pad_token_id=PAD,
            bos_token_id=1,
            eos_token_id=2,
        )
    )


# --------------------------------------------------------------------------------------
# flip_pairs
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("fraction", "expected"), [(0.0, 0), (0.1, 2), (0.3, 6), (1.0, 20)])
def test_flip_pairs_flips_an_exact_count(fraction: float, expected: int) -> None:
    pairs = make_pairs(20)
    noisy, flipped = flip_pairs(pairs, fraction, seed=0)
    assert len(flipped) == expected == len(set(flipped))
    assert flipped == sorted(flipped)
    assert len(noisy) == len(pairs)
    for i, (before, after) in enumerate(zip(pairs, noisy, strict=True)):
        assert after.prompt_ids == before.prompt_ids  # the prompt never changes
        if i in flipped:
            assert after.chosen_ids == before.rejected_ids
            assert after.rejected_ids == before.chosen_ids
        else:
            assert after == before


def test_flip_pairs_is_deterministic_and_depends_on_the_seed() -> None:
    pairs = make_pairs(40)
    assert flip_pairs(pairs, 0.25, seed=3) == flip_pairs(pairs, 0.25, seed=3)
    assert flip_pairs(pairs, 0.25, seed=3)[1] != flip_pairs(pairs, 0.25, seed=4)[1]


def test_flip_pairs_does_not_modify_its_input() -> None:
    pairs = make_pairs(10)
    snapshot = copy.deepcopy(pairs)
    flip_pairs(pairs, 0.5, seed=0)
    assert pairs == snapshot


def test_flipping_the_same_indices_twice_restores_the_pairs() -> None:
    pairs = make_pairs(12)
    noisy, flipped = flip_pairs(pairs, 0.5, seed=1)
    assert noisy != pairs
    assert flip_at(noisy, flipped) == pairs


def test_flip_pairs_rejects_a_bad_fraction_and_flip_at_a_bad_index() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        flip_pairs(make_pairs(4), 1.5, seed=0)
    with pytest.raises(ValueError, match="out of range"):
        flip_at(make_pairs(4), [4])


def test_indices_fingerprint_changes_with_the_indices() -> None:
    assert indices_fingerprint([1, 2, 3]) == indices_fingerprint([1, 2, 3])
    assert indices_fingerprint([1, 2, 3]) != indices_fingerprint([1, 2, 4])


# --------------------------------------------------------------------------------------
# robust_dpo_loss
# --------------------------------------------------------------------------------------


def logps(seed: int = 0, n: int = 16) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    return tuple((-torch.rand(4, n, dtype=torch.float64) * 30).unbind(0))


def test_robust_loss_with_zero_smoothing_is_the_ordinary_dpo_loss() -> None:
    args = logps()
    ours = robust_dpo_loss(*args, beta=0.3, label_smoothing=0.0)
    plain = dpo_loss(*args, beta=0.3)
    assert ours.loss.item() == pytest.approx(plain.loss.item(), abs=1e-12)
    assert torch.equal(ours.reward_margin, plain.reward_margin)
    assert ours.accuracy.item() == plain.accuracy.item()


@pytest.mark.parametrize("eps", [0.0, 0.1, 0.3, 0.45])
def test_robust_loss_is_log2_when_policy_equals_reference(eps: float) -> None:
    pol_c, pol_r, _, _ = logps()
    out = robust_dpo_loss(pol_c, pol_r, pol_c, pol_r, beta=0.1, label_smoothing=eps)
    assert out.loss.item() == pytest.approx(LOG2, abs=1e-12)


def test_robust_loss_hand_computed_value() -> None:
    # Bracket u = 5 (chosen up 2 nats, rejected down 3), beta = 0.1, so beta*u = 0.5.
    #   L(u)  = log(1 + e^-0.5) = 0.474077
    #   L(-u) = log(1 + e^+0.5) = 0.974077
    # With e = 0.3: (0.7 * 0.474077 - 0.3 * 0.974077) / 0.4 = 0.099077
    out = robust_dpo_loss(
        torch.tensor([-8.0], dtype=torch.float64),
        torch.tensor([-13.0], dtype=torch.float64),
        torch.tensor([-10.0], dtype=torch.float64),
        torch.tensor([-10.0], dtype=torch.float64),
        beta=0.1,
        label_smoothing=0.3,
    )
    expected = (0.7 * math.log1p(math.exp(-0.5)) - 0.3 * math.log1p(math.exp(0.5))) / 0.4
    assert out.loss.item() == pytest.approx(expected, abs=1e-12)
    assert out.loss.item() == pytest.approx(0.0990769842, abs=1e-9)


@pytest.mark.parametrize("eps", [0.05, 0.1, 0.3])
def test_robust_loss_matches_trl_formula(eps: float) -> None:
    trl_trainer = pytest.importorskip("trl.trainer.dpo_trainer")
    source = inspect.getsource(trl_trainer)
    # The three lines of TRL's loss_type == "robust" branch. If TRL changes them, this
    # fails and the comparison below needs to be re-derived from the new source.
    for line in (
        "clean_loss_term = -(1 - self.label_smoothing) * F.logsigmoid(self.beta * delta_score)",
        "flipped_loss_term = -self.label_smoothing * F.logsigmoid(-self.beta * delta_score)",
        "per_sequence_loss = (clean_loss_term - flipped_loss_term) "
        "/ (1 - 2 * self.label_smoothing)",
    ):
        assert line in source

    beta = 0.25
    pol_c, pol_r, ref_c, ref_r = logps(seed=1)
    delta_score = (pol_c - ref_c) - (pol_r - ref_r)
    clean_loss_term = -(1 - eps) * F.logsigmoid(beta * delta_score)
    flipped_loss_term = -eps * F.logsigmoid(-beta * delta_score)
    trl_loss = ((clean_loss_term - flipped_loss_term) / (1 - 2 * eps)).mean()

    ours = robust_dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=beta, label_smoothing=eps)
    assert ours.loss.item() == pytest.approx(trl_loss.item(), abs=1e-12)


def test_label_smoothing_does_nothing_to_trl_default_sigmoid_loss() -> None:
    # The plan says "TRL's label-smoothing option". In trl 1.18.0 the default loss is one
    # line with no label_smoothing in it; the option only acts through other loss types.
    trl_trainer = pytest.importorskip("trl.trainer.dpo_trainer")
    source = inspect.getsource(trl_trainer)
    start = source.index('if loss_type == "sigmoid":')
    end = source.index('elif loss_type == "hinge":')
    assert "label_smoothing" not in source[start:end]
    assert "per_sequence_loss = -F.logsigmoid(self.beta * delta_score)" in source[start:end]


def test_robust_loss_is_unbiased_under_the_assumed_flip_rate() -> None:
    # The defining property: averaged over the label noise, the robust loss of the noisy
    # pair equals the ordinary loss of the clean pair. Checked exactly, pair by pair.
    eps, beta = 0.3, 0.2
    pol_c, pol_r, ref_c, ref_r = logps(seed=2, n=8)

    def robust_each(a: torch.Tensor, b: torch.Tensor, ra: torch.Tensor, rb: torch.Tensor):
        return torch.stack(
            [
                robust_dpo_loss(
                    a[i : i + 1], b[i : i + 1], ra[i : i + 1], rb[i : i + 1], beta, eps
                ).loss
                for i in range(len(a))
            ]
        )

    seen_correct = robust_each(pol_c, pol_r, ref_c, ref_r)
    seen_flipped = robust_each(pol_r, pol_c, ref_r, ref_c)
    expected = (1 - eps) * seen_correct + eps * seen_flipped
    clean = torch.stack(
        [
            dpo_loss(
                pol_c[i : i + 1], pol_r[i : i + 1], ref_c[i : i + 1], ref_r[i : i + 1], beta
            ).loss
            for i in range(len(pol_c))
        ]
    )
    assert torch.allclose(expected, clean, atol=1e-12)


def test_robust_loss_keeps_pushing_and_can_go_negative() -> None:
    # Unlike ordinary DPO, the push on a pair that is already ranked correctly does not
    # fade to zero: it tends to beta * e / (1 - 2e). And the loss is not bounded below.
    eps, beta = 0.3, 0.1
    pol_c = torch.tensor([200.0], dtype=torch.float64, requires_grad=True)
    zero = torch.zeros(1, dtype=torch.float64)
    out = robust_dpo_loss(pol_c, zero, zero, zero, beta=beta, label_smoothing=eps)
    out.loss.backward()
    assert out.loss.item() < 0
    assert -pol_c.grad.item() == pytest.approx(beta * eps / (1 - 2 * eps), rel=1e-6)

    plain_c = torch.tensor([200.0], dtype=torch.float64, requires_grad=True)
    dpo_loss(plain_c, zero, zero, zero, beta=beta).loss.backward()
    assert abs(plain_c.grad.item()) < 1e-9


def test_robust_loss_gradient_direction_and_no_gradient_to_reference() -> None:
    pol_c, pol_r, ref_c, ref_r = (t.clone().requires_grad_(True) for t in logps(seed=3))
    robust_dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.2, label_smoothing=0.2).loss.backward()
    assert (pol_c.grad < 0).all() and (pol_r.grad > 0).all()
    assert ref_c.grad is None and ref_r.grad is None


def test_robust_loss_rejects_bad_arguments() -> None:
    x = torch.zeros(2)
    with pytest.raises(ValueError, match="label_smoothing"):
        robust_dpo_loss(x, x, x, x, label_smoothing=0.5)
    with pytest.raises(ValueError, match="beta"):
        robust_dpo_loss(x, x, x, x, beta=0.0)


# --------------------------------------------------------------------------------------
# training on noisy pairs
# --------------------------------------------------------------------------------------


def make_cfg(**overrides: object) -> DPOConfig:
    values = {
        "run_name": "noise_test",
        "n_train": 8,
        "n_eval": 8,
        "max_length": 32,
        "beta": 0.5,
        "lr": 1e-2,
        "warmup_ratio": 0.0,
        "epochs": 1,
        "batch_size": 4,
        "grad_accum": 1,
        "eval_every": 1000,
        "eval_batch_size": 4,
        "seed": 0,
        "use_lora": False,
        "device": "cpu",
        "save_adapter": False,
    }
    values.update(overrides)
    return DPOConfig(**values)


def test_training_loss_swaps_the_loss_and_puts_it_back() -> None:
    original = train_dpo.dpo_loss
    with training_loss(robust_dpo_loss):
        assert train_dpo.dpo_loss is robust_dpo_loss
    assert train_dpo.dpo_loss is original
    with pytest.raises(RuntimeError), training_loss(robust_dpo_loss):
        raise RuntimeError("boom")
    assert train_dpo.dpo_loss is original


def test_train_with_label_noise_records_what_it_did(tmp_path: Path) -> None:
    policy = tiny_model()
    train_pairs, eval_pairs = make_pairs(8), make_pairs(8, seed=1)
    info = train_with_label_noise(
        make_cfg(),
        policy,
        copy.deepcopy(policy),
        train_pairs,
        eval_pairs,
        PAD,
        tmp_path / "run",
        noise_fraction=0.25,
        noise_seed=0,
    )
    assert info["label_noise"]["n_flipped"] == 2
    assert info["label_noise"]["fraction_flipped"] == 0.25
    assert info["loss_type"] == "sigmoid" and info["label_smoothing"] == 0.0
    assert info["optimizer_steps"] == 2
    # Held-out pairs are scored exactly as given: their fingerprint is the clean one.
    from tiny_rlhf.data import pairs_fingerprint

    assert info["eval_pairs_sha256"] == pairs_fingerprint(eval_pairs, 32)
    on_disk = json.loads((tmp_path / "run" / "run_info.json").read_text())
    assert on_disk["label_noise"] == info["label_noise"]
    assert on_disk["label_noise"]["flipped_indices_sha256"] == indices_fingerprint(
        flip_pairs(train_pairs, 0.25, 0)[1]
    )


def test_train_with_robust_loss_starts_at_log2_and_restores_the_loss(tmp_path: Path) -> None:
    original = train_dpo.dpo_loss
    policy = tiny_model()
    info = train_with_label_noise(
        make_cfg(),
        policy,
        copy.deepcopy(policy),
        make_pairs(8),
        make_pairs(8, seed=1),
        PAD,
        tmp_path / "run",
        noise_fraction=0.25,
        noise_seed=0,
        label_smoothing=0.25,
    )
    assert info["loss_type"] == "robust" and info["label_smoothing"] == 0.25
    assert info["first_eval_loss"] == pytest.approx(LOG2, abs=1e-6)
    assert train_dpo.dpo_loss is original


def test_all_flipped_training_learns_the_opposite_preference(tmp_path: Path) -> None:
    # With every training label flipped, training on the pairs and scoring the SAME
    # pairs un-flipped must give a negative margin: the noise really reaches the loss.
    policy = tiny_model()
    pairs = make_pairs(8)
    info_dir = tmp_path / "run"
    train_with_label_noise(
        make_cfg(epochs=20),
        policy,
        copy.deepcopy(policy),
        pairs,
        pairs,
        PAD,
        info_dir,
        noise_fraction=1.0,
        noise_seed=0,
    )
    last = (info_dir / "eval.csv").read_text().strip().splitlines()[-1].split(",")
    header = (info_dir / "eval.csv").read_text().splitlines()[0].split(",")
    assert float(last[header.index("reward_margin")]) < 0
    assert float(last[header.index("accuracy")]) == 0.0


# --------------------------------------------------------------------------------------
# KL drift helper (tiny_rlhf.trackb.general.response_nll_and_kl, used by eval_drift.py)
# --------------------------------------------------------------------------------------


class FixedLogits:
    """Stands in for a model: returns log(probs[token]) as the logits at each position."""

    def __init__(self, probs: list[list[float]]) -> None:
        self.table = torch.tensor(probs, dtype=torch.float64).log()

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.table[input_ids])


def test_kl_of_a_model_with_itself_is_zero() -> None:
    general = pytest.importorskip("tiny_rlhf.trackb.general")
    model = FixedLogits([[0.5, 0.5], [0.9, 0.1]])
    pairs = [TokenizedPair([0, 1], [1, 0, 1], [0])]
    out = general.response_nll_and_kl(model, model, pairs, pad_token_id=0, max_length=None)
    assert out["kl"] == [pytest.approx(0.0, abs=1e-12)]
    assert out["nll_base"] == out["nll_trained"]


def test_kl_matches_a_two_token_example_computed_by_hand() -> None:
    general = pytest.importorskip("tiny_rlhf.trackb.general")
    # Vocabulary of two tokens. After token 0 both models predict (0.5, 0.5).
    # After token 1 the reference predicts (0.9, 0.1) and the policy (0.6, 0.4).
    reference = FixedLogits([[0.5, 0.5], [0.9, 0.1]])
    policy = FixedLogits([[0.5, 0.5], [0.6, 0.4]])
    # Sequence 0 1 | 1 0: prompt (0, 1), response (1, 0).
    #   response token at position 2 is predicted from position 1 (token 1): KL = k
    #   response token at position 3 is predicted from position 2 (token 1): KL = k
    # with k = 0.9*log(0.9/0.6) + 0.1*log(0.1/0.4) = 0.226288...
    k = 0.9 * math.log(0.9 / 0.6) + 0.1 * math.log(0.1 / 0.4)
    assert k == pytest.approx(0.2262884, abs=1e-6)
    pairs = [TokenizedPair([0, 1], [1, 0], [0])]
    out = general.response_nll_and_kl(reference, policy, pairs, pad_token_id=0, max_length=None)
    assert out["kl"] == [pytest.approx(k, abs=1e-6)]
    # Sequence 0 1 | 0 0: the second response token is predicted from token 0, where the
    # two models agree, so the mean over the two response positions is k / 2.
    pairs = [TokenizedPair([0, 1], [0, 0], [1])]
    out = general.response_nll_and_kl(reference, policy, pairs, pad_token_id=0, max_length=None)
    assert out["kl"] == [pytest.approx(k / 2, abs=1e-6)]
    # NLL of the response under each model: tokens (0, 0), predicted from tokens (1, 0).
    assert out["nll_base"] == [pytest.approx(-(math.log(0.9) + math.log(0.5)) / 2, abs=1e-6)]
    assert out["nll_trained"] == [pytest.approx(-(math.log(0.6) + math.log(0.5)) / 2, abs=1e-6)]
