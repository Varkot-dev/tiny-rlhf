"""Tests for the hand-written GRPO pieces (tiny_rlhf.grpo).

Two kinds:
- hand-computed values: group advantages on small reward lists, and the gradient
  identities the docstrings claim (REINFORCE with a baseline, the KL term's gradient);
- parity with TRL 1.18's GRPOTrainer, offline on a tiny random Llama with LoRA: a real
  generation batch is built by TRL, then TRL's own loss and ours are computed from it and
  their values and gradients compared (within 1e-6), for every aggregation and with and
  without the KL term. The advantages TRL computes from the same rewards are compared with
  group_advantages, and TRL's reference log-probs with ours.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from tiny_rlhf.grpo import completion_logps, group_advantages, grpo_loss, nanstd

# --------------------------------------------------------------------------------------
# Advantages, by hand
# --------------------------------------------------------------------------------------


def test_one_success_in_a_group_of_four() -> None:
    # mean 0.25; sample std sqrt((0.75^2 + 3 * 0.25^2) / 3) = sqrt(0.25) = 0.5
    adv, all_equal = group_advantages(torch.tensor([1.0, 0.0, 0.0, 0.0]), group_size=4)
    expected = torch.tensor([0.75, -0.25, -0.25, -0.25]) / (0.5 + 1e-4)
    torch.testing.assert_close(adv, expected)
    assert adv[0].item() == pytest.approx(1.49970, abs=1e-5)
    assert adv[1].item() == pytest.approx(-0.49990, abs=1e-5)
    assert adv.sum().abs().item() < 1e-6  # advantages of a group always sum to zero
    assert not all_equal.any()


def test_all_equal_rewards_teach_nothing() -> None:
    for scale in ("group", "batch", "none"):
        adv, all_equal = group_advantages(torch.full((8,), 0.75), group_size=4, scale=scale)
        assert torch.equal(adv, torch.zeros(8)) and all_equal.all()
    # The flag is TRL's isclose(std, 0) with atol 1e-8, so it relies on an exact zero. The
    # Track B reward takes only multiples of 0.25, whose sums and means are exact in float32.


def test_mixed_batch_flags_only_the_all_equal_group() -> None:
    rewards = torch.tensor([1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    adv, all_equal = group_advantages(rewards, group_size=4)
    # first group: mean 0, std sqrt(2 / 3) = 0.8165
    assert adv[0].item() == pytest.approx(1 / (math.sqrt(2 / 3) + 1e-4), abs=1e-6)
    assert adv[0].item() == pytest.approx(1.2246, abs=1e-4)
    assert torch.equal(adv[4:], torch.zeros(4))
    assert all_equal.tolist() == [False] * 4 + [True] * 4
    assert all_equal.float().mean().item() == 0.5  # TRL's frac_reward_zero_std


def test_batch_and_no_scaling() -> None:
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    centred = torch.tensor([0.75, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, -0.75])
    adv, _ = group_advantages(rewards, group_size=4, scale="none")
    torch.testing.assert_close(adv, centred)
    adv, _ = group_advantages(rewards, group_size=4, scale="batch")
    batch_std = torch.tensor([1.0, 0, 0, 0, 1, 1, 1, 0]).std()  # unbiased, over all 8
    torch.testing.assert_close(adv, centred / (batch_std + 1e-4))


def test_unscorable_completion_is_left_out_of_the_baseline_and_gets_zero() -> None:
    rewards = torch.tensor([1.0, float("nan"), 0.0, 0.0])
    adv, _ = group_advantages(rewards, group_size=4)
    # The group is effectively (1, 0, 0): mean 1/3, std sqrt(1/3).
    std = math.sqrt(((2 / 3) ** 2 + 2 * (1 / 3) ** 2) / 2)
    assert adv[0].item() == pytest.approx((2 / 3) / (std + 1e-4), abs=1e-6)
    assert adv[1].item() == 0.0


def test_nanstd_matches_torch_std_and_needs_two_values() -> None:
    x = torch.tensor([[1.0, 2.0, 4.0], [3.0, 3.0, 3.0]])
    torch.testing.assert_close(nanstd(x, dim=1), x.std(dim=1))
    assert math.isnan(nanstd(torch.tensor([2.0])).item())


def test_bad_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="groups of 3"):
        group_advantages(torch.zeros(4), group_size=3)
    with pytest.raises(ValueError, match="scale"):
        group_advantages(torch.zeros(4), group_size=2, scale="rank")
    logps = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="ref_logps"):
        grpo_loss(logps, torch.zeros(2), torch.ones(2, 3), beta=0.1)
    with pytest.raises(ValueError, match="loss_type"):
        grpo_loss(logps, torch.zeros(2), torch.ones(2, 3), loss_type="ppo")
    with pytest.raises(ValueError, match="max_completion_length"):
        grpo_loss(logps, torch.zeros(2), torch.ones(2, 3), loss_type="dr_grpo")


# --------------------------------------------------------------------------------------
# The loss, by hand
# --------------------------------------------------------------------------------------

MASK = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
ADV = torch.tensor([1.5, -0.5, -1.0])


def leaf(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (-torch.rand(3, 3, generator=g) * 3).requires_grad_()


def test_one_step_per_batch_is_reinforce_with_the_group_baseline() -> None:
    # With old_logps = logps.detach() the ratio is 1, so the gradient with respect to each
    # token's log-prob is -A_i / N (N = 6 completion tokens): REINFORCE, token-averaged.
    logps = leaf(0)
    out = grpo_loss(logps, ADV, MASK, loss_type="dapo")
    out.loss.backward()
    torch.testing.assert_close(logps.grad, -ADV[:, None] * MASK / MASK.sum())
    # Its value is -(sum of A_i times length_i) / N, not a measure of progress.
    assert out.loss.item() == pytest.approx(-(1.5 * 3 - 0.5 * 1 - 1.0 * 2) / 6)
    assert out.clip_ratio.item() == 0.0
    # "grpo" averages per completion first: gradient -A_i / (B * length_i).
    logps = leaf(0)
    grpo_loss(logps, ADV, MASK, loss_type="grpo").loss.backward()
    lengths = MASK.sum(-1, keepdim=True)
    torch.testing.assert_close(logps.grad, -ADV[:, None] * MASK / (3 * lengths))
    # "dr_grpo" divides by B * max_completion_length whatever the lengths.
    logps = leaf(0)
    grpo_loss(logps, ADV, MASK, loss_type="dr_grpo", max_completion_length=4).loss.backward()
    torch.testing.assert_close(logps.grad, -ADV[:, None] * MASK / (3 * 4))


def test_kl_term_gradient_is_the_kl_penalised_return() -> None:
    # d/dlogp of beta * rho * k3 at rho = 1 is beta * (logp - ref).
    logps, ref = leaf(0), leaf(1).detach()
    beta = 0.3
    out = grpo_loss(logps, torch.zeros(3), MASK, ref_logps=ref, beta=beta, loss_type="dapo")
    out.loss.backward()
    expected = beta * (logps.detach() - ref) * MASK / MASK.sum()
    torch.testing.assert_close(logps.grad, expected)
    diff = ref - logps.detach()
    k3 = torch.exp(diff) - diff - 1
    assert (k3 >= 0).all()
    assert out.kl.item() == pytest.approx(((k3 * MASK).sum() / MASK.sum()).item(), rel=1e-6)


def test_clipping_stops_the_push_once_the_ratio_leaves_the_range() -> None:
    logps = leaf(0)
    # Token (0, 0) is already 1.5 times as likely as when sampled, with A > 0: clipped.
    old = logps.detach().clone()
    old[0, 0] -= math.log(1.5)
    out = grpo_loss(logps, ADV, MASK, old_logps=old, epsilon=0.2)
    out.loss.backward()
    assert logps.grad[0, 0].item() == 0.0
    assert logps.grad[0, 1].item() == pytest.approx(-1.5 / 6)
    assert out.clip_ratio.item() == pytest.approx(1 / 6)


# --------------------------------------------------------------------------------------
# Parity with TRL's GRPOTrainer
# --------------------------------------------------------------------------------------

G = 4  # completions per prompt
N_PROMPTS = 4
MAX_COMPLETION = 6
TEMPERATURE = 0.7
REWARDS: list[float | None] = [1.0, 0, 0, 0, 1, 1, 1, 1, 0.5, -1, 0, 1, 0, 0, 1, -1]


# The same toy tokenizer and model as tests/test_trl_parity.py.
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> "
    "{% endfor %}{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
)
WORDS = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>", "user", "assistant"] + [
    f"w{i}" for i in range(40)
]


def make_tokenizer() -> Any:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<|im_end|>",
        chat_template=CHAT_TEMPLATE,
    )


def make_model() -> Any:
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(WORDS),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
    )
    return LlamaForCausalLM(config)


def _trl_parts() -> tuple[Any, ...]:
    pytest.importorskip("trl")
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    return Dataset, LoraConfig, GRPOConfig, GRPOTrainer


def prompt_rows() -> list[dict[str, Any]]:
    return [
        {
            "prompt": [
                {"role": "user", "content": " ".join(f"w{(7 * i + j) % 40}" for j in range(3 + i))}
            ],
            "id": f"p{i}",
        }
        for i in range(N_PROMPTS)
    ]


def make_trainer(tmp_path: Path, rewards: list[float | None] | None = None, **overrides: Any):
    Dataset, LoraConfig, GRPOConfig, GRPOTrainer = _trl_parts()
    values: dict[str, Any] = {
        "output_dir": str(tmp_path / "trl"),
        "per_device_train_batch_size": G * N_PROMPTS,  # one micro-batch = one generation batch
        "gradient_accumulation_steps": 1,
        "num_generations": G,
        "max_completion_length": MAX_COMPLETION,
        "temperature": TEMPERATURE,
        "beta": 0.0,
        "report_to": [],
        "use_cpu": True,
        "bf16": False,
        "seed": 0,
        "gradient_checkpointing": False,
    }
    values.update(overrides)

    def reward(prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float | None]:
        return list(rewards if rewards is not None else REWARDS)[: len(completions)]

    trainer = GRPOTrainer(
        model=make_model(),
        reward_funcs=reward,
        args=GRPOConfig(**values),
        train_dataset=Dataset.from_list(prompt_rows()),
        processing_class=make_tokenizer(),
        peft_config=LoraConfig(
            r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], lora_dropout=0.0
        ),
    )
    # A fresh LoRA adapter is the identity (B = 0), so the policy would equal the reference
    # and the KL would be 0. Move it away, reproducibly.
    g = torch.Generator().manual_seed(0)
    for name, p in trainer.model.named_parameters():
        if "lora_B" in name:
            p.data = torch.randn(p.shape, generator=g) * 0.3
    trainer.model.train()
    return trainer


def generation_batch(trainer: Any) -> dict[str, Any]:
    """A real batch from TRL (prompts repeated G times in a row, as its sampler does), with
    completions cut to different lengths so masking is exercised."""
    rows = prompt_rows()
    inputs = trainer._generate_and_score_completions([r for r in rows for _ in range(G)])
    mask = inputs["completion_mask"].clone()
    for i in range(mask.shape[0]):
        mask[i, 1 + i % MAX_COMPLETION :] = 0
    inputs["completion_mask"] = mask
    inputs["num_items_in_batch"] = mask.sum()
    return inputs


def our_loss(trainer: Any, inputs: dict[str, Any], loss_type: str, beta: float):
    logps = completion_logps(
        trainer.model,
        inputs["prompt_ids"],
        inputs["prompt_mask"],
        inputs["completion_ids"],
        inputs["completion_mask"],
        temperature=TEMPERATURE,
    )
    return grpo_loss(
        logps,
        inputs["advantages"],
        inputs["completion_mask"],
        ref_logps=inputs.get("ref_per_token_logps"),
        beta=beta,
        loss_type=loss_type,
        max_completion_length=MAX_COMPLETION,
    )


def loss_and_grads(model: Any, compute) -> tuple[float, list[torch.Tensor]]:
    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad()
    loss = compute()
    loss.backward()
    return loss.item(), [p.grad.detach().clone() for p in params]


@pytest.mark.parametrize("scale", ["group", "batch", "none"])
def test_advantages_match_trl(tmp_path: Path, scale: str) -> None:
    trainer = make_trainer(tmp_path, scale_rewards=scale)
    inputs = trainer._generate_and_score_completions([r for r in prompt_rows() for _ in range(G)])
    ours, all_equal = group_advantages(torch.tensor(REWARDS), G, scale=scale)
    torch.testing.assert_close(inputs["advantages"], ours, atol=1e-6, rtol=0)
    logged = trainer._metrics["train"]["frac_reward_zero_std"][-1]
    assert logged == pytest.approx(all_equal.float().mean().item())


def test_unscorable_completion_matches_trl(tmp_path: Path) -> None:
    rewards: list[float | None] = list(REWARDS)
    rewards[1] = None  # TRL turns None into NaN
    trainer = make_trainer(tmp_path, rewards=rewards)
    inputs = trainer._generate_and_score_completions([r for r in prompt_rows() for _ in range(G)])
    tensor = torch.tensor([float("nan") if r is None else r for r in rewards])
    ours, _ = group_advantages(tensor, G)
    torch.testing.assert_close(inputs["advantages"], ours, atol=1e-6, rtol=0)
    assert inputs["advantages"][1].item() == 0.0


@pytest.mark.parametrize("beta", [0.0, 0.04])
@pytest.mark.parametrize("loss_type", ["dapo", "grpo", "bnpo", "dr_grpo"])
def test_loss_and_gradients_match_trl(tmp_path: Path, loss_type: str, beta: float) -> None:
    trainer = make_trainer(tmp_path, loss_type=loss_type, beta=beta)
    inputs = generation_batch(trainer)
    trainer.current_gradient_accumulation_steps = 1
    model = trainer.model
    trl_loss, trl_grads = loss_and_grads(model, lambda: trainer._compute_loss(model, inputs))
    ours = {}
    our_value, our_grads = loss_and_grads(
        model, lambda: ours.setdefault("out", our_loss(trainer, inputs, loss_type, beta)).loss
    )
    assert our_value == pytest.approx(trl_loss, abs=1e-6)
    assert any(g.abs().max() > 1e-4 for g in trl_grads)  # the comparison is not of zeros
    for theirs, mine in zip(trl_grads, our_grads, strict=True):
        torch.testing.assert_close(mine, theirs, atol=1e-6, rtol=0)
    if beta:
        assert ours["out"].kl.item() > 1e-3
        logged_kl = trainer._metrics["train"]["kl"][-1]
        assert ours["out"].kl.item() == pytest.approx(logged_kl, abs=1e-6)


def test_reference_logps_are_the_model_with_its_adapter_switched_off(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, beta=0.04)
    inputs = generation_batch(trainer)
    with trainer.model.disable_adapter():
        ref = completion_logps(
            trainer.model,
            inputs["prompt_ids"],
            inputs["prompt_mask"],
            inputs["completion_ids"],
            inputs["completion_mask"],
            temperature=TEMPERATURE,
        )
    mask = inputs["completion_mask"].bool()
    torch.testing.assert_close(ref[mask], inputs["ref_per_token_logps"][mask], atol=1e-6, rtol=0)


def test_dapo_accumulation_adds_up_to_the_whole_batch_loss(tmp_path: Path) -> None:
    # TRL splits the generation batch into micro-batches and divides each by the token count
    # of the WHOLE generation batch, so the accumulated loss and gradient are ours.
    trainer = make_trainer(
        tmp_path,
        per_device_train_batch_size=G * N_PROMPTS // 2,
        gradient_accumulation_steps=2,
        beta=0.04,
    )
    assert trainer.args.steps_per_generation == 2
    inputs = generation_batch(trainer)
    trainer.current_gradient_accumulation_steps = 2
    model = trainer.model
    half = G * N_PROMPTS // 2

    def part(lo: int, hi: int) -> dict[str, Any]:
        return {k: v[lo:hi] if torch.is_tensor(v) and v.dim() > 0 else v for k, v in inputs.items()}

    def accumulated() -> torch.Tensor:
        return trainer._compute_loss(model, part(0, half)) + trainer._compute_loss(
            model, part(half, 2 * half)
        )

    trl_loss, trl_grads = loss_and_grads(model, accumulated)
    our_value, our_grads = loss_and_grads(
        model, lambda: our_loss(trainer, inputs, "dapo", 0.04).loss
    )
    assert our_value == pytest.approx(trl_loss, abs=1e-6)
    for theirs, mine in zip(trl_grads, our_grads, strict=True):
        torch.testing.assert_close(mine, theirs, atol=1e-6, rtol=0)


def test_trl_tokenizes_prompts_exactly_as_the_dpo_runs(tmp_path: Path) -> None:
    from tiny_rlhf.data import tokenize_pair

    trainer = make_trainer(tmp_path)
    inputs = trainer._generate_and_score_completions([r for r in prompt_rows() for _ in range(G)])
    for i, row in enumerate(prompt_rows()):
        conversation = row["prompt"]
        pair = tokenize_pair(
            trainer.processing_class,
            conversation + [{"role": "assistant", "content": "w1"}],
            conversation + [{"role": "assistant", "content": "w2"}],
        )
        k = i * G
        trl_ids = inputs["prompt_ids"][k][inputs["prompt_mask"][k].bool()].tolist()
        assert trl_ids == pair.prompt_ids
