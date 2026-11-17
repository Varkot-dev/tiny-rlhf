"""Numeric parity between the hand-written DPO code and TRL's DPOTrainer.

Everything runs offline on a tiny randomly initialised model with a toy tokenizer.
Three levels: one batch (loss and logged metrics), one backward pass (gradients),
and a short training run (weights and logged curves).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

pytest.importorskip("trl")

from datasets import Dataset  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # noqa: E402
from trl import DPOTrainer  # noqa: E402

from tiny_rlhf.data import PreferenceCollator  # noqa: E402
from tiny_rlhf.losses import dpo_loss  # noqa: E402
from tiny_rlhf.train_dpo import DPOConfig, pair_logps, train  # noqa: E402
from tiny_rlhf.train_trl import (  # noqa: E402
    build_trl_args,
    keep_rows_that_fit,
    tokenization_matches,
    train_with_trl,
)

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> "
    "{% endfor %}{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
)
SPECIALS = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>", "user", "assistant"]
WORDS = SPECIALS + [f"w{i}" for i in range(40)]
BETA = 0.5


def make_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<|im_end|>",
        chat_template=CHAT_TEMPLATE,
    )


def make_model(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
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


def make_rows(n: int, seed: int = 0) -> Dataset:
    """n conversations with random word content and responses of different lengths."""
    g = torch.Generator().manual_seed(seed)

    def words(k: int) -> str:
        return " ".join(f"w{i}" for i in torch.randint(0, 40, (k,), generator=g).tolist())

    rows = []
    for i in range(n):
        prompt = [{"role": "user", "content": words(3 + i % 3)}]
        rows.append(
            {
                "chosen": prompt + [{"role": "assistant", "content": words(2 + i % 4)}],
                "rejected": prompt + [{"role": "assistant", "content": words(2 + (i + 2) % 4)}],
            }
        )
    return Dataset.from_list(rows)


def make_cfg(**overrides: object) -> DPOConfig:
    values = {
        "run_name": "parity_test",
        "max_length": 32,
        "beta": BETA,
        "lr": 1e-3,
        "warmup_ratio": 0.25,
        "epochs": 2,
        "batch_size": 4,
        "grad_accum": 1,
        "eval_every": 3,
        "eval_batch_size": 4,
        "seed": 0,
        "shuffle": False,
        "use_lora": False,
        "device": "cpu",
        "save_adapter": False,
    }
    values.update(overrides)
    return DPOConfig(**values)


def make_trainer(cfg: DPOConfig, tmp_path: Path, n: int = 8):
    tokenizer = make_tokenizer()
    rows, pairs = keep_rows_that_fit(make_rows(n), tokenizer, cfg.max_length)
    policy = make_model()
    reference = make_model(seed=1)  # a different model, so rewards are not all zero
    trainer = DPOTrainer(
        model=policy,
        ref_model=reference,
        args=build_trl_args(cfg, len(pairs), tmp_path / "trl_output"),
        train_dataset=rows,
        eval_dataset=rows,
        processing_class=tokenizer,
    )
    return trainer, pairs, tokenizer


def test_trl_tokenizes_the_rows_exactly_as_we_do(tmp_path: Path) -> None:
    trainer, pairs, _ = make_trainer(make_cfg(), tmp_path)
    assert tokenization_matches(trainer.train_dataset, pairs)
    ours = PreferenceCollator(pad_token_id=0, max_length=32)(pairs[:4])
    theirs = trainer.data_collator([trainer.train_dataset[i] for i in range(4)])
    assert torch.equal(ours["input_ids"], theirs["input_ids"])
    assert torch.equal(ours["attention_mask"], theirs["attention_mask"])
    assert torch.equal(ours["response_mask"], theirs["completion_mask"].bool())


def test_same_batch_gives_the_same_loss_metrics_and_gradient(tmp_path: Path) -> None:
    trainer, pairs, _ = make_trainer(make_cfg(), tmp_path)
    our_policy = make_model()  # same seed as the trainer's policy: identical weights
    our_reference = make_model(seed=1)

    batch = PreferenceCollator(pad_token_id=0, max_length=32)(pairs[:4])
    pol_c, pol_r = pair_logps(our_policy, batch)
    with torch.no_grad():
        ref_c, ref_r = pair_logps(our_reference, batch)
    ours = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=BETA)
    ours.loss.backward()

    trainer.model.train()
    trl_batch = trainer.data_collator([trainer.train_dataset[i] for i in range(4)])
    trl_loss = trainer.compute_loss(trainer.model, trl_batch)
    trl_loss.backward()
    logged = {k: v[-1] for k, v in trainer._metrics["train"].items()}

    # (a) the loss and every logged quantity
    assert trl_loss.item() == pytest.approx(ours.loss.item(), abs=1e-5)
    assert logged["rewards/margins"] == pytest.approx(ours.reward_margin.mean().item(), abs=1e-5)
    assert logged["rewards/accuracies"] == pytest.approx(ours.accuracy.item(), abs=1e-9)
    assert logged["rewards/chosen"] == pytest.approx(ours.chosen_rewards.mean().item(), abs=1e-5)
    assert logged["rewards/rejected"] == pytest.approx(
        ours.rejected_rewards.mean().item(), abs=1e-5
    )
    assert logged["logps/chosen"] == pytest.approx(pol_c.mean().item(), abs=1e-4)
    assert logged["logps/rejected"] == pytest.approx(pol_r.mean().item(), abs=1e-4)
    assert abs(ours.reward_margin.mean().item()) > 1e-2  # the comparison is not trivially 0 = 0

    # (b) the gradient with respect to every policy parameter
    ours_grad = torch.cat([p.grad.flatten() for p in our_policy.parameters()])
    trl_grad = torch.cat([p.grad.flatten() for p in trainer.model.parameters()])
    cosine = torch.nn.functional.cosine_similarity(ours_grad, trl_grad, dim=0).item()
    relative = ((ours_grad - trl_grad).norm() / ours_grad.norm()).item()
    assert cosine > 1 - 1e-6
    assert relative < 1e-4


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def with_lora(model: LlamaForCausalLM) -> torch.nn.Module:
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)  # the same random LoRA A matrices on both sides
    config = LoraConfig(
        r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
    )
    return get_peft_model(model, config)


@pytest.mark.parametrize(
    ("grad_accum", "lora"), [(1, False), (2, False), (1, True)], ids=["full", "accum2", "lora"]
)
def test_training_runs_agree_step_by_step(tmp_path: Path, grad_accum: int, lora: bool) -> None:
    # 8 pairs, 2 epochs, dataset order. With grad_accum 2 the micro-batch is 2 pairs.
    # The lora case trains only a LoRA adapter, as the real runs do.
    cfg = make_cfg(batch_size=4 // grad_accum, grad_accum=grad_accum, lr=1e-2 if lora else 1e-3)
    tokenizer = make_tokenizer()
    rows, pairs = keep_rows_that_fit(make_rows(8), tokenizer, cfg.max_length)

    ours_ref = make_model()
    ours_policy = with_lora(make_model()) if lora else make_model()
    train(cfg, ours_policy, ours_ref, pairs, pairs, 0, tmp_path / "ours")

    trl_policy = with_lora(make_model()) if lora else make_model()
    info = train_with_trl(
        cfg,
        trl_policy,
        make_model(),
        tokenizer,
        rows,
        rows,
        pairs,
        pairs,
        tmp_path / "trl",
    )
    assert info["tokenization_matches_handwritten"]
    assert info["optimizer_steps"] == 4

    # The logged curves.
    for name in ("train.csv", "eval.csv"):
        ours_rows, trl_rows = (
            read_rows(tmp_path / "ours" / name),
            read_rows(tmp_path / "trl" / name),
        )
        assert [r["step"] for r in ours_rows] == [r["step"] for r in trl_rows]
        for a, b in zip(ours_rows, trl_rows, strict=True):
            for key in ("loss", "reward_margin", "chosen_reward", "rejected_reward"):
                assert float(a[key]) == pytest.approx(float(b[key]), abs=TRAIN_TOL), (name, key)
            assert float(a["accuracy"]) == pytest.approx(float(b["accuracy"]), abs=1e-9)
            assert float(a["chosen_logp"]) == pytest.approx(float(b["chosen_logp"]), abs=1e-4)
            if name == "train.csv":
                assert float(a["lr"]) == pytest.approx(float(b["lr"]), rel=1e-6, abs=1e-12)
                assert float(a["grad_norm"]) == pytest.approx(float(b["grad_norm"]), rel=1e-4)
    assert float(read_rows(tmp_path / "trl" / "eval.csv")[0]["loss"]) == pytest.approx(
        0.6931472, abs=1e-6
    )

    # (c) the trained weights.
    start = with_lora(make_model()) if lora else make_model()
    ours_w, trl_w, start_w = (
        [p for p in m.parameters() if p.requires_grad or not lora]
        for m in (ours_policy, trl_policy, start)
    )
    if lora:  # requires_grad is set on the adapter weights only
        start_w = [p for n, p in start.named_parameters() if "lora_" in n]
        assert len(ours_w) == len(trl_w) == len(start_w) > 0
    worst = max((a - b).abs().max().item() for a, b in zip(ours_w, trl_w, strict=True))
    moved = max((a - b).abs().max().item() for a, b in zip(ours_w, start_w, strict=True))
    assert moved > 1e-3  # training really changed the weights
    assert worst < WEIGHT_TOL

    # TRL's final eval, recomputed with the hand-written evaluate(), is the same number.
    for key, value in info["final_eval_by_trl"].items():
        assert info["final_eval_recomputed_by_handwritten_evaluate"][key] == pytest.approx(
            value, abs=1e-4
        )


# Measured on CPU in float32: logged values differ by at most 3e-7 and trained weights by
# at most 2e-8, while training moves the weights by about 2e-3. The two implementations
# compute the same numbers in a different order of operations (TRL scores only the
# response positions through its fused LM head), so bit equality is not expected.
TRAIN_TOL = 1e-5
WEIGHT_TOL = 1e-6
