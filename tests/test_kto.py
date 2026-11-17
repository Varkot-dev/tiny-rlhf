"""Tests for the hand-written KTO (tiny_rlhf.kto), including parity with TRL's KTOTrainer.

Everything is offline and runs on tiny randomly initialised models.
"""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from tiny_rlhf.data import TokenizedPair
from tiny_rlhf.kto import (
    EVAL_COLUMNS,
    TRAIN_COLUMNS,
    KTOConfig,
    UnpairedCollator,
    UnpairedExample,
    add_kl_completions,
    answer_logps,
    kto_loss,
    load_kto_config,
    plan_steps,
    step0_loss,
    train,
    unpair,
)

PAD = 0
VOCAB = 50
ROOT = Path(__file__).resolve().parents[1]


def sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def t(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


# --------------------------------------------------------------------------------------
# The loss on hand-computed numbers
# --------------------------------------------------------------------------------------


def test_kto_loss_hand_computed_with_a_positive_reference_point() -> None:
    # One good answer raised by 2 nats, one bad answer lowered by 4 nats. The mismatched
    # answers moved by +1 and 0 nats, so the reference point is z0 = mean(1, 0) = 0.5.
    beta = 0.5
    out = kto_loss(
        policy_logps=t(-8.0, -14.0),
        ref_logps=t(-10.0, -10.0),
        labels=torch.tensor([True, False]),
        kl_policy_logps=t(-5.0, -6.0),
        kl_ref_logps=t(-6.0, -6.0),
        beta=beta,
    )
    good = 1 - sig(beta * (2.0 - 0.5))  # 1 - sigmoid(0.75)   = 0.32082...
    bad = 1 - sig(beta * (0.5 - (-4.0)))  # 1 - sigmoid(2.25) = 0.09534...
    assert out.kl.item() == pytest.approx(0.5)
    assert out.losses.tolist() == pytest.approx([good, bad])
    assert out.loss.item() == pytest.approx((good + bad) / 2)
    assert out.loss.item() == pytest.approx(0.2080854, abs=1e-6)
    assert out.chosen_rewards.tolist() == pytest.approx([1.0])  # beta * 2
    assert out.rejected_rewards.tolist() == pytest.approx([-2.0])  # beta * -4


def test_a_negative_kl_estimate_is_clamped_to_zero() -> None:
    out = kto_loss(t(-8.0), t(-10.0), torch.tensor([True]), t(-9.0, -9.0), t(-6.0, -6.0), beta=0.5)
    assert out.kl.item() == 0.0
    assert out.loss.item() == pytest.approx(1 - sig(0.5 * 2.0))


def test_weights_scale_their_own_answers_only() -> None:
    args = (t(-8.0, -14.0), t(-10.0, -10.0), torch.tensor([True, False]), t(-6.0), t(-6.0))
    plain = kto_loss(*args, beta=0.5)
    weighted = kto_loss(*args, beta=0.5, desirable_weight=2.0, undesirable_weight=1.0)
    assert weighted.losses[0].item() == pytest.approx(2 * plain.losses[0].item())
    assert weighted.losses[1].item() == pytest.approx(plain.losses[1].item())


@pytest.mark.parametrize("beta", [0.01, 0.1, 0.5, 5.0])
def test_step_zero_loss_is_one_half(beta: float) -> None:
    logps = t(-12.0, -30.0, -7.0, -19.0)
    labels = torch.tensor([True, False, True, False])
    out = kto_loss(logps, logps, labels, logps, logps, beta=beta)
    assert out.kl.item() == 0.0
    assert out.loss.item() == pytest.approx(0.5, abs=1e-12)
    assert step0_loss(1.0, 1.0, 2, 2) == 0.5
    weighted = kto_loss(logps, logps, labels, logps, logps, beta=beta, desirable_weight=2.0)
    assert weighted.loss.item() == pytest.approx(step0_loss(2.0, 1.0, 2, 2)) == 0.75


def test_gradients_raise_good_answers_lower_bad_ones_and_skip_the_reference_point() -> None:
    pol = t(-8.0, -14.0).requires_grad_(True)
    kl_pol = t(-5.0, -6.0).requires_grad_(True)
    out = kto_loss(pol, t(-10.0, -10.0), torch.tensor([True, False]), kl_pol, t(-6.0, -6.0), 0.5)
    out.loss.backward()
    assert pol.grad[0] < 0  # gradient descent raises the good answer's log-probability
    assert pol.grad[1] > 0  # and lowers the bad one's
    assert kl_pol.grad is None  # z0 is detached: it only sets the bar
    # Exact values: d/dr of (1 - sigmoid(beta (r - z0))) / 2 = -beta s (1 - s) / 2
    s = sig(0.5 * 1.5)
    assert pol.grad[0].item() == pytest.approx(-0.5 * s * (1 - s) / 2)


def test_kto_loss_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        kto_loss(t(1.0, 2.0), t(1.0), torch.tensor([True, False]), t(0.0), t(0.0))


# --------------------------------------------------------------------------------------
# From pairs to single answers
# --------------------------------------------------------------------------------------


def make_pairs(n: int, seed: int = 0) -> list[TokenizedPair]:
    g = torch.Generator().manual_seed(seed)

    def ids(length: int) -> list[int]:
        return torch.randint(3, VOCAB, (length,), generator=g).tolist()

    return [
        TokenizedPair(
            prompt_ids=ids(3 + i % 3), chosen_ids=ids(2 + i % 4), rejected_ids=ids(2 + (i + 2) % 4)
        )
        for i in range(n)
    ]


def test_unpair_keeps_every_answer_once_with_the_right_label() -> None:
    pairs = make_pairs(25)
    examples, counts = unpair(pairs, seed=0, batch_size=4)
    assert counts["n_examples"] == 50 and counts["n_desirable"] == counts["n_undesirable"] == 25
    for i, p in enumerate(pairs):
        mine = [e for e in examples if e.pair_index == i]
        assert sorted(e.label for e in mine) == [False, True]
        good = next(e for e in mine if e.label)
        bad = next(e for e in mine if not e.label)
        assert good.completion_ids == p.chosen_ids and bad.completion_ids == p.rejected_ids
        assert good.prompt_ids == bad.prompt_ids == p.prompt_ids


def test_unpair_keeps_the_halves_of_a_pair_out_of_the_same_micro_batch() -> None:
    examples, counts = unpair(make_pairs(200), seed=3, batch_size=4)
    clashes = 0
    for start in range(0, len(examples), 4):
        block = [e.pair_index for e in examples[start : start + 4]]
        clashes += len(block) - len(set(block))
    assert clashes == counts["micro_batches_holding_both_halves_of_a_pair"] == 0


def test_unpair_is_deterministic_and_depends_on_the_seed() -> None:
    pairs = make_pairs(30)
    a = [(e.pair_index, e.label) for e in unpair(pairs, 0, 4)[0]]
    b = [(e.pair_index, e.label) for e in unpair(pairs, 0, 4)[0]]
    c = [(e.pair_index, e.label) for e in unpair(pairs, 1, 4)[0]]
    assert a == b and a != c


def test_kl_completions_rotate_within_each_micro_batch() -> None:
    ex = [UnpairedExample([1], [10 + i], i % 2 == 0, i) for i in range(5)]
    out = add_kl_completions(ex, batch_size=2)
    assert [e.kl_completion_ids for e in out] == [[11], [10], [13], [12], [14]]
    out3 = add_kl_completions(ex, batch_size=3)
    # (x1,y1),(x2,y2),(x3,y3) -> mismatched answers y3, y1, y2, as in TRL
    assert [e.kl_completion_ids for e in out3] == [[12], [10], [11], [14], [13]]


def test_config_requires_two_answers_per_micro_batch(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("batch_size: 1\n")
    with pytest.raises(ValueError, match="at least 2"):
        load_kto_config(path)
    for name in ("trackb_kto", "trackb_kto_beta0.5", "trackb_kto_tiny"):
        load_kto_config(ROOT / "configs" / f"{name}.yaml")


def test_full_configs_see_the_same_answers_per_step_as_the_dpo_run() -> None:
    import yaml

    dpo = yaml.safe_load((ROOT / "configs/trackb_full.yaml").read_text())
    for name in ("trackb_kto", "trackb_kto_beta0.5"):
        kto = load_kto_config(ROOT / "configs" / f"{name}.yaml")
        assert kto.batch_size * kto.grad_accum == 2 * dpo["batch_size"] * dpo["grad_accum"]
        assert plan_steps(2 * kto.n_train, kto)[1] == 375
        for key in ("model_name", "dataset_name", "n_train", "n_eval", "max_length", "lr", "seed"):
            assert getattr(kto, key) == dpo[key], key
        for key in ("lora_r", "lora_alpha", "lora_dropout", "lora_target_modules"):
            assert getattr(kto, key) == dpo[key], key
    assert load_kto_config(ROOT / "configs/trackb_kto.yaml").beta == dpo["beta"] == 0.1
    assert load_kto_config(ROOT / "configs/trackb_kto_beta0.5.yaml").beta == 0.5


# --------------------------------------------------------------------------------------
# The training loop on a tiny model
# --------------------------------------------------------------------------------------


def tiny_model(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
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
    return LlamaForCausalLM(config)


def make_cfg(**overrides: object) -> KTOConfig:
    values = {
        "run_name": "kto_test",
        "max_length": 32,
        "beta": 0.5,
        "lr": 1e-2,
        "warmup_ratio": 0.0,
        "epochs": 20,
        "batch_size": 4,
        "grad_accum": 1,
        "eval_every": 1000,
        "eval_batch_size": 3,
        "seed": 0,
        "use_lora": False,
        "device": "cpu",
        "save_adapter": False,
    }
    values.update(overrides)
    return KTOConfig(**values)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def test_training_starts_at_one_half_and_learns_the_labels(tmp_path: Path) -> None:
    policy = tiny_model()
    reference = copy.deepcopy(policy)
    ref_state = {k: v.clone() for k, v in reference.state_dict().items()}
    pairs = make_pairs(8)
    info = train(make_cfg(), policy, reference, pairs, pairs, PAD, tmp_path / "run")
    assert info["first_eval_loss"] == pytest.approx(0.5, abs=1e-6)
    assert info["unpaired"]["n_examples"] == 16
    header, evals = read_csv(tmp_path / "run" / "eval.csv")
    assert tuple(header) == EVAL_COLUMNS
    assert float(evals[0]["accuracy"]) == 0.0 and float(evals[0]["kl"]) == 0.0
    assert float(evals[-1]["loss"]) < 0.5
    assert float(evals[-1]["accuracy"]) >= 0.75  # most held-out (here: training) pairs ranked right
    assert float(evals[-1]["chosen_reward"]) > 0 > float(evals[-1]["rejected_reward"])
    train_header, rows = read_csv(tmp_path / "run" / "train.csv")
    assert tuple(train_header) == TRAIN_COLUMNS
    assert all(int(r["n_desirable"]) + int(r["n_undesirable"]) == 4 for r in rows)
    for k, v in reference.state_dict().items():
        assert torch.equal(v, ref_state[k]), k  # the reference never moves


def test_same_seed_gives_identical_training(tmp_path: Path) -> None:
    logs = []
    for name in ("a", "b"):
        policy = tiny_model()
        cfg = make_cfg(epochs=2)
        train(
            cfg, policy, copy.deepcopy(policy), make_pairs(8), make_pairs(8), PAD, tmp_path / name
        )
        _, rows = read_csv(tmp_path / name / "train.csv")
        logs.append([{k: v for k, v in r.items() if k != "elapsed_s"} for r in rows])
    assert logs[0] == logs[1]


def test_step_zero_guard_rejects_a_reference_that_differs(tmp_path: Path) -> None:
    pairs = make_pairs(8)
    with pytest.raises(RuntimeError, match="step-0 KTO loss"):
        train(make_cfg(epochs=1), tiny_model(0), tiny_model(1), pairs, pairs, PAD, tmp_path / "r")


# --------------------------------------------------------------------------------------
# Parity with TRL 1.18.0's KTOTrainer
# --------------------------------------------------------------------------------------

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> "
    "{% endfor %}{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
)
SPECIALS = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>", "user", "assistant"]
WORDS = SPECIALS + [f"w{i}" for i in range(40)]


def parity_tokenizer():
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


def parity_model(seed: int = 0) -> LlamaForCausalLM:
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


def parity_conversations(n: int, seed: int = 0) -> list[tuple[list[dict], list[dict]]]:
    g = torch.Generator().manual_seed(seed)

    def words(k: int) -> str:
        return " ".join(f"w{i}" for i in torch.randint(0, 40, (k,), generator=g).tolist())

    out = []
    for i in range(n):
        prompt = [{"role": "user", "content": words(3 + i % 3)}]
        out.append(
            (
                prompt + [{"role": "assistant", "content": words(2 + i % 4)}],
                prompt + [{"role": "assistant", "content": words(2 + (i + 2) % 4)}],
            )
        )
    return out


def parity_setup(cfg: KTOConfig, n_pairs: int, tmp_path: Path, policy, reference):
    """Our unpaired order, the same rows for TRL, and a KTOTrainer built offline."""
    trl = pytest.importorskip("trl")
    from datasets import Dataset

    from tiny_rlhf.data import tokenize_pair

    tokenizer = parity_tokenizer()
    convs = parity_conversations(n_pairs)
    pairs = [tokenize_pair(tokenizer, c, r) for c, r in convs]
    examples, _ = unpair(pairs, cfg.seed, cfg.batch_size)
    examples = add_kl_completions(examples, cfg.batch_size)
    # TRL gets the answers already split, in our order: it then skips its own unpairing.
    rows = []
    for e in examples:
        chosen, rejected = convs[e.pair_index]
        conversation = chosen if e.label else rejected
        rows.append(
            {"prompt": conversation[:-1], "completion": conversation[-1:], "label": e.label}
        )
    _, total = plan_steps(len(examples), cfg)
    args = trl.KTOConfig(
        output_dir=str(tmp_path / "trl_output"),
        beta=cfg.beta,
        desirable_weight=cfg.desirable_weight,
        undesirable_weight=cfg.undesirable_weight,
        max_length=cfg.max_length,
        learning_rate=cfg.lr,
        lr_scheduler_type="linear",
        warmup_steps=math.ceil(cfg.warmup_ratio * total),
        weight_decay=cfg.weight_decay,
        optim="adamw_torch",
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        max_grad_norm=cfg.max_grad_norm,
        num_train_epochs=cfg.epochs,
        fp16=False,
        bf16=False,
        disable_dropout=False,
        gradient_checkpointing=False,
        train_sampling_strategy="sequential",
        seed=cfg.seed,
        data_seed=cfg.seed,
        use_cpu=True,
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        dataloader_num_workers=0,
    )
    trainer = trl.KTOTrainer(
        model=policy,
        ref_model=reference,
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
    )
    return trainer, pairs, examples


def test_trl_builds_the_same_tokens_and_mismatched_answers(tmp_path: Path) -> None:
    cfg = make_cfg(max_length=32, batch_size=4, epochs=1)
    trainer, _, examples = parity_setup(cfg, 6, tmp_path, parity_model(), parity_model())
    data = trainer.train_dataset
    assert len(data) == len(examples) == 12
    for i, e in enumerate(examples):
        assert data[i]["prompt_ids"] == e.prompt_ids
        assert data[i]["completion_ids"] == e.completion_ids
        assert data[i]["KL_completion_ids"] == e.kl_completion_ids
        assert bool(data[i]["label"]) == e.label
    ours = UnpairedCollator(PAD, 32)(examples[:4])
    theirs = trainer.data_collator([data[i] for i in range(4)])
    for key in ("input_ids", "attention_mask"):
        assert torch.equal(ours[key], theirs[key]) and torch.equal(
            ours[f"kl_{key}"], theirs[f"KL_{key}"]
        )
    assert torch.equal(ours["response_mask"], theirs["completion_mask"].bool())
    assert torch.equal(ours["kl_response_mask"], theirs["KL_completion_mask"].bool())
    assert torch.equal(ours["labels"], theirs["label"].bool())


def test_same_batch_gives_the_same_loss_reference_point_and_gradient(tmp_path: Path) -> None:
    cfg = make_cfg(max_length=32, batch_size=4, epochs=1, beta=0.5)
    # Different policy and reference, so the rewards and the reference point are not 0.
    trainer, _, examples = parity_setup(cfg, 6, tmp_path, parity_model(0), parity_model(3))
    our_policy, our_reference = parity_model(0), parity_model(3)

    batch = UnpairedCollator(PAD, 32)(examples[:4])
    with torch.no_grad():
        ref = answer_logps(our_reference, batch)
        ref_kl = answer_logps(our_reference, batch, "kl_")
        pol_kl = answer_logps(our_policy, batch, "kl_")
    pol = answer_logps(our_policy, batch)
    ours = kto_loss(pol, ref, batch["labels"], pol_kl, ref_kl, beta=cfg.beta)
    ours.loss.backward()
    assert ours.kl.item() > 0.05  # the reference-point term is really exercised

    trainer.model.train()
    trl_batch = trainer.data_collator([trainer.train_dataset[i] for i in range(4)])
    trl_loss = trainer.compute_loss(trainer.model, trl_batch)
    trl_loss.backward()
    logged = {k: v[-1] for k, v in trainer._metrics["train"].items()}

    assert trl_loss.item() == pytest.approx(ours.loss.item(), abs=1e-6)
    assert logged["kl"] == pytest.approx(ours.kl.item(), abs=1e-5)
    assert logged["rewards/chosen"] == pytest.approx(ours.chosen_rewards.mean().item(), abs=1e-5)
    assert logged["rewards/rejected"] == pytest.approx(
        ours.rejected_rewards.mean().item(), abs=1e-5
    )
    ours_grad = torch.cat([p.grad.flatten() for p in our_policy.parameters()])
    trl_grad = torch.cat([p.grad.flatten() for p in trainer.model.parameters()])
    cosine = torch.nn.functional.cosine_similarity(ours_grad, trl_grad, dim=0).item()
    relative = ((ours_grad - trl_grad).norm() / ours_grad.norm()).item()
    assert cosine > 1 - 1e-6
    assert relative < 1e-4


@pytest.mark.parametrize("grad_accum", [1, 2], ids=["accum1", "accum2"])
def test_training_runs_agree_step_by_step(tmp_path: Path, grad_accum: int) -> None:
    # 6 pairs -> 12 answers -> 3 micro-batches of 4 per epoch, 2 epochs, fixed order.
    cfg = make_cfg(
        max_length=32, batch_size=4, grad_accum=grad_accum, epochs=2, lr=1e-3, warmup_ratio=0.25
    )
    trl_policy = parity_model()
    trainer, pairs, _ = parity_setup(cfg, 6, tmp_path, trl_policy, parity_model())
    trainer.train()
    trl_steps = [h for h in trainer.state.log_history if "loss" in h and "train_runtime" not in h]

    ours_policy = parity_model()
    train(cfg, ours_policy, parity_model(), pairs, pairs, PAD, tmp_path / "ours")
    _, rows = read_csv(tmp_path / "ours" / "train.csv")

    assert len(rows) == len(trl_steps) == plan_steps(12, cfg)[1]
    for ours, theirs in zip(rows, trl_steps, strict=True):
        assert float(ours["loss"]) == pytest.approx(theirs["loss"], abs=1e-5)
        assert float(ours["kl"]) == pytest.approx(theirs["kl"], abs=1e-5)
        assert float(ours["grad_norm"]) == pytest.approx(theirs["grad_norm"], rel=1e-4)
    start = parity_model()
    worst = max(
        (a - b).abs().max().item()
        for a, b in zip(ours_policy.parameters(), trl_policy.parameters(), strict=True)
    )
    moved = max(
        (a - b).abs().max().item()
        for a, b in zip(ours_policy.parameters(), start.parameters(), strict=True)
    )
    assert moved > 1e-3  # training really changed the weights
    assert worst < 1e-6
    assert any(float(r["kl"]) > 0 for r in rows)  # the reference point left 0 during training
