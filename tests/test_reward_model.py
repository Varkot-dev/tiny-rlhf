"""Tests for the explicit reward model. Offline: a tiny random backbone, handmade pairs."""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaModel

from tiny_rlhf.data import PreferenceCollator, TokenizedPair
from tiny_rlhf.losses import bradley_terry_loss
from tiny_rlhf.reward_model import (
    EVAL_COLUMNS,
    PAIR_COLUMNS,
    TRAIN_COLUMNS,
    RewardConfig,
    RewardModel,
    bootstrap_ci,
    evaluate,
    load_reward_config,
    pair_rewards,
    pairs_fingerprint,
    train,
)

LOG2 = math.log(2.0)
PAD = 0
VOCAB = 50
HIDDEN = 32


def tiny_backbone(seed: int = 0) -> LlamaModel:
    """A 2-layer random causal transformer without a vocabulary head."""
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        pad_token_id=PAD,
        bos_token_id=1,
        eos_token_id=2,
    )
    return LlamaModel(config)


def tiny_reward_model(seed: int = 0, random_head: bool = False) -> RewardModel:
    model = RewardModel(tiny_backbone(seed), HIDDEN)
    if random_head:  # a non-zero head, for tests that need rewards to differ between rows
        torch.manual_seed(seed + 100)
        torch.nn.init.normal_(model.head.weight, std=0.5)
        torch.nn.init.normal_(model.head.bias, std=0.5)
    return model.eval()


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


def make_cfg(**overrides: object) -> RewardConfig:
    values = {
        "run_name": "test",
        "n_train": 8,
        "n_eval": 8,
        "max_length": 32,
        "lr": 1e-2,
        "warmup_ratio": 0.0,
        "epochs": 30,
        "batch_size": 4,
        "grad_accum": 1,
        "eval_every": 1000,
        "eval_batch_size": 3,
        "seed": 0,
        "use_lora": False,
        "train_backbone": True,
        "device": "cpu",
    }
    values.update(overrides)
    return RewardConfig(**values)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def run(tmp_path: Path, name: str = "run", **overrides: object) -> tuple[dict, Path, RewardModel]:
    model = tiny_reward_model()
    pairs = make_pairs(8)
    out_dir = tmp_path / name
    info = train(make_cfg(**overrides), model, pairs, pairs, PAD, out_dir)
    return info, out_dir, model


# --------------------------------------------------------------------------------------
# Pooling: the reward is read from the last real token of each row
# --------------------------------------------------------------------------------------


def test_reward_is_read_from_the_last_real_token() -> None:
    model = tiny_reward_model(random_head=True)
    ids = torch.tensor([[5, 6, 7, 8, 9, PAD, PAD], [10, 11, 12, PAD, PAD, PAD, PAD]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0]])
    with torch.no_grad():
        hidden = model.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        expected = torch.stack(
            [model.head(hidden[0, 4]).squeeze(), model.head(hidden[1, 2]).squeeze()]
        )
        got = model(ids, mask)
    assert got.shape == (2,)
    assert torch.allclose(got, expected, atol=1e-6)
    # And it is not simply the last column, which is padding here.
    last_column = model.head(hidden[:, -1]).squeeze(-1)
    assert not torch.allclose(got, last_column, atol=1e-4)


def test_extra_padding_and_pad_token_id_do_not_change_the_reward() -> None:
    model = tiny_reward_model(random_head=True)
    row = [5, 6, 7, 8]
    with torch.no_grad():
        alone = model(torch.tensor([row]), torch.ones(1, 4, dtype=torch.long))
        for pad_id in (PAD, 7, 49):
            padded = torch.tensor([row + [pad_id] * 5])
            mask = torch.tensor([[1] * 4 + [0] * 5])
            assert torch.allclose(model(padded, mask), alone, atol=1e-5)


def test_reward_is_per_row_batching_does_not_change_it() -> None:
    model = tiny_reward_model(random_head=True)
    pairs = make_pairs(5)
    batch = PreferenceCollator(pad_token_id=PAD)(pairs)
    with torch.no_grad():
        together_c, together_r = pair_rewards(model, batch)
        for i, pair in enumerate(pairs):
            one = PreferenceCollator(pad_token_id=PAD)([pair])
            alone_c, alone_r = pair_rewards(model, one)
            assert together_c[i].item() == pytest.approx(alone_c.item(), abs=1e-5)
            assert together_r[i].item() == pytest.approx(alone_r.item(), abs=1e-5)


def test_every_real_token_influences_the_reward() -> None:
    # The last state has seen every token, so changing any real token moves the reward.
    model = tiny_reward_model(random_head=True)
    mask = torch.ones(1, 5, dtype=torch.long)
    base = torch.tensor([[5, 6, 7, 8, 9]])
    with torch.no_grad():
        r = model(base, mask)
        for position in range(5):
            changed = base.clone()
            changed[0, position] = 40
            assert not torch.allclose(model(changed, mask), r, atol=1e-6)


# --------------------------------------------------------------------------------------
# The zero head and the first loss
# --------------------------------------------------------------------------------------


def test_zero_head_gives_zero_rewards_and_a_first_loss_of_exactly_log2(tmp_path: Path) -> None:
    info, out_dir, _ = run(tmp_path, epochs=1)
    _, rows = read_csv(out_dir / "eval.csv")
    assert rows[0]["step"] == "0"
    assert float(rows[0]["loss"]) == pytest.approx(LOG2, abs=1e-7)  # float32 log 2
    assert float(rows[0]["chosen_reward"]) == 0.0
    assert float(rows[0]["rejected_reward"]) == 0.0
    assert float(rows[0]["accuracy"]) == 0.0  # every pair is a tie, and ties count as wrong
    assert info["first_eval_loss"] == pytest.approx(LOG2, abs=1e-7)


def test_zero_head_still_gets_gradient_but_the_backbone_does_not_at_step_zero() -> None:
    model = tiny_reward_model().train()
    batch = PreferenceCollator(pad_token_id=PAD)(make_pairs(4))
    chosen, rejected = pair_rewards(model, batch)
    bradley_terry_loss(chosen, rejected).backward()
    assert model.head.weight.grad.abs().sum() > 0
    # The gradient into the backbone passes through the head weight, which is zero.
    for p in model.backbone.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0


def test_head_gradient_at_step_zero_matches_the_formula() -> None:
    # d loss / d w = -(1/2) * mean(h_chosen - h_rejected) when every reward is zero.
    model = tiny_reward_model().train()
    batch = PreferenceCollator(pad_token_id=PAD)(make_pairs(4))
    chosen, rejected = pair_rewards(model, batch)
    bradley_terry_loss(chosen, rejected).backward()
    with torch.no_grad():
        hidden = model.backbone(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state
        last = batch["attention_mask"].sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0)), last]
        h_chosen, h_rejected = pooled.chunk(2, dim=0)
        expected = -0.5 * (h_chosen - h_rejected).mean(dim=0)
    assert torch.allclose(model.head.weight.grad.squeeze(0), expected, atol=1e-6)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def test_training_lowers_the_loss_and_reaches_positive_margins(tmp_path: Path) -> None:
    info, out_dir, _ = run(tmp_path)
    _, rows = read_csv(out_dir / "eval.csv")
    assert float(rows[-1]["loss"]) < 0.5 * LOG2
    assert float(rows[-1]["reward_margin"]) > 0
    assert float(rows[-1]["accuracy"]) == 1.0
    assert info["heldout_accuracy"] == 1.0
    assert info["final_eval_loss"] == pytest.approx(float(rows[-1]["loss"]))


def test_adding_a_constant_to_all_rewards_leaves_the_loss_unchanged() -> None:
    model = tiny_reward_model(random_head=True)
    batch = PreferenceCollator(pad_token_id=PAD)(make_pairs(6))
    with torch.no_grad():
        chosen, rejected = pair_rewards(model, batch)
        base = bradley_terry_loss(chosen, rejected)
        model.head.bias += 123.0  # shifts every reward by the same amount
        shifted_c, shifted_r = pair_rewards(model, batch)
        shifted = bradley_terry_loss(shifted_c, shifted_r)
    assert torch.allclose(shifted_c - chosen, torch.full_like(chosen, 123.0), atol=1e-4)
    assert shifted.item() == pytest.approx(base.item(), abs=1e-5)


def test_frozen_backbone_trains_only_the_head(tmp_path: Path) -> None:
    model = tiny_reward_model()
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    before = copy.deepcopy(model.backbone.state_dict())
    pairs = make_pairs(8)
    info = train(
        make_cfg(use_lora=False, train_backbone=False, lr=5e-2), model, pairs, pairs, PAD, tmp_path
    )
    after = model.backbone.state_dict()
    assert all(torch.equal(after[k], v) for k, v in before.items())
    assert model.head.weight.abs().sum() > 0
    assert info["trainable_params"] == info["head_params"] == HIDDEN + 1
    _, rows = read_csv(tmp_path / "eval.csv")
    assert float(rows[-1]["loss"]) < LOG2  # a linear probe on frozen features still learns


def test_lora_trains_only_the_adapter_and_the_head(tmp_path: Path) -> None:
    from peft import LoraConfig, get_peft_model

    backbone = tiny_backbone()
    base_before = copy.deepcopy(backbone.state_dict())
    peft_backbone = get_peft_model(
        backbone,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="FEATURE_EXTRACTION",
        ),
    )
    model = RewardModel(peft_backbone, HIDDEN)
    pairs = make_pairs(8)
    info = train(make_cfg(use_lora=True, epochs=20), model, pairs, pairs, PAD, tmp_path)

    trained = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trained and all("lora_" in n or n.startswith("head.") for n in trained)
    assert any("lora_" in n for n in trained)
    after = peft_backbone.state_dict()
    for name, value in base_before.items():
        key = f"base_model.model.{name}"
        if key not in after:  # a wrapped layer keeps its original weight under base_layer
            key = key.replace(".weight", ".base_layer.weight")
        assert torch.equal(after[key], value)
    # The adapter did move: lora_B starts at zero and must not be zero any more.
    moved = [p for n, p in model.named_parameters() if "lora_B" in n]
    assert any(p.abs().sum() > 0 for p in moved)
    assert info["head_params"] < info["trainable_params"] < info["total_params"]
    _, rows = read_csv(tmp_path / "eval.csv")
    assert float(rows[0]["loss"]) == pytest.approx(LOG2, abs=1e-7)
    assert float(rows[-1]["loss"]) < LOG2


def test_head_lr_gives_the_head_its_own_learning_rate(tmp_path: Path) -> None:
    # With lr 0 for the backbone and a head_lr, only the head can move.
    model = tiny_reward_model()
    before = copy.deepcopy(model.backbone.state_dict())
    pairs = make_pairs(8)
    train(make_cfg(lr=0.0, head_lr=5e-2, epochs=5), model, pairs, pairs, PAD, tmp_path)
    after = model.backbone.state_dict()
    assert all(torch.equal(after[k], v) for k, v in before.items())
    assert model.head.weight.abs().sum() > 0


def test_same_seed_gives_identical_metrics(tmp_path: Path) -> None:
    _, dir_a, _ = run(tmp_path, name="a", epochs=3)
    _, dir_b, _ = run(tmp_path, name="b", epochs=3)

    def metrics(path: Path) -> list[dict[str, str]]:
        return [{k: v for k, v in r.items() if k != "elapsed_s"} for r in read_csv(path)[1]]

    assert metrics(dir_a / "train.csv") == metrics(dir_b / "train.csv")
    assert read_csv(dir_a / "eval_pairs_final.csv") == read_csv(dir_b / "eval_pairs_final.csv")


def test_shuffle_false_keeps_dataset_order_and_differs_from_shuffled(tmp_path: Path) -> None:
    _, dir_a, _ = run(tmp_path, name="a", epochs=2, shuffle=False)
    _, dir_b, _ = run(tmp_path, name="b", epochs=2, shuffle=False, seed=5)
    _, dir_c, _ = run(tmp_path, name="c", epochs=2, shuffle=True)
    losses = [[r["loss"] for r in read_csv(d / "train.csv")[1]] for d in (dir_a, dir_b, dir_c)]
    assert losses[0] == losses[1]  # no shuffling and no dropout: the seed does not matter
    assert losses[0] != losses[2]


def test_output_files_and_csv_headers(tmp_path: Path) -> None:
    info, out_dir, _ = run(tmp_path, epochs=2, eval_every=1)
    assert read_csv(out_dir / "train.csv")[0] == list(TRAIN_COLUMNS)
    assert read_csv(out_dir / "eval.csv")[0] == list(EVAL_COLUMNS)
    header, pair_rows = read_csv(out_dir / "eval_pairs_final.csv")
    assert header == list(PAIR_COLUMNS)
    assert list(TRAIN_COLUMNS) == [
        "step",
        "epoch",
        "loss",
        "reward_margin",
        "accuracy",
        "chosen_reward",
        "rejected_reward",
        "lr",
        "grad_norm",
        "elapsed_s",
    ]
    assert list(EVAL_COLUMNS) == [
        "step",
        "loss",
        "reward_margin",
        "accuracy",
        "chosen_reward",
        "rejected_reward",
        "n_pairs",
    ]
    assert [r["pair_index"] for r in pair_rows] == [str(i) for i in range(8)]
    for name in ("config.yaml", "run_info.json"):
        assert (out_dir / name).exists()
    # 2 epochs of 2 steps, eval at 0 and after each step, no duplicate final row.
    assert [r["step"] for r in read_csv(out_dir / "eval.csv")[1]] == ["0", "1", "2", "3", "4"]
    assert info["optimizer_steps"] == 4


def test_final_pair_rows_agree_with_the_final_eval_row_and_run_info(tmp_path: Path) -> None:
    info, out_dir, _ = run(tmp_path, epochs=3)
    _, eval_rows = read_csv(out_dir / "eval.csv")
    _, pair_rows = read_csv(out_dir / "eval_pairs_final.csv")
    margins = [float(r["reward_margin"]) for r in pair_rows]
    assert sum(margins) / len(margins) == pytest.approx(
        float(eval_rows[-1]["reward_margin"]), abs=1e-5
    )
    accuracy = sum(m > 0 for m in margins) / len(margins)
    assert info["heldout_accuracy"] == pytest.approx(accuracy)
    low, high = info["heldout_accuracy_ci95"]
    assert low <= accuracy <= high
    assert info["bootstrap_resamples"] == 2000


def test_evaluate_restores_training_mode_and_weights_batches_by_pairs() -> None:
    model = tiny_reward_model(random_head=True).train()
    collate = PreferenceCollator(pad_token_id=PAD)
    pairs = make_pairs(7)
    batches = [collate(pairs[i : i + 3]) for i in range(0, 7, 3)]  # sizes 3, 3, 1
    metrics, rows = evaluate(model, batches, torch.device("cpu"))
    assert model.training
    assert metrics["n_pairs"] == 7 and len(rows) == 7
    margins = [r["reward_margin"] for r in rows]
    assert metrics["reward_margin"] == pytest.approx(sum(margins) / 7, abs=1e-5)


def test_non_finite_gradient_stops_the_run(tmp_path: Path) -> None:
    model = tiny_reward_model()
    pairs = make_pairs(8)
    with pytest.raises(RuntimeError, match="non-finite gradient"):
        train(make_cfg(lr=1e30, epochs=50), model, pairs, pairs, PAD, tmp_path)


def test_empty_eval_set_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        train(make_cfg(), tiny_reward_model(), make_pairs(4), [], PAD, tmp_path)


# --------------------------------------------------------------------------------------
# Bootstrap interval, alignment fingerprint, config
# --------------------------------------------------------------------------------------


def test_bootstrap_interval_contains_the_estimate_and_is_reproducible() -> None:
    values = [1.0] * 30 + [0.0] * 20
    low, high = bootstrap_ci(values, n_resamples=2000, seed=0)
    assert 0.0 <= low <= 0.6 <= high <= 1.0
    assert low < high
    assert bootstrap_ci(values, n_resamples=2000, seed=0) == (low, high)
    assert bootstrap_ci(values, n_resamples=2000, seed=1) != (low, high)


def test_bootstrap_interval_shrinks_with_more_pairs() -> None:
    small = bootstrap_ci([1.0] * 12 + [0.0] * 8, seed=0)
    large = bootstrap_ci([1.0] * 1200 + [0.0] * 800, seed=0)
    assert (large[1] - large[0]) < 0.25 * (small[1] - small[0])
    # For a proportion p on n pairs the standard error is sqrt(p(1-p)/n): about 0.011
    # here, so the 95% interval should be close to 0.6 plus or minus 0.021.
    assert large[0] == pytest.approx(0.6 - 1.96 * 0.011, abs=0.006)
    assert large[1] == pytest.approx(0.6 + 1.96 * 0.011, abs=0.006)


def test_bootstrap_interval_of_a_constant_sample_has_zero_width() -> None:
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([])


def test_fingerprint_identifies_pairs_order_and_max_length() -> None:
    pairs = make_pairs(6)
    fp = pairs_fingerprint(pairs, 32)
    assert fp == pairs_fingerprint(make_pairs(6), 32)
    assert len(fp) == 64
    assert fp != pairs_fingerprint(pairs[::-1], 32)
    assert fp != pairs_fingerprint(pairs[:5], 32)
    assert fp != pairs_fingerprint(pairs, 16)
    assert fp != pairs_fingerprint(make_pairs(6, seed=1), 32)


def test_run_info_records_the_eval_fingerprint(tmp_path: Path) -> None:
    info, _, _ = run(tmp_path, epochs=1)
    assert info["eval_pairs_sha256"] == pairs_fingerprint(make_pairs(8), 32)


def test_config_loads_applies_overrides_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("run_name: r\nlr: 1.0e-3\namp: off\n")
    cfg = load_reward_config(path, {"head_lr": 0.01})
    assert cfg.run_name == "r" and cfg.lr == 1e-3 and cfg.head_lr == 0.01
    assert cfg.amp is False  # YAML reads a bare off as a boolean
    for bad in ("nope: 1\n", "amp: sometimes\n", "eval_every: 0\n", "batch_size: 0\n"):
        path.write_text(bad)
        with pytest.raises(ValueError):
            load_reward_config(path)


@pytest.mark.parametrize("name", ["reward_tiny.yaml", "reward_full.yaml"])
def test_shipped_configs_load_and_align_with_the_dpo_configs(name: str) -> None:
    import yaml

    configs = Path(__file__).resolve().parents[1] / "configs"
    cfg = load_reward_config(configs / name)
    dpo = yaml.safe_load((configs / name.replace("reward_", "dpo_")).read_text())
    # These five fields decide which held-out pairs are scored and how they are cut.
    for key in ("model_name", "dataset_name", "n_eval", "seed", "max_length"):
        assert getattr(cfg, key) == dpo[key], key


# --------------------------------------------------------------------------------------
# fp16 retry path and stale outputs (same behaviour as the DPO loop)
# --------------------------------------------------------------------------------------


def poison_head_gradients(model: RewardModel, times: int | None) -> dict[str, int]:
    state = {"calls": 0}

    def hook(grad: torch.Tensor) -> torch.Tensor:
        state["calls"] += 1
        if times is None or state["calls"] <= times:
            return grad * float("nan")
        return grad

    model.head.weight.register_hook(hook)
    return state


def test_reward_training_redoes_an_overflowed_step(tmp_path: Path) -> None:
    def one(name: str, poison: int | None) -> tuple:
        model = tiny_reward_model()
        state = poison_head_gradients(model, poison) if poison else {"calls": 0}
        scaler = torch.amp.GradScaler("cpu", enabled=True)
        cfg = make_cfg(epochs=3, batch_size=8, warmup_ratio=0.5, lr=1e-3)
        info = train(cfg, model, make_pairs(8), make_pairs(8), PAD, tmp_path / name, scaler=scaler)
        return info, read_csv(tmp_path / name / "train.csv")[1], model, scaler, state

    clean_info, clean_rows, clean_model, clean_scaler, _ = one("clean", None)
    info, rows, model, scaler, state = one("poisoned", 1)
    assert clean_info["scaler_retries"] == 0 and info["scaler_retries"] == 1
    assert state["calls"] == 3 + 1
    assert info["optimizer_steps"] == clean_info["optimizer_steps"] == 3
    assert [float(r["lr"]) for r in rows] == [float(r["lr"]) for r in clean_rows]
    assert scaler.get_scale() == pytest.approx(clean_scaler.get_scale() / 2)
    for a, b in zip(rows, clean_rows, strict=True):
        assert float(a["loss"]) == pytest.approx(float(b["loss"]), abs=1e-6)
    for p, q in zip(model.parameters(), clean_model.parameters(), strict=True):
        assert torch.allclose(p, q, atol=1e-6)


def test_reward_training_stops_at_the_retry_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tiny_rlhf.train_dpo as train_dpo

    monkeypatch.setattr(train_dpo, "MAX_SCALER_RETRIES", 2)
    model = tiny_reward_model()
    state = poison_head_gradients(model, None)
    with pytest.raises(RuntimeError, match="non-finite gradient norm at optimizer step 1 after 3"):
        train(
            make_cfg(epochs=3, batch_size=8),
            model,
            make_pairs(8),
            make_pairs(8),
            PAD,
            tmp_path / "run",
            scaler=torch.amp.GradScaler("cpu", enabled=True),
        )
    assert state["calls"] == 3
    assert not (tmp_path / "run" / "run_info.json").exists()


def test_reward_training_removes_a_previous_runs_completion_files(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    for name in ("run_info.json", "eval_pairs_final.csv", "reward_vs_dpo.json"):
        (out / name).write_text("left over")
    model = tiny_reward_model()
    poison_head_gradients(model, None)
    with pytest.raises(RuntimeError, match="non-finite"):
        train(make_cfg(epochs=1, batch_size=8), model, make_pairs(8), make_pairs(8), PAD, out)
    assert not any((out / n).exists() for n in ("run_info.json", "eval_pairs_final.csv"))
    assert not (out / "reward_vs_dpo.json").exists()
