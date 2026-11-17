"""Tests for the hand-written DPO training loop. Offline: a tiny random model, handmade pairs."""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from tiny_rlhf.data import PreferenceCollator, TokenizedPair
from tiny_rlhf.train_dpo import (
    EVAL_COLUMNS,
    PAIR_COLUMNS,
    TRAIN_COLUMNS,
    DPOConfig,
    evaluate,
    load_config,
    lr_factor,
    pair_logps,
    plan_steps,
    reference_logps,
    train,
    use_amp,
)

LOG2 = math.log(2.0)
PAD = 0
VOCAB = 50


def tiny_model(seed: int = 0) -> LlamaForCausalLM:
    """A 2-layer random causal LM, about 24k parameters."""
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


def make_pairs(n: int, seed: int = 0) -> list[TokenizedPair]:
    """Random pairs with different prompt and response lengths, so padding is exercised."""
    g = torch.Generator().manual_seed(seed)

    def ids(length: int) -> list[int]:
        return torch.randint(3, VOCAB, (length,), generator=g).tolist()

    return [
        TokenizedPair(
            prompt_ids=ids(3 + i % 3), chosen_ids=ids(2 + i % 4), rejected_ids=ids(2 + (i + 2) % 4)
        )
        for i in range(n)
    ]


def make_cfg(**overrides: object) -> DPOConfig:
    values = {
        "run_name": "test",
        "n_train": 8,
        "n_eval": 8,
        "max_length": 32,
        "beta": 0.5,
        "lr": 1e-2,
        "warmup_ratio": 0.0,
        "epochs": 30,
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
    return DPOConfig(**values)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def run(tmp_path: Path, name: str = "run", **overrides: object) -> tuple[dict, Path, object]:
    policy = tiny_model()
    reference = copy.deepcopy(policy)
    pairs = make_pairs(8)
    out_dir = tmp_path / name
    info = train(make_cfg(**overrides), policy, reference, pairs, pairs, PAD, out_dir)
    return info, out_dir, reference


def test_step_zero_eval_is_log2_with_zero_accuracy(tmp_path: Path) -> None:
    info, out_dir, _ = run(tmp_path, epochs=1)
    _, rows = read_csv(out_dir / "eval.csv")
    assert rows[0]["step"] == "0"
    assert float(rows[0]["loss"]) == pytest.approx(LOG2, abs=1e-7)  # float32 log 2
    assert float(rows[0]["reward_margin"]) == 0.0
    assert float(rows[0]["accuracy"]) == 0.0  # every pair is a tie before training
    assert info["first_eval_loss"] == pytest.approx(LOG2, abs=1e-7)  # float32 log 2


def test_training_lowers_the_loss_and_opens_a_positive_margin(tmp_path: Path) -> None:
    _, out_dir, _ = run(tmp_path)
    _, rows = read_csv(out_dir / "eval.csv")
    first, last = rows[0], rows[-1]
    assert float(last["loss"]) < float(first["loss"]) - 0.1
    assert float(last["reward_margin"]) > 0
    assert float(last["accuracy"]) == 1.0
    assert int(last["n_pairs"]) == 8


def test_reference_is_untouched_by_training(tmp_path: Path) -> None:
    policy = tiny_model()
    reference = copy.deepcopy(policy)
    before = copy.deepcopy(reference.state_dict())
    pairs = make_pairs(8)
    train(make_cfg(epochs=3), policy, reference, pairs, pairs, PAD, tmp_path / "run")
    after = reference.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert not any(p.requires_grad for p in reference.parameters())
    assert not reference.training
    # ... while the policy did move.
    assert any(not torch.equal(before[k], v) for k, v in policy.state_dict().items())


def test_output_files_and_csv_headers(tmp_path: Path) -> None:
    info, out_dir, _ = run(tmp_path, epochs=2, eval_every=2)
    train_header, train_rows = read_csv(out_dir / "train.csv")
    eval_header, eval_rows = read_csv(out_dir / "eval.csv")
    pair_header, pair_rows = read_csv(out_dir / "eval_pairs_final.csv")
    assert tuple(train_header) == TRAIN_COLUMNS
    assert ",".join(train_header) == (
        "step,epoch,loss,reward_margin,accuracy,chosen_logp,rejected_logp,"
        "chosen_reward,rejected_reward,lr,grad_norm,elapsed_s"
    )
    assert tuple(eval_header) == EVAL_COLUMNS
    assert ",".join(eval_header) == (
        "step,loss,reward_margin,accuracy,chosen_logp,rejected_logp,"
        "chosen_reward,rejected_reward,n_pairs"
    )
    assert tuple(pair_header) == PAIR_COLUMNS
    assert [r["step"] for r in train_rows] == ["1", "2", "3", "4"]
    # Step 0, then every 2 steps; the final step is not evaluated twice.
    assert [r["step"] for r in eval_rows] == ["0", "2", "4"]
    assert len(pair_rows) == 8
    assert (out_dir / "config.yaml").exists()
    assert (out_dir / "run_info.json").exists()
    assert info["optimizer_steps"] == 4
    assert info["trainable_params"] == info["total_params"] > 0


def test_final_eval_runs_when_the_last_step_is_off_the_eval_grid(tmp_path: Path) -> None:
    _, out_dir, _ = run(tmp_path, epochs=3, eval_every=4)  # 6 steps
    _, eval_rows = read_csv(out_dir / "eval.csv")
    assert [r["step"] for r in eval_rows] == ["0", "4", "6"]


def test_final_pair_rows_agree_with_the_final_eval_row(tmp_path: Path) -> None:
    _, out_dir, _ = run(tmp_path, epochs=3)
    _, eval_rows = read_csv(out_dir / "eval.csv")
    _, pair_rows = read_csv(out_dir / "eval_pairs_final.csv")
    mean_margin = sum(float(r["reward_margin"]) for r in pair_rows) / len(pair_rows)
    assert mean_margin == pytest.approx(float(eval_rows[-1]["reward_margin"]), abs=1e-6)
    for r in pair_rows:
        chosen = 0.5 * (float(r["chosen_logp"]) - float(r["ref_chosen_logp"]))
        assert chosen == pytest.approx(float(r["chosen_reward"]), abs=1e-5)


def test_same_seed_gives_identical_metrics(tmp_path: Path) -> None:
    _, dir_a, _ = run(tmp_path, name="a", epochs=3)
    _, dir_b, _ = run(tmp_path, name="b", epochs=3)

    def metrics(path: Path) -> list[dict[str, str]]:
        return [{k: v for k, v in row.items() if k != "elapsed_s"} for row in read_csv(path)[1]]

    assert metrics(dir_a / "train.csv") == metrics(dir_b / "train.csv")
    assert read_csv(dir_a / "eval.csv")[1] == read_csv(dir_b / "eval.csv")[1]


def test_a_different_seed_changes_the_batch_order(tmp_path: Path) -> None:
    _, dir_a, _ = run(tmp_path, name="a", epochs=1, seed=0)
    _, dir_b, _ = run(tmp_path, name="b", epochs=1, seed=1)
    assert (
        read_csv(dir_a / "train.csv")[1][0]["chosen_logp"]
        != (read_csv(dir_b / "train.csv")[1][0]["chosen_logp"])
    )


def test_guard_raises_when_reference_differs_from_policy(tmp_path: Path) -> None:
    policy = tiny_model(seed=0)
    reference = tiny_model(seed=1)  # different weights: not a valid starting point
    pairs = make_pairs(8)
    with pytest.raises(RuntimeError, match="log 2"):
        train(make_cfg(epochs=1), policy, reference, pairs, pairs, PAD, tmp_path / "run")


def test_lr_schedule_endpoints() -> None:
    total, warmup = 100, 10
    assert lr_factor(0, total, warmup) == 0.0
    assert lr_factor(5, total, warmup) == pytest.approx(0.5)
    assert lr_factor(10, total, warmup) == 1.0  # the peak, right after warmup
    assert lr_factor(55, total, warmup) == pytest.approx(0.5)
    assert lr_factor(100, total, warmup) == 0.0
    assert lr_factor(0, total, 0) == 1.0  # no warmup: start at the full lr
    factors = [lr_factor(s, total, warmup) for s in range(total + 1)]
    assert max(factors) == 1.0 and min(factors) == 0.0


def test_logged_lr_follows_the_schedule(tmp_path: Path) -> None:
    _, out_dir, _ = run(tmp_path, epochs=5, warmup_ratio=0.2, lr=1e-3)  # 10 steps, 2 warmup
    lrs = [float(r["lr"]) for r in read_csv(out_dir / "train.csv")[1]]
    assert lrs[0] == 0.0
    assert lrs[2] == pytest.approx(1e-3)
    assert lrs == pytest.approx([1e-3 * lr_factor(s, 10, 2) for s in range(10)])


def test_grad_accumulation_step_counts(tmp_path: Path) -> None:
    # 8 pairs, 2 per micro-batch = 4 micro-batches per epoch.
    assert plan_steps(8, make_cfg(batch_size=2, grad_accum=1, epochs=3)) == (4, 12)
    assert plan_steps(8, make_cfg(batch_size=2, grad_accum=4, epochs=3)) == (1, 3)
    assert plan_steps(8, make_cfg(batch_size=2, grad_accum=3, epochs=3)) == (2, 6)
    assert plan_steps(8, make_cfg(batch_size=2, grad_accum=1, epochs=3, max_steps=5)) == (4, 5)
    info, out_dir, _ = run(tmp_path, batch_size=2, grad_accum=3, epochs=3)
    assert info["optimizer_steps"] == 6
    assert len(read_csv(out_dir / "train.csv")[1]) == 6


def test_accumulated_step_equals_one_big_batch(tmp_path: Path) -> None:
    # 4 micro-batches of 2 pairs accumulated must give the same first update as one
    # batch of 8 pairs: the same mean loss, so the same gradient norm.
    _, dir_a, _ = run(tmp_path, name="a", batch_size=2, grad_accum=4, epochs=1)
    _, dir_b, _ = run(tmp_path, name="b", batch_size=8, grad_accum=1, epochs=1)
    row_a, row_b = read_csv(dir_a / "train.csv")[1][0], read_csv(dir_b / "train.csv")[1][0]
    assert float(row_a["loss"]) == pytest.approx(float(row_b["loss"]), abs=1e-6)
    assert float(row_a["grad_norm"]) == pytest.approx(float(row_b["grad_norm"]), rel=1e-4)


def test_max_steps_caps_training(tmp_path: Path) -> None:
    info, _, _ = run(tmp_path, epochs=30, max_steps=3)
    assert info["optimizer_steps"] == 3


def test_pair_logps_matches_scoring_each_sequence_alone() -> None:
    model = tiny_model().eval()
    pairs = make_pairs(4)
    batch = PreferenceCollator(pad_token_id=PAD)(pairs)
    with torch.no_grad():
        chosen, rejected = pair_logps(model, batch)
        assert chosen.shape == rejected.shape == (4,)
        for i, pair in enumerate(pairs):
            single = PreferenceCollator(pad_token_id=PAD)([pair])
            c, r = pair_logps(model, single)
            assert chosen[i].item() == pytest.approx(c.item(), abs=1e-4)
            assert rejected[i].item() == pytest.approx(r.item(), abs=1e-4)


def test_evaluate_restores_training_mode_and_weights_batches_by_pairs() -> None:
    policy = tiny_model(seed=0).train()
    reference = tiny_model(seed=1).eval()
    pairs = make_pairs(7)
    collate = PreferenceCollator(pad_token_id=PAD)
    uneven = [collate(pairs[:5]), collate(pairs[5:])]
    one = [collate(pairs)]
    device = torch.device("cpu")
    m_uneven, rows = evaluate(policy, reference, uneven, 0.5, device)
    m_one, _ = evaluate(policy, reference, one, 0.5, device)
    assert policy.training
    assert m_uneven["n_pairs"] == 7 and len(rows) == 7
    assert [r["pair_index"] for r in rows] == list(range(7))
    assert m_uneven["loss"] == pytest.approx(m_one["loss"], abs=1e-5)
    assert m_uneven["reward_margin"] == pytest.approx(m_one["reward_margin"], abs=1e-5)


def test_lora_policy_starts_at_log2_and_trains_only_the_adapter(tmp_path: Path) -> None:
    from peft import LoraConfig, get_peft_model

    base = tiny_model()
    reference = copy.deepcopy(base)
    base_before = copy.deepcopy(base.state_dict())
    policy = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    pairs = make_pairs(8)
    ckpt = tmp_path / "ckpt"
    info = train(
        make_cfg(use_lora=True, save_adapter=True, epochs=20),
        policy,
        reference,
        pairs,
        pairs,
        PAD,
        tmp_path / "run",
        checkpoint_dir=ckpt,
    )
    _, rows = read_csv(tmp_path / "run" / "eval.csv")
    # B starts at zero, so the adapter adds exactly nothing before the first update.
    assert float(rows[0]["loss"]) == pytest.approx(LOG2, abs=1e-7)  # float32 log 2
    assert float(rows[-1]["loss"]) < LOG2
    assert 0 < info["trainable_params"] < info["total_params"]
    # The original weights are frozen: only lora_A / lora_B moved.
    after = policy.state_dict()
    for name, value in base_before.items():
        key = f"base_model.model.{name}"
        if key not in after:  # a wrapped layer keeps its original weight under base_layer
            key = key.replace(".weight", ".base_layer.weight")
        assert torch.equal(after[key], value)
    trained = [n for n, p in policy.named_parameters() if p.requires_grad]
    assert trained and all("lora_" in n for n in trained)
    assert (ckpt / "adapter_config.json").exists()


def test_load_config_reads_yaml_applies_overrides_and_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("run_name: x\nbeta: 0.1\nlr: 1.0e-4\n")
    cfg = load_config(path, {"beta": 0.5, "max_steps": 3})
    assert (cfg.run_name, cfg.beta, cfg.lr, cfg.max_steps) == ("x", 0.5, 1e-4, 3)
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(path, {"betta": 0.5})


@pytest.mark.parametrize("name", ["dpo_tiny.yaml", "dpo_full.yaml"])
def test_shipped_configs_load(name: str) -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / name)
    assert cfg.beta == 0.1 and cfg.use_lora and isinstance(cfg.lr, float)


def test_guard_raises_on_a_nan_loss(tmp_path: Path) -> None:
    policy = tiny_model()
    reference = copy.deepcopy(policy)
    with torch.no_grad():
        next(policy.parameters()).fill_(float("nan"))
    pairs = make_pairs(8)
    with pytest.raises(RuntimeError, match="log 2"):
        train(make_cfg(epochs=1), policy, reference, pairs, pairs, PAD, tmp_path / "run")


def test_non_finite_gradient_stops_the_run(tmp_path: Path) -> None:
    # An absurd learning rate blows the weights up within a few steps. The run must
    # stop with an error instead of logging optimizer steps that were skipped.
    policy = tiny_model()
    reference = copy.deepcopy(policy)
    pairs = make_pairs(8)
    with pytest.raises(RuntimeError, match="non-finite gradient"):
        train(make_cfg(lr=1e30, epochs=50), policy, reference, pairs, pairs, PAD, tmp_path / "r")


def test_uneven_accumulation_equals_one_big_batch(tmp_path: Path) -> None:
    # 7 pairs in micro-batches of 2 gives sizes 2, 2, 2, 1. Weighting each micro-batch
    # by its share of the pairs must reproduce the single batch of 7.
    rows = {}
    for name, batch_size, grad_accum in (("a", 2, 4), ("b", 7, 1)):
        policy = tiny_model()
        cfg = make_cfg(batch_size=batch_size, grad_accum=grad_accum, epochs=1)
        train(
            cfg, policy, copy.deepcopy(policy), make_pairs(7), make_pairs(7), PAD, tmp_path / name
        )
        rows[name] = read_csv(tmp_path / name / "train.csv")[1]
    assert len(rows["a"]) == len(rows["b"]) == 1
    assert float(rows["a"][0]["grad_norm"]) == pytest.approx(
        float(rows["b"][0]["grad_norm"]), rel=1e-4
    )


def test_cached_reference_logps_give_the_same_evaluation() -> None:
    policy = tiny_model(seed=0)
    reference = tiny_model(seed=1)
    collate = PreferenceCollator(pad_token_id=PAD)
    pairs = make_pairs(7)
    batches = [collate(pairs[i : i + 3]) for i in range(0, 7, 3)]
    device = torch.device("cpu")
    cached = reference_logps(reference, batches, device)
    direct, direct_rows = evaluate(policy, reference, batches, 0.5, device)
    with_cache, cache_rows = evaluate(policy, reference, batches, 0.5, device, ref_logps=cached)
    assert direct == with_cache
    assert direct_rows == cache_rows


def test_amp_setting_parses_and_is_respected(tmp_path: Path) -> None:
    cuda, cpu = torch.device("cuda"), torch.device("cpu")
    for text, on_cuda in (
        ("auto", True),
        ("true", True),
        ("on", True),
        ("false", False),
        ("off", False),
    ):
        path = tmp_path / f"{text}.yaml"
        path.write_text(f"amp: {text}\n")
        cfg = load_config(path)
        assert use_amp(cfg, cuda) is on_cuda
        assert use_amp(cfg, cpu) is False  # fp16 autocast is only ever used on cuda
    bad = tmp_path / "bad.yaml"
    bad.write_text("amp: sometimes\n")
    with pytest.raises(ValueError, match="amp"):
        load_config(bad)


def test_config_rejects_zero_eval_every(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("eval_every: 0\n")
    with pytest.raises(ValueError, match="at least 1"):
        load_config(path)


def test_empty_eval_set_is_rejected(tmp_path: Path) -> None:
    policy = tiny_model()
    with pytest.raises(ValueError, match="non-empty"):
        train(make_cfg(), policy, copy.deepcopy(policy), make_pairs(4), [], PAD, tmp_path / "r")


def test_shuffle_false_visits_pairs_in_dataset_order() -> None:
    from tiny_rlhf.train_dpo import epoch_steps

    generator = torch.Generator().manual_seed(0)
    cfg = make_cfg(batch_size=2, grad_accum=2, shuffle=False)
    assert epoch_steps(7, cfg, generator) == [[[0, 1], [2, 3]], [[4, 5], [6]]]
    assert epoch_steps(7, cfg, generator) == [[[0, 1], [2, 3]], [[4, 5], [6]]]  # every epoch
    shuffled = epoch_steps(7, make_cfg(batch_size=2, grad_accum=2), generator)
    assert sorted(i for step in shuffled for micro in step for i in micro) == list(range(7))


@pytest.mark.parametrize("name", ["parity_tiny.yaml", "parity_full.yaml"])
def test_parity_configs_load_and_fix_the_order(name: str) -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / name)
    assert cfg.shuffle is False
    assert cfg.lora_dropout == 0.0
