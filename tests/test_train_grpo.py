"""Tests for the GRPO training run (tiny_rlhf.train_grpo): config, prompt selection, TRL
settings, logging, and a two-step run end to end on a tiny random model (offline)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from tiny_rlhf.train_grpo import (
    COMPONENT_COLUMNS,
    TRAIN_COLUMNS,
    GRPORunConfig,
    load_grpo_config,
    log_row,
    prompt_rows,
    select_prompts,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "trackb" / "train.jsonl"

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
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(WORDS),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=1024,  # Track B prompts are a few hundred words
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
    )
    return LlamaForCausalLM(config)


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


def test_both_configs_load_and_name_every_field() -> None:
    fields = set(GRPORunConfig.__dataclass_fields__)
    for name in ("trackb_grpo", "trackb_grpo_tiny"):
        path = ROOT / "configs" / f"{name}.yaml"
        values = yaml.safe_load(path.read_text())
        assert set(values) == fields, f"{name}: every field is written out"
        cfg = load_grpo_config(path)
        assert cfg.run_name == name
        assert cfg.completions_per_step() % cfg.num_generations == 0
    full = load_grpo_config(ROOT / "configs" / "trackb_grpo.yaml")
    assert full.num_generations == 8
    assert (full.prompts_per_step(), full.completions_per_step()) == (4, 32)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"num_generations": 1}, "at least 2"),
        ({"per_device_train_batch_size": 3}, "multiple of num_generations"),
        ({"amp": "yes"}, "amp"),
        ({"max_minutes": 0}, "max_minutes"),
        ({"max_steps": 0}, "max_steps"),
        ({"bogus": 1}, "unknown config keys"),
    ],
)
def test_bad_configs_are_rejected(override: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_grpo_config(ROOT / "configs" / "trackb_grpo_tiny.yaml", override)


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("max_length", [None, 90])
def test_prompts_are_the_dpo_runs_prompts_in_the_same_order(max_length: int | None) -> None:
    from tiny_rlhf.data import _template_ids
    from tiny_rlhf.trackb.train import select_pairs

    tokenizer = make_tokenizer()
    examples = select_prompts(TRAIN_FILE, tokenizer, max_length, n=40, seed=0)
    pairs = select_pairs(TRAIN_FILE, tokenizer, max_length, n=40, seed=0)
    assert len(examples) == len(pairs)
    if max_length is not None:
        assert len(examples) < 40  # the filter is exercised
    for example, pair in zip(examples, pairs, strict=True):
        ids = _template_ids(tokenizer, example["chosen"][:-1], add_generation_prompt=True)
        assert ids == pair.prompt_ids
    rows = prompt_rows(examples)
    assert rows[0] == {"prompt": examples[0]["chosen"][:-1], "id": examples[0]["id"]}
    assert rows[0]["prompt"][-1]["role"] == "user"


# --------------------------------------------------------------------------------------
# TRL settings and logging
# --------------------------------------------------------------------------------------


def test_trl_settings_follow_the_config() -> None:
    pytest.importorskip("trl")
    from tiny_rlhf.train_grpo import build_grpo_args, trl_settings

    cfg = load_grpo_config(ROOT / "configs" / "trackb_grpo.yaml")
    args = build_grpo_args(cfg, "unused_output_dir", device="cpu")
    settings = trl_settings(args)
    assert settings["num_generations"] == 8
    assert settings["generation_batch_size"] == 32
    assert settings["steps_per_generation"] == 4  # None in the config: one per step
    assert settings["max_completion_length"] == 96
    assert settings["beta"] == 0.04 and settings["loss_type"] == "dapo"
    assert settings["num_iterations"] == 1 and settings["use_bias_correction_kl"] is True
    assert settings["learning_rate"] == 2e-5 and settings["warmup_steps"] == 36
    assert settings["fp16"] is False and settings["bf16"] is False  # cpu here
    assert settings["max_steps"] == 360 and settings["seed"] == 0
    assert args.save_strategy == "no" and args.logging_steps == 1


def test_log_row_maps_trls_names() -> None:
    logs = {
        "reward": 0.25,
        "reward_std": 0.8,
        "frac_reward_zero_std": 0.5,
        "kl": 0.01,
        "completions/mean_length": 12.0,
        "loss": 0.1,
        "learning_rate": 1e-5,
        "grad_norm": 0.3,
        "entropy": 1.2,
        "clip_ratio/region_mean": 0.0,
        "completions/clipped_ratio": 0.125,
        "trackb/task_correct_strict": 0.5,
        "trackb/followed_injection": 0.25,
    }
    row = log_row(logs, step=3, elapsed_s=10.0)
    assert list(row) == list(TRAIN_COLUMNS)
    assert row["reward_mean"] == 0.25 and row["frac_groups_all_equal"] == 0.5
    assert row["completion_length"] == 12.0 and row["truncated_share"] == 0.125
    assert row["rate_task_correct_strict"] == 0.5 and row["rate_refused"] == ""
    assert set(COMPONENT_COLUMNS) <= set(row)


# --------------------------------------------------------------------------------------
# End to end, offline
# --------------------------------------------------------------------------------------


def tiny_cfg(**overrides: Any) -> Any:
    values = dict(
        run_name="grpo_test",
        num_generations=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        max_completion_length=4,
        max_steps=2,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "v_proj"],
        sample_every=1,
        device="cpu",
        warmup_ratio=0.0,
    )
    values.update(overrides)
    return GRPORunConfig(**values)


def run_tiny(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    pytest.importorskip("trl")
    from tiny_rlhf.train_grpo import train

    tokenizer = make_tokenizer()
    examples = select_prompts(TRAIN_FILE, tokenizer, None, n=6, seed=0)
    return train(
        tiny_cfg(**overrides),
        make_model(),
        tokenizer,
        examples,
        out_dir=tmp_path / "results",
        checkpoint_dir=tmp_path / "checkpoints",
    )


def test_two_steps_write_every_output(tmp_path: Path) -> None:
    info = run_tiny(tmp_path)
    out = tmp_path / "results"
    assert info["optimizer_steps"] == 2 and not info["stopped_by_time_limit"]
    assert info["prompts_per_step"] == 2 and info["completions_per_step"] == 8
    assert info["trl_settings"]["beta"] == 0.04
    assert info["peak_gpu_memory_mib"] is None  # cpu
    assert json.loads((out / "run_info.json").read_text())["run_name"] == "grpo_test"
    assert yaml.safe_load((out / "config.yaml").read_text())["max_steps"] == 2
    with (out / "train.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == [1, 2]
    assert list(rows[0]) == list(TRAIN_COLUMNS)
    for r in rows:
        assert r["reward_mean"] != "" and r["kl"] != "" and r["rate_truncated"] != ""
        assert float(r["frac_groups_all_equal"]) >= 0
    samples = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    assert {s["step"] for s in samples} == {1, 2} and len(samples) == 16
    assert all(s["reward"] <= 0.75 for s in samples)  # truncated: at most 1 - 0.25
    # Random 4-token answers never end: every one hit the limit.
    assert all(s["checks"]["truncated"] for s in samples)
    assert (tmp_path / "checkpoints" / "adapter_config.json").is_file()


def test_time_limit_stops_after_the_first_step(tmp_path: Path) -> None:
    info = run_tiny(tmp_path, max_minutes=1e-6)
    assert info["optimizer_steps"] == 1 and info["planned_steps"] == 2
    assert info["stopped_by_time_limit"]
    assert (tmp_path / "checkpoints" / "adapter_config.json").is_file()
