"""GRPO on Track B (M8): TRL's GRPOTrainer with the checker reward of tiny_rlhf.grpo_reward.

What comes from where:
- prompts: the user message of each Track B training example, selected exactly as the DPO
  and KTO runs select their pairs (tiny_rlhf.trackb.train.select_pairs: a seeded shuffle of
  data/trackb/train.jsonl, the first n_train, prompts of max_length tokens or more dropped),
  given to TRL as a conversation so that it applies the chat template with the same call
  the DPO runs use (tiny_rlhf.data._template_ids);
- reward: tiny_rlhf.grpo_reward.make_reward_func (strict checks only);
- loss and advantages: TRL's, which tests/test_grpo.py checks against the hand-written
  tiny_rlhf.grpo within 1e-6;
- LoRA: the same adapter as the Track B DPO and KTO runs (rank 16, alpha 32, dropout 0.05,
  all seven projection matrices); the reference model for the KL term is the same network
  with the adapter switched off, so only one copy of the weights is in memory.

Writes results/<run_name>/{config.yaml, train.csv, samples.jsonl, run_info.json} and the
adapter to checkpoints/<run_name>/, where scripts/eval_trackb.py, eval_general.py and
compare_trackb.py read it exactly as they read the DPO and KTO adapters.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from tiny_rlhf.grpo_reward import COMPONENTS, RewardWeights, make_reward_func

# One row per optimizer step. TRL's names on the right.
TRL_LOG_KEYS = {
    "reward_mean": "reward",  # mean reward over the generation batch
    "reward_std": "reward_std",  # std over all completions of the batch (not within groups)
    "frac_groups_all_equal": "frac_reward_zero_std",  # share of groups that teach nothing
    "kl": "kl",  # token mean of the k3 estimate (only when beta != 0)
    "completion_length": "completions/mean_length",  # tokens
    "loss": "loss",
    "lr": "learning_rate",
    "grad_norm": "grad_norm",
    "entropy": "entropy",  # mean next-token entropy over completion tokens
    "clip_ratio": "clip_ratio/region_mean",  # 0 when every batch is used for one step
    "truncated_share": "completions/clipped_ratio",  # completions that hit the limit
}
COMPONENT_COLUMNS = tuple(f"rate_{key}" for key in COMPONENTS)
TRAIN_COLUMNS = ("step", *TRL_LOG_KEYS, *COMPONENT_COLUMNS, "elapsed_s")


@dataclass
class GRPORunConfig:
    """Everything that defines a GRPO run. Loaded from a YAML file under configs/."""

    run_name: str = "grpo"
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    dataset_name: str = "data/trackb"  # the directory holding train.jsonl
    n_train: int = 3000  # prompts to sample from, selected as the DPO runs select pairs
    max_length: int | None = 320  # prompts of this many tokens or more are dropped
    seed: int = 0
    # Batching, in TRL's terms. Each optimizer step uses
    #   per_device_train_batch_size * gradient_accumulation_steps completions
    # = that / num_generations prompts. Generation happens in blocks of
    #   per_device_train_batch_size * steps_per_generation completions.
    num_generations: int = 4  # G, completions per prompt
    per_device_train_batch_size: int = 4  # completions per forward/backward pass
    gradient_accumulation_steps: int = 2
    steps_per_generation: int | None = None  # None: one generation per optimizer step
    max_completion_length: int = 96  # same as the evaluation's max_new_tokens
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    beta: float = 0.04  # KL weight; TRL's default is 0
    epsilon: float = 0.2
    loss_type: str = "dapo"
    scale_rewards: str = "group"
    lr: float = 2e-5
    lr_scheduler: str = "linear"  # warmup, then linear decay to 0, as the DPO runs
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_steps: int = 10  # optimizer steps
    # Stop after the optimizer step during which this many minutes have passed, then save
    # and record as usual (run_info.json says so). None: no limit. A safety stop for a
    # free GPU session: the learning-rate schedule is still planned for max_steps.
    max_minutes: float | None = None
    gradient_checkpointing: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    # The reward's numbers (tiny_rlhf.grpo_reward.RewardWeights).
    reward_correct: float = 1.0
    reward_violation: float = -1.0
    reward_echoed: float = -0.5
    reward_truncated: float = -0.25
    # Every this many steps, every scored completion of the step goes to samples.jsonl
    # (0: never), so what the reward paid for can be read afterwards.
    sample_every: int = 25
    # "auto" or true: fp16 mixed precision on cuda (a T4 has no bfloat16). false: float32.
    amp: bool | str = "auto"
    device: str = "auto"
    save_adapter: bool = True

    def reward_weights(self) -> RewardWeights:
        return RewardWeights(
            correct=self.reward_correct,
            violation=self.reward_violation,
            echoed=self.reward_echoed,
            truncated=self.reward_truncated,
        )

    def completions_per_step(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def prompts_per_step(self) -> int:
        return self.completions_per_step() // self.num_generations


def load_grpo_config(path: str | Path, overrides: dict[str, Any] | None = None) -> GRPORunConfig:
    """Read a YAML config and apply `key: value` overrides. Unknown keys are an error."""
    values = yaml.safe_load(Path(path).read_text()) or {}
    values.update(overrides or {})
    known = {f.name for f in dataclasses.fields(GRPORunConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    cfg = GRPORunConfig(**values)
    if cfg.amp not in ("auto", True, False):
        raise ValueError(f"amp must be auto, true or false, got {cfg.amp!r}")
    if cfg.num_generations < 2:
        raise ValueError("num_generations must be at least 2: a group of one has no baseline")
    spg = cfg.steps_per_generation or cfg.gradient_accumulation_steps
    if (cfg.per_device_train_batch_size * spg) % cfg.num_generations:
        raise ValueError(
            "per_device_train_batch_size * steps_per_generation must be a multiple of "
            "num_generations (each generation block holds whole groups)"
        )
    if cfg.completions_per_step() % cfg.num_generations:
        raise ValueError("completions per optimizer step must be a multiple of num_generations")
    if cfg.max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if cfg.max_minutes is not None and cfg.max_minutes <= 0:
        raise ValueError("max_minutes must be positive (or null for no limit)")
    return cfg


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------


def select_prompts(
    path: str | Path, tokenizer: Any, max_length: int | None, n: int, seed: int
) -> list[dict[str, Any]]:
    """The examples whose prompts GRPO trains on, in the order the DPO runs use them.

    Same selection as tiny_rlhf.trackb.train.select_pairs (seeded shuffle, first n, prompts
    of max_length tokens or more dropped), so the GRPO prompts are exactly the prompts of
    the DPO and KTO runs' training pairs. Returns the full example dicts.
    """
    from tiny_rlhf.data import _template_ids
    from tiny_rlhf.trackb.load import load_trackb_examples

    examples = load_trackb_examples(path)
    random.Random(seed).shuffle(examples)
    examples = examples[:n]
    if max_length is not None:
        examples = [
            e
            for e in examples
            if len(_template_ids(tokenizer, e["chosen"][:-1], add_generation_prompt=True))
            < max_length
        ]
    return examples


def prompt_rows(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """TRL dataset rows: the conversation up to the answer, and the id the reward looks up."""
    return [{"prompt": e["chosen"][:-1], "id": e["id"]} for e in examples]


def prompts_fingerprint(tokenizer: Any, examples: list[dict[str, Any]]) -> str:
    """sha256 of the selected prompts' token ids, to check two runs trained on the same."""
    from tiny_rlhf.data import _template_ids

    h = hashlib.sha256()
    for e in examples:
        ids = _template_ids(tokenizer, e["chosen"][:-1], add_generation_prompt=True)
        h.update(json.dumps([e["id"], ids]).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# TRL settings
# --------------------------------------------------------------------------------------


def build_grpo_args(cfg: GRPORunConfig, output_dir: str | Path, device: str) -> Any:
    """TRL's GRPOConfig for this run. Every setting that changes the maths is set here
    explicitly, even when it equals TRL's default, so a TRL upgrade cannot change it."""
    from trl import GRPOConfig

    amp = cfg.amp is not False and device == "cuda"
    return GRPOConfig(
        output_dir=str(output_dir),
        # batching and generation
        num_generations=cfg.num_generations,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        steps_per_generation=cfg.steps_per_generation,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=1.0,
        # the objective
        beta=cfg.beta,
        epsilon=cfg.epsilon,
        loss_type=cfg.loss_type,
        scale_rewards=cfg.scale_rewards,
        num_iterations=1,  # one optimizer step per generated batch: on-policy, ratio = 1
        importance_sampling_level="token",
        use_bias_correction_kl=True,
        mask_truncated_completions=False,  # a rambling answer is penalised, not hidden
        multi_objective_aggregation="sum_then_normalize",
        # optimisation
        learning_rate=cfg.lr,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_steps=round(cfg.warmup_ratio * cfg.max_steps),
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        max_steps=cfg.max_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        fp16=amp,
        bf16=False,
        use_cpu=device == "cpu",
        seed=cfg.seed,
        data_seed=cfg.seed,
        shuffle_dataset=True,
        # bookkeeping: log every step, save nothing during training (the adapter is saved
        # once at the end), report to no external service
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        log_completions=False,
        disable_tqdm=True,
    )


TRL_SETTINGS = (
    "num_generations",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "steps_per_generation",
    "generation_batch_size",
    "max_completion_length",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "beta",
    "epsilon",
    "epsilon_high",
    "loss_type",
    "scale_rewards",
    "num_iterations",
    "importance_sampling_level",
    "use_bias_correction_kl",
    "mask_truncated_completions",
    "multi_objective_aggregation",
    "learning_rate",
    "lr_scheduler_type",
    "warmup_steps",
    "weight_decay",
    "max_grad_norm",
    "max_steps",
    "optim",
    "gradient_checkpointing",
    "fp16",
    "bf16",
    "seed",
    "data_seed",
    "shuffle_dataset",
    "use_vllm",
)


def trl_settings(args: Any) -> dict[str, Any]:
    """The TRL settings a run used, for run_info.json."""
    out = {}
    for name in TRL_SETTINGS:
        value = getattr(args, name, None)
        out[name] = value if isinstance(value, (int, float, str, bool, type(None))) else str(value)
    return out


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------


def log_row(logs: dict[str, Any], step: int, elapsed_s: float) -> dict[str, Any]:
    """One train.csv row from one TRL log record (blank where TRL logged nothing)."""
    row: dict[str, Any] = {"step": step}
    for column, key in TRL_LOG_KEYS.items():
        row[column] = logs.get(key, "")
    for key, column in zip(COMPONENTS, COMPONENT_COLUMNS, strict=True):
        row[column] = logs.get(f"trackb/{key}", "")
    row["elapsed_s"] = round(elapsed_s, 2)
    return row


def make_csv_callback(path: Path, start: float, rows: list[dict[str, Any]]) -> Any:
    """A TrainerCallback that appends a train.csv row for every training log record."""
    from transformers import TrainerCallback

    class CsvLogger(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any):
            if not logs or "reward" not in logs:  # the end-of-training summary has no reward
                return
            row = log_row(logs, state.global_step, time.perf_counter() - start)
            rows.append(row)
            new = not path.exists()
            with path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=TRAIN_COLUMNS)
                if new:
                    writer.writeheader()
                writer.writerow(row)

    return CsvLogger()


def make_time_limit_callback(start: float, max_minutes: float | None, stopped: list[int]) -> Any:
    """A TrainerCallback that ends training after the step during which max_minutes passed;
    the step at which it stopped is appended to `stopped`."""
    from transformers import TrainerCallback

    class TimeLimit(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any):
            if max_minutes is not None and time.perf_counter() - start > 60 * max_minutes:
                if state.global_step < state.max_steps:
                    stopped.append(int(state.global_step))
                control.should_training_stop = True
            return control

    return TimeLimit()


def make_sample_writer(path: Path, every: int, examples_by_id: dict[str, dict[str, Any]]) -> Any:
    """An on_scored hook for make_reward_func: on step 1 and every `every` steps, write each
    scored completion (with its reward and checks) to samples.jsonl."""

    def write(records: list[dict[str, Any]], trainer_state: Any) -> None:
        if not every or trainer_state is None:
            return
        # Generation for optimizer step k happens while global_step is still k - 1.
        step = int(trainer_state.global_step) + 1
        if step != 1 and step % every:
            return
        with path.open("a", encoding="utf-8") as f:
            for record in records:
                kind = examples_by_id[record["id"]]["kind"]
                f.write(json.dumps({"step": step, "kind": kind, **record}) + "\n")

    return write


# --------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------


def train(
    cfg: GRPORunConfig,
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    out_dir: str | Path,
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Run GRPO with TRL on the given examples; write the results and, if cfg.save_adapter,
    the LoRA adapter to checkpoint_dir; return run_info."""
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import set_seed
    from trl import GRPOTrainer

    from tiny_rlhf.trackb.evaluate import _stop_token_ids
    from tiny_rlhf.train_dpo import clear_stale_outputs, git_sha, library_versions

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    removed = clear_stale_outputs(out_dir)
    for name in ("train.csv", "samples.jsonl"):
        if (out_dir / name).exists():
            (out_dir / name).unlink()
            removed.append(name)
    if removed:
        print(f"removed outputs of an earlier run: {', '.join(removed)}")
    (out_dir / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))

    device = resolve_device(cfg.device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples_by_id = {e["id"]: e for e in examples}
    eos_ids = _stop_token_ids(model, tokenizer)
    reward = make_reward_func(
        examples_by_id,
        cfg.max_completion_length,
        eos_ids,
        cfg.reward_weights(),
        on_scored=make_sample_writer(out_dir / "samples.jsonl", cfg.sample_every, examples_by_id),
    )

    # TRL's output_dir is where the adapter goes; nothing is saved there during training.
    args = build_grpo_args(cfg, checkpoint_dir, device)
    set_seed(cfg.seed)  # the LoRA A matrices are drawn when the trainer wraps the model
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        task_type="CAUSAL_LM",
    )
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    stopped: list[int] = []
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=args,
        train_dataset=Dataset.from_list(prompt_rows(examples)),
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[
            make_csv_callback(out_dir / "train.csv", start, rows),
            make_time_limit_callback(start, cfg.max_minutes, stopped),
        ],
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    trainer.train()
    wall_time = time.perf_counter() - start

    policy = trainer.model
    if cfg.save_adapter:
        policy.save_pretrained(str(checkpoint_dir))

    trainable = [p for p in policy.parameters() if p.requires_grad]
    steps = int(trainer.state.global_step)
    run_info = {
        "run_name": cfg.run_name,
        "implementation": "trl_grpo_trainer",
        "wall_time_s": round(wall_time, 2),
        "optimizer_steps": steps,
        "planned_steps": cfg.max_steps,
        "stopped_by_time_limit": bool(stopped),
        "seconds_per_step": round(wall_time / max(steps, 1), 2),
        "device": device,
        "amp_fp16": bool(args.fp16),
        "param_dtype": str(next(policy.parameters()).dtype),
        # MiB = bytes / 2**20.
        "peak_gpu_memory_mib": (
            round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None
        ),
        "trainable_params": sum(p.numel() for p in trainable),
        "total_params": sum(p.numel() for p in policy.parameters()),
        "n_train_prompts": len(examples),
        "prompts_per_step": cfg.prompts_per_step(),
        "completions_per_step": cfg.completions_per_step(),
        "prompt_visits": steps * cfg.prompts_per_step(),
        "prompts_sha256": prompts_fingerprint(tokenizer, examples),
        "eos_token_ids": sorted(eos_ids),
        "reward_weights": cfg.reward_weights().to_dict(),
        "trl_settings": trl_settings(args),
        "final_logged": rows[-1] if rows else None,
        "seed": cfg.seed,
        "git_sha": git_sha(),
        "versions": library_versions(),
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    return run_info


def load_model_and_tokenizer(cfg: GRPORunConfig) -> tuple[Any, Any]:
    """The base model in float32 (fp16 autocast does the speed-up on cuda, as in the DPO
    runs; a T4 has no bfloat16) and its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=torch.float32)
    return model, tokenizer


def main(cfg: GRPORunConfig, results_root: str | Path = "results") -> dict[str, Any]:
    """Load the model and the Track B prompts (the DPO runs' selection), then train."""
    data_dir = Path(cfg.dataset_name)
    if not (data_dir / "train.jsonl").exists():
        raise FileNotFoundError(f"{data_dir}/train.jsonl not found (dataset_name is data/trackb)")
    model, tokenizer = load_model_and_tokenizer(cfg)
    examples = select_prompts(
        data_dir / "train.jsonl", tokenizer, cfg.max_length, cfg.n_train, cfg.seed
    )
    return train(
        cfg,
        model,
        tokenizer,
        examples,
        out_dir=Path(results_root) / cfg.run_name,
        checkpoint_dir=Path("checkpoints") / cfg.run_name,
    )
