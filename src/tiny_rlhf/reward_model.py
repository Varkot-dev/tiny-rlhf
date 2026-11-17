"""An explicit reward model r(x, y), trained with the Bradley-Terry loss (milestone M3).

This is step 2 of the classic RLHF pipeline, the step DPO skips:
    1. read the whole sequence (prompt + response) with a language-model backbone
    2. take the hidden state at the last real token and map it to one number, r(x, y)
    3. loss = -log sigmoid(r(x, chosen) - r(x, rejected))

DPO never builds this model. It uses the implicit reward
beta * (log pi(y|x) - log pi_ref(y|x)) instead. Training both on the same pairs and
scoring the same held-out pairs lets the two rankings be compared
(scripts/plot_reward_vs_dpo.py).
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from tiny_rlhf.data import PreferenceCollator, TokenizedPair, pairs_fingerprint, truncation_stats
from tiny_rlhf.losses import bradley_terry_loss
from tiny_rlhf.train_dpo import (
    git_sha,
    library_versions,
    lr_factor,
    resolve_device,
    to_device,
    use_amp,
)

LOG2 = math.log(2.0)
METRICS = ("loss", "reward_margin", "accuracy", "chosen_reward", "rejected_reward")
TRAIN_COLUMNS = ("step", "epoch", *METRICS, "lr", "grad_norm", "elapsed_s")
EVAL_COLUMNS = ("step", *METRICS, "n_pairs")
PAIR_COLUMNS = ("pair_index", "chosen_reward", "rejected_reward", "reward_margin")
BOOTSTRAP_RESAMPLES = 2000


@dataclass
class RewardConfig:
    """Everything that defines a reward-model run. Loaded from a YAML file under configs/."""

    run_name: str = "reward"
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    dataset_name: str = "trl-lib/ultrafeedback_binarized"
    n_train: int = 200
    n_eval: int = 50
    max_length: int | None = 256
    lr: float = 1e-4  # for the backbone (LoRA adapter or full weights)
    head_lr: float | None = None  # for the new linear head; None means the same as lr
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    epochs: int = 1
    batch_size: int = 4  # pairs per micro-batch; the model sees 2 * batch_size sequences
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    eval_every: int = 10
    eval_batch_size: int = 8
    seed: int = 0
    shuffle: bool = True  # false: go through the pairs in dataset order every epoch
    # What is trained besides the head. use_lora: a LoRA adapter on the backbone.
    # Otherwise train_backbone decides between all backbone weights and none of them
    # (none = a linear probe on frozen features).
    use_lora: bool = True
    train_backbone: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    amp: bool | str = "auto"  # "auto" or true: fp16 autocast on cuda. false: float32.
    device: str = "auto"
    max_steps: int | None = None


def load_reward_config(path: str | Path, overrides: dict[str, Any] | None = None) -> RewardConfig:
    """Read a YAML config and apply `key: value` overrides. Unknown keys are an error."""
    values = yaml.safe_load(Path(path).read_text()) or {}
    values.update(overrides or {})
    known = {f.name for f in dataclasses.fields(RewardConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    cfg = RewardConfig(**values)
    if cfg.amp not in ("auto", True, False):
        raise ValueError(f"amp must be auto, true or false, got {cfg.amp!r}")
    if cfg.eval_every < 1 or cfg.batch_size < 1 or cfg.grad_accum < 1:
        raise ValueError("eval_every, batch_size and grad_accum must be at least 1")
    return cfg


class RewardModel(nn.Module):
    """A language-model backbone plus a linear head that outputs one scalar per sequence.

    Args:
        backbone: a causal transformer without its vocabulary head. Called with
            input_ids and attention_mask, it must return an object whose
            `last_hidden_state` is [batch, seq_len, hidden].
        hidden_size: width of the backbone's hidden states.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, 1)
        # Zero head: every reward starts at exactly 0, so the first loss is exactly
        # log 2, the same hand check as DPO. A random head would start near log 2 but
        # not on it, because the two responses have different hidden states.
        # Starting at zero does not block learning. The gradient of the loss with
        # respect to the head weight is -(1/2) * (h_chosen - h_rejected) at step 0,
        # which is not zero. What does start at zero is the gradient into the
        # backbone (it flows through the head weight), so the backbone only begins
        # to move after the head's first update.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Reward for each row. Returns a float32 tensor of shape [batch]."""
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # A causal model reads left to right: the state at position t has seen tokens
        # 0..t and nothing after. Only the state at the last real token has seen the
        # whole prompt and the whole response, so that is the one to score. Rows are
        # padded on the right, so the last real token sits at (number of real tokens - 1),
        # not at the last column.
        last = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        # The head runs in float32 even when the backbone ran under fp16 autocast.
        with torch.autocast(pooled.device.type, enabled=False):
            return self.head(pooled.float()).squeeze(-1)


def pair_rewards(model: nn.Module, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """r(x, chosen) and r(x, rejected), each [B], from one forward pass over the 2B rows."""
    rewards = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    chosen, rejected = rewards.chunk(2, dim=0)
    return chosen, rejected


def batch_metrics(loss: Tensor, chosen: Tensor, rejected: Tensor) -> dict[str, float]:
    margin = (chosen - rejected).detach()
    return {
        "loss": loss.item(),
        "reward_margin": margin.mean().item(),
        # Ties count as wrong, the same rule as the DPO preference accuracy.
        "accuracy": (margin > 0).float().mean().item(),
        "chosen_reward": chosen.detach().mean().item(),
        "rejected_reward": rejected.detach().mean().item(),
    }


def weighted_mean(rows: list[dict[str, float]], weights: list[int]) -> dict[str, float]:
    total = sum(weights)
    return {k: sum(r[k] * w for r, w in zip(rows, weights, strict=True)) / total for k in METRICS}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    eval_batches: list[dict[str, Tensor]],
    device: torch.device,
    amp: bool = False,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Score held-out pairs. Returns (mean metrics, one row per pair in dataset order)."""
    was_training = model.training
    model.eval()
    per_batch, weights, pair_rows = [], [], []
    for cpu_batch in eval_batches:
        batch = to_device(cpu_batch, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            chosen, rejected = pair_rewards(model, batch)
        loss = bradley_terry_loss(chosen, rejected)
        per_batch.append(batch_metrics(loss, chosen, rejected))
        weights.append(len(chosen))
        for i in range(len(chosen)):
            pair_rows.append(
                {
                    "pair_index": len(pair_rows),
                    "chosen_reward": chosen[i].item(),
                    "rejected_reward": rejected[i].item(),
                    "reward_margin": (chosen[i] - rejected[i]).item(),
                }
            )
    model.train(was_training)
    metrics = weighted_mean(per_batch, weights)
    metrics["n_pairs"] = sum(weights)
    return metrics, pair_rows


def bootstrap_ci(
    values: list[float] | Tensor,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of `values` (for accuracy: 0s and 1s).

    Resample the pairs with replacement n_resamples times, take the mean each time,
    and report the middle `level` share of those means. It answers: how much would
    this number move if a different set of held-out pairs of the same size had been drawn?
    """
    x = torch.as_tensor(values, dtype=torch.float64)
    if x.numel() == 0:
        raise ValueError("cannot bootstrap an empty sample")
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(0, x.numel(), (n_resamples, x.numel()), generator=generator)
    means = x[index].mean(dim=1)
    tail = (1.0 - level) / 2.0
    lo, hi = torch.quantile(means, torch.tensor([tail, 1.0 - tail], dtype=torch.float64))
    return lo.item(), hi.item()


def plan_steps(n_pairs: int, cfg: RewardConfig) -> tuple[int, int]:
    """(optimizer steps per epoch, total optimizer steps after the max_steps cap)."""
    micro_batches = math.ceil(n_pairs / cfg.batch_size)
    per_epoch = math.ceil(micro_batches / cfg.grad_accum)
    total = per_epoch * cfg.epochs
    if cfg.max_steps is not None:
        total = min(total, cfg.max_steps)
    return per_epoch, total


def epoch_steps(
    n_pairs: int, cfg: RewardConfig, generator: torch.Generator
) -> list[list[list[int]]]:
    """steps[s][m] = pair indices in micro-batch m of optimizer step s, for one epoch."""
    if cfg.shuffle:
        order = torch.randperm(n_pairs, generator=generator).tolist()
    else:
        order = list(range(n_pairs))
    micro = [order[i : i + cfg.batch_size] for i in range(0, n_pairs, cfg.batch_size)]
    return [micro[i : i + cfg.grad_accum] for i in range(0, len(micro), cfg.grad_accum)]


def train(
    cfg: RewardConfig,
    model: RewardModel,
    train_pairs: list[TokenizedPair],
    eval_pairs: list[TokenizedPair],
    pad_token_id: int,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Train the reward model on preference pairs. Writes logs to out_dir.

    Whatever has requires_grad=True in `model` is trained; the caller decides that
    (see `main`). Returns the run_info dict that is also written to run_info.json.
    """
    if not train_pairs or not eval_pairs:
        raise ValueError("train_pairs and eval_pairs must both be non-empty")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))

    device = resolve_device(cfg.device)
    amp = use_amp(cfg, device)
    torch.manual_seed(cfg.seed)  # LoRA dropout draws from the global generator
    shuffle_generator = torch.Generator().manual_seed(cfg.seed)
    model.to(device).train()

    collate = PreferenceCollator(pad_token_id=pad_token_id, max_length=cfg.max_length)
    eval_batches = [
        collate(eval_pairs[i : i + cfg.eval_batch_size])
        for i in range(0, len(eval_pairs), cfg.eval_batch_size)
    ]

    # The head is new and starts at zero; the backbone is pretrained. They can take
    # different learning rates, so they go in separate parameter groups.
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    trainable = head_params + backbone_params
    groups = [{"params": head_params, "lr": cfg.head_lr if cfg.head_lr is not None else cfg.lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": cfg.lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    steps_per_epoch, total_steps = plan_steps(len(train_pairs), cfg)
    warmup_steps = math.ceil(cfg.warmup_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, total_steps, warmup_steps)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    train_file = (out_dir / "train.csv").open("w", newline="")
    eval_file = (out_dir / "eval.csv").open("w", newline="")
    train_log = csv.DictWriter(train_file, fieldnames=TRAIN_COLUMNS)
    eval_log = csv.DictWriter(eval_file, fieldnames=EVAL_COLUMNS)
    train_log.writeheader()
    eval_log.writeheader()

    def run_eval(step: int) -> tuple[dict[str, float], list[dict[str, float]]]:
        metrics, pair_rows = evaluate(model, eval_batches, device, amp)
        eval_log.writerow({"step": step, **metrics})
        eval_file.flush()
        print(
            f"eval  step {step:5d}  loss {metrics['loss']:.4f}  "
            f"margin {metrics['reward_margin']:+.4f}  acc {metrics['accuracy']:.3f}"
        )
        return metrics, pair_rows

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()

    first_metrics, pair_rows = run_eval(0)
    final_metrics = first_metrics
    if not math.isfinite(first_metrics["loss"]):
        raise RuntimeError(f"step-0 eval loss is {first_metrics['loss']}, expected a finite number")

    step = 0
    last_eval_step = 0
    for epoch in range(cfg.epochs):
        for micro_batches in epoch_steps(len(train_pairs), cfg, shuffle_generator):
            if step >= total_steps:
                break
            n_step_pairs = sum(len(m) for m in micro_batches)
            per_batch, weights = [], []
            for indices in micro_batches:
                batch = to_device(collate([train_pairs[i] for i in indices]), device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    chosen, rejected = pair_rewards(model, batch)
                loss = bradley_terry_loss(chosen, rejected)
                # Weight each micro-batch by its share of the step's pairs, so the
                # accumulated gradient is the gradient of the mean loss over the step.
                scaler.scale(loss * len(indices) / n_step_pairs).backward()
                per_batch.append(batch_metrics(loss, chosen, rejected))
                weights.append(len(indices))

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
            if not math.isfinite(grad_norm.item()):
                raise RuntimeError(
                    f"non-finite gradient norm at optimizer step {step + 1}. "
                    "Try amp: false, a lower lr, or a shorter max_length."
                )
            # Logged lr is the backbone lr when a backbone is trained, else the head lr.
            lr_used = scheduler.get_last_lr()[-1]
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            train_log.writerow(
                {
                    "step": step,
                    "epoch": epoch,
                    **weighted_mean(per_batch, weights),
                    "lr": lr_used,
                    "grad_norm": grad_norm.item(),
                    "elapsed_s": round(time.perf_counter() - start, 3),
                }
            )
            train_file.flush()
            if step % cfg.eval_every == 0:
                final_metrics, pair_rows = run_eval(step)
                last_eval_step = step

    if last_eval_step != step:
        final_metrics, pair_rows = run_eval(step)
    wall_time = time.perf_counter() - start
    train_file.close()
    eval_file.close()

    with (out_dir / "eval_pairs_final.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        writer.writerows(pair_rows)

    correct = [1.0 if row["reward_margin"] > 0 else 0.0 for row in pair_rows]
    ci_low, ci_high = bootstrap_ci(correct, BOOTSTRAP_RESAMPLES, cfg.seed)
    run_info = {
        "run_name": cfg.run_name,
        "wall_time_s": round(wall_time, 2),
        "optimizer_steps": step,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup_steps,
        "first_eval_loss": first_metrics["loss"],
        "final_eval_loss": final_metrics["loss"],
        "heldout_accuracy": sum(correct) / len(correct),
        "heldout_accuracy_ci95": [ci_low, ci_high],
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": cfg.seed,
        "eval_pairs_sha256": pairs_fingerprint(eval_pairs, cfg.max_length),
        "device": str(device),
        "amp_fp16": amp,
        "param_dtype": str(next(model.parameters()).dtype),
        "peak_gpu_memory_mb": (
            round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
            if device.type == "cuda"
            else None
        ),
        "trainable_params": sum(p.numel() for p in trainable),
        "head_params": sum(p.numel() for p in head_params),
        "total_params": sum(p.numel() for p in model.parameters()),
        "n_train_pairs": len(train_pairs),
        "n_eval_pairs": len(eval_pairs),
        "truncation_train": truncation_stats(train_pairs, cfg.max_length),
        "truncation_eval": truncation_stats(eval_pairs, cfg.max_length),
        "seed": cfg.seed,
        "git_sha": git_sha(),
        "versions": library_versions(),
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    return run_info


def main(cfg: RewardConfig, results_root: str | Path = "results") -> dict[str, Any]:
    """Load tokenizer, backbone and data from the Hugging Face Hub, then train."""
    from transformers import AutoModel, AutoTokenizer

    from tiny_rlhf.data import load_pairs

    torch.manual_seed(cfg.seed)  # LoRA's A matrices are randomly initialised
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # AutoModel loads the transformer without its vocabulary head: the reward model
    # needs hidden states, not next-token logits. float32 for the same reason as in DPO.
    backbone = AutoModel.from_pretrained(cfg.model_name, dtype=torch.float32)
    hidden_size = backbone.config.hidden_size

    if cfg.use_lora:
        from peft import LoraConfig, get_peft_model

        backbone = get_peft_model(
            backbone,
            LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=list(cfg.lora_target_modules),
                task_type="FEATURE_EXTRACTION",
            ),
        )
    elif not cfg.train_backbone:
        for p in backbone.parameters():
            p.requires_grad_(False)

    model = RewardModel(backbone, hidden_size)

    # Same call as the DPO run. With the same dataset, n_eval, seed, max_length and
    # tokenizer, the held-out pairs are identical and in the same order.
    train_pairs = load_pairs(
        cfg.dataset_name, "train", cfg.n_train, cfg.seed, tokenizer, cfg.max_length
    )
    eval_pairs = load_pairs(
        cfg.dataset_name, "test", cfg.n_eval, cfg.seed, tokenizer, cfg.max_length
    )
    return train(
        cfg,
        model,
        train_pairs,
        eval_pairs,
        pad_token_id=tokenizer.pad_token_id,
        out_dir=Path(results_root) / cfg.run_name,
    )
