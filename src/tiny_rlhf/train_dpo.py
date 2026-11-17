"""Hand-written DPO training loop in plain PyTorch (milestone M1).

The whole method in five lines:
    1. load two copies of the same model: `policy` (trained) and `reference` (frozen)
    2. for a batch of (prompt, chosen, rejected): get log pi(y|x) for both responses
       from both models, summed over response tokens only
    3. loss = -log sigmoid(beta * [(pol_chosen - ref_chosen) - (pol_rejected - ref_rejected)])
    4. backpropagate into the policy only
    5. log the loss, the reward margin, preference accuracy and the chosen log-probability

There is no reward model, no sampling and no RL loop. It is supervised learning on pairs.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from tiny_rlhf.data import PreferenceCollator, TokenizedPair, pairs_fingerprint, truncation_stats
from tiny_rlhf.losses import dpo_loss, sequence_logprob

LOG2 = math.log(2.0)
METRICS = (
    "loss",
    "reward_margin",
    "accuracy",
    "chosen_logp",
    "rejected_logp",
    "chosen_reward",
    "rejected_reward",
)
TRAIN_COLUMNS = ("step", "epoch", *METRICS, "lr", "grad_norm", "elapsed_s")
EVAL_COLUMNS = ("step", *METRICS, "n_pairs")
PAIR_COLUMNS = (
    "pair_index",
    "chosen_logp",
    "rejected_logp",
    "ref_chosen_logp",
    "ref_rejected_logp",
    "chosen_reward",
    "rejected_reward",
    "reward_margin",
)


@dataclass
class DPOConfig:
    """Everything that defines a run. Loaded from a YAML file under configs/."""

    run_name: str = "dpo"
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    dataset_name: str = "trl-lib/ultrafeedback_binarized"
    n_train: int = 200
    n_eval: int = 50
    max_length: int | None = 256  # tokens per sequence (prompt + response); longer is cut
    beta: float = 0.1
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    epochs: int = 1
    batch_size: int = 4  # pairs per micro-batch; the model sees 2 * batch_size sequences
    grad_accum: int = 1  # micro-batches per optimizer step
    max_grad_norm: float = 1.0
    eval_every: int = 10  # in optimizer steps
    eval_batch_size: int = 8
    seed: int = 0
    # false: visit pairs in dataset order every epoch. Used by the parity run, so that
    # this loop and TRL see identical batches and can be compared step by step.
    shuffle: bool = True
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    # "auto" or true: fp16 autocast on cuda (never on cpu or mps). false: always float32.
    # YAML reads a bare on/off as a boolean, so booleans are the accepted spelling.
    amp: bool | str = "auto"
    device: str = "auto"
    max_steps: int | None = None  # cap on optimizer steps, for smoke tests
    save_adapter: bool = True


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> DPOConfig:
    """Read a YAML config and apply `key: value` overrides. Unknown keys are an error."""
    values = yaml.safe_load(Path(path).read_text()) or {}
    values.update(overrides or {})
    known = {f.name for f in dataclasses.fields(DPOConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    cfg = DPOConfig(**values)
    if cfg.amp not in ("auto", True, False):
        raise ValueError(f"amp must be auto, true or false, got {cfg.amp!r}")
    if cfg.eval_every < 1 or cfg.batch_size < 1 or cfg.grad_accum < 1:
        raise ValueError("eval_every, batch_size and grad_accum must be at least 1")
    return cfg


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def use_amp(cfg: DPOConfig, device: torch.device) -> bool:
    """Half-precision forward passes roughly halve memory and time on a T4."""
    if cfg.amp is False:
        return False
    return device.type == "cuda"


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """Multiplier on the base lr for optimizer step number `step` (0-based).

    Linear warmup from 0 to 1 over `warmup_steps`, then linear decay to 0 at
    `total_steps`. This is the Hugging Face Trainer's default "linear" schedule,
    chosen so the TRL parity run in M2 uses the same one.
    """
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def pair_logps(model: nn.Module, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """log pi(chosen | prompt) and log pi(rejected | prompt), each of shape [B].

    The batch holds 2B rows: the B prompt+chosen sequences, then the B prompt+rejected
    sequences. One forward pass scores both.
    """
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    logps = sequence_logprob(logits, batch["input_ids"], batch["response_mask"])
    chosen, rejected = logps.chunk(2, dim=0)
    return chosen, rejected


def to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def batch_metrics(out: Any, chosen_logps: Tensor, rejected_logps: Tensor) -> dict[str, float]:
    return {
        "loss": out.loss.item(),
        "reward_margin": out.reward_margin.mean().item(),
        "accuracy": out.accuracy.item(),
        # The absolute log-probability of the preferred response. DPO can lower it
        # while still widening the margin, so it is logged next to the margin.
        "chosen_logp": chosen_logps.detach().mean().item(),
        "rejected_logp": rejected_logps.detach().mean().item(),
        "chosen_reward": out.chosen_rewards.mean().item(),
        "rejected_reward": out.rejected_rewards.mean().item(),
    }


def weighted_mean(rows: list[dict[str, float]], weights: list[int]) -> dict[str, float]:
    """Mean of per-batch metrics weighted by pairs per batch, so a short last batch
    does not count as much as a full one."""
    total = sum(weights)
    return {k: sum(r[k] * w for r, w in zip(rows, weights, strict=True)) / total for k in METRICS}


@torch.no_grad()
def evaluate(
    policy: nn.Module,
    reference: nn.Module,
    eval_batches: list[dict[str, Tensor]],
    beta: float,
    device: torch.device,
    amp: bool = False,
    ref_logps: list[tuple[Tensor, Tensor]] | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Score held-out pairs. Returns (mean metrics, one row per pair).

    The reference never changes, so its log-probabilities on the held-out pairs can be
    computed once with `reference_logps` and passed in as `ref_logps`.
    """
    was_training = policy.training
    policy.eval()  # turns LoRA dropout off, so step 0 compares identical functions
    per_batch, weights, pair_rows = [], [], []
    for i, cpu_batch in enumerate(eval_batches):
        batch = to_device(cpu_batch, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            pol_c, pol_r = pair_logps(policy, batch)
            if ref_logps is None:
                ref_c, ref_r = pair_logps(reference, batch)
        if ref_logps is not None:
            ref_c, ref_r = (t.to(device) for t in ref_logps[i])
        out = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=beta)
        per_batch.append(batch_metrics(out, pol_c, pol_r))
        weights.append(len(pol_c))
        for i in range(len(pol_c)):
            pair_rows.append(
                {
                    "pair_index": len(pair_rows),
                    "chosen_logp": pol_c[i].item(),
                    "rejected_logp": pol_r[i].item(),
                    "ref_chosen_logp": ref_c[i].item(),
                    "ref_rejected_logp": ref_r[i].item(),
                    "chosen_reward": out.chosen_rewards[i].item(),
                    "rejected_reward": out.rejected_rewards[i].item(),
                    "reward_margin": out.reward_margin[i].item(),
                }
            )
    policy.train(was_training)
    metrics = weighted_mean(per_batch, weights)
    metrics["n_pairs"] = sum(weights)
    return metrics, pair_rows


@torch.no_grad()
def reference_logps(
    reference: nn.Module, batches: list[dict[str, Tensor]], device: torch.device, amp: bool = False
) -> list[tuple[Tensor, Tensor]]:
    """The frozen reference's (chosen, rejected) log-probabilities for each batch."""
    out = []
    for cpu_batch in batches:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            ref_c, ref_r = pair_logps(reference, to_device(cpu_batch, device))
        out.append((ref_c.cpu(), ref_r.cpu()))
    return out


def plan_steps(n_pairs: int, cfg: DPOConfig) -> tuple[int, int]:
    """(optimizer steps per epoch, total optimizer steps after the max_steps cap)."""
    micro_batches = math.ceil(n_pairs / cfg.batch_size)
    per_epoch = math.ceil(micro_batches / cfg.grad_accum)
    total = per_epoch * cfg.epochs
    if cfg.max_steps is not None:
        total = min(total, cfg.max_steps)
    return per_epoch, total


def epoch_steps(n_pairs: int, cfg: DPOConfig, generator: torch.Generator) -> list[list[list[int]]]:
    """Order pair indices, cut into micro-batches, group micro-batches into optimizer steps.

    Returns steps[s][m] = list of pair indices in micro-batch m of optimizer step s.
    Every pair is used once per epoch; the last step of an epoch may be smaller.
    The order is a fresh seeded shuffle each epoch, or dataset order if cfg.shuffle is false.
    """
    if cfg.shuffle:
        order = torch.randperm(n_pairs, generator=generator).tolist()
    else:
        order = list(range(n_pairs))
    micro = [order[i : i + cfg.batch_size] for i in range(0, n_pairs, cfg.batch_size)]
    return [micro[i : i + cfg.grad_accum] for i in range(0, len(micro), cfg.grad_accum)]


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def library_versions() -> dict[str, str]:
    import importlib.metadata as md

    versions = {}
    for name in ("torch", "transformers", "peft", "datasets", "trl"):
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def train(
    cfg: DPOConfig,
    policy: nn.Module,
    reference: nn.Module,
    train_pairs: list[TokenizedPair],
    eval_pairs: list[TokenizedPair],
    pad_token_id: int,
    out_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train `policy` with DPO against the frozen `reference`. Writes logs to out_dir.

    Returns the run_info dict that is also written to out_dir/run_info.json.
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

    policy.to(device).train()
    # The reference is the anchor the policy is measured against. It never changes.
    reference.to(device).eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    collate = PreferenceCollator(pad_token_id=pad_token_id, max_length=cfg.max_length)
    eval_batches = [
        collate(eval_pairs[i : i + cfg.eval_batch_size])
        for i in range(0, len(eval_pairs), cfg.eval_batch_size)
    ]

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch, total_steps = plan_steps(len(train_pairs), cfg)
    warmup_steps = math.ceil(cfg.warmup_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, total_steps, warmup_steps)
    )
    # fp16 gradients can underflow to zero; the scaler multiplies the loss up before
    # backward and divides the gradients back down before the optimizer step.
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    train_file = (out_dir / "train.csv").open("w", newline="")
    eval_file = (out_dir / "eval.csv").open("w", newline="")
    train_log = csv.DictWriter(train_file, fieldnames=TRAIN_COLUMNS)
    eval_log = csv.DictWriter(eval_file, fieldnames=EVAL_COLUMNS)
    train_log.writeheader()
    eval_log.writeheader()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    eval_ref_logps = reference_logps(reference, eval_batches, device, amp)

    def run_eval(step: int) -> tuple[dict[str, float], list[dict[str, float]]]:
        metrics, pair_rows = evaluate(
            policy, reference, eval_batches, cfg.beta, device, amp, eval_ref_logps
        )
        eval_log.writerow({"step": step, **metrics})
        eval_file.flush()
        print(
            f"eval  step {step:5d}  loss {metrics['loss']:.4f}  "
            f"margin {metrics['reward_margin']:+.4f}  acc {metrics['accuracy']:.3f}  "
            f"chosen_logp {metrics['chosen_logp']:.2f}"
        )
        return metrics, pair_rows

    # The sanity check from the plan: before any update the policy equals the
    # reference, the bracket is 0 for every pair, and the loss must be log 2.
    first_metrics, pair_rows = run_eval(0)
    worst_margin = max(abs(row["reward_margin"]) for row in pair_rows)
    first_loss = first_metrics["loss"]
    # `not (x <= tol)` instead of `x > tol` so that a nan fails the check too.
    if not (abs(first_loss - LOG2) <= 1e-3 and worst_margin <= 1e-2):
        raise RuntimeError(
            f"largest step-0 reward margin over held-out pairs is {worst_margin:.6f}, and the "
            f"step-0 eval loss is {first_loss:.6f}; expected 0 and log 2 = {LOG2:.6f}. "
            "The policy and the reference disagree before training: check that both were "
            "loaded from the same weights and see the same tokens and masks."
        )

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
                    pol_c, pol_r = pair_logps(policy, batch)
                    with torch.no_grad():
                        ref_c, ref_r = pair_logps(reference, batch)
                out = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=cfg.beta)
                # Weight each micro-batch by its share of the step's pairs, so the
                # accumulated gradient is the gradient of the mean loss over the step.
                scaler.scale(out.loss * len(indices) / n_step_pairs).backward()
                per_batch.append(batch_metrics(out, pol_c, pol_r))
                weights.append(len(indices))

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
            if not math.isfinite(grad_norm.item()):
                # GradScaler would silently skip this update. Stop instead, so a run
                # never logs steps that did not happen.
                raise RuntimeError(
                    f"non-finite gradient norm at optimizer step {step + 1}. "
                    "Try amp: false, a lower lr, or a shorter max_length."
                )
            lr_used = scheduler.get_last_lr()[0]
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
                _, pair_rows = run_eval(step)
                last_eval_step = step

    if last_eval_step != step:
        _, pair_rows = run_eval(step)
    wall_time = time.perf_counter() - start
    train_file.close()
    eval_file.close()

    with (out_dir / "eval_pairs_final.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        writer.writerows(pair_rows)

    if checkpoint_dir is not None and cfg.save_adapter and hasattr(policy, "save_pretrained"):
        policy.save_pretrained(str(checkpoint_dir))

    run_info = {
        "run_name": cfg.run_name,
        "implementation": "handwritten",
        "wall_time_s": round(wall_time, 2),
        "optimizer_steps": step,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup_steps,
        "first_eval_loss": first_metrics["loss"],
        "device": str(device),
        "amp_fp16": amp,
        "param_dtype": str(next(policy.parameters()).dtype),
        "peak_gpu_memory_mb": (
            round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
            if device.type == "cuda"
            else None
        ),
        "trainable_params": sum(p.numel() for p in trainable),
        "total_params": sum(p.numel() for p in policy.parameters()),
        "n_train_pairs": len(train_pairs),
        "n_eval_pairs": len(eval_pairs),
        "truncation_train": truncation_stats(train_pairs, cfg.max_length),
        "truncation_eval": truncation_stats(eval_pairs, cfg.max_length),
        # Lets another run (the reward model, the TRL run) prove it scored the same
        # held-out pairs in the same order before per-pair rows are joined.
        "eval_pairs_sha256": pairs_fingerprint(eval_pairs, cfg.max_length),
        "seed": cfg.seed,
        "git_sha": git_sha(),
        "versions": library_versions(),
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    return run_info


def load_models(cfg: DPOConfig) -> tuple[Any, nn.Module, nn.Module]:
    """Load the tokenizer, the trainable policy and the frozen reference from the Hub.

    Shared with the TRL parity run (train_trl.py) so both start from identical weights,
    including the random LoRA initialisation.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(cfg.seed)  # LoRA's A matrices are randomly initialised
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Two separate loads of the same weights. The reference is never touched again.
    # float32 is asked for explicitly: some checkpoints are stored in bfloat16 and
    # would otherwise load that way. Half precision is applied by autocast instead.
    policy = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=torch.float32)
    reference = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=torch.float32)

    if cfg.use_lora:
        from peft import LoraConfig, get_peft_model

        # LoRA freezes every original weight W and learns a low-rank update B @ A next
        # to it. B starts at zero, so at step 0 the policy still equals the reference.
        policy = get_peft_model(
            policy,
            LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=list(cfg.lora_target_modules),
                task_type="CAUSAL_LM",
            ),
        )
    return tokenizer, policy, reference


def main(cfg: DPOConfig, results_root: str | Path = "results") -> dict[str, Any]:
    """Load tokenizer, both models and the data from the Hugging Face Hub, then train."""
    from tiny_rlhf.data import load_pairs

    tokenizer, policy, reference = load_models(cfg)
    train_pairs = load_pairs(
        cfg.dataset_name, "train", cfg.n_train, cfg.seed, tokenizer, cfg.max_length
    )
    eval_pairs = load_pairs(
        cfg.dataset_name, "test", cfg.n_eval, cfg.seed, tokenizer, cfg.max_length
    )

    return train(
        cfg,
        policy,
        reference,
        train_pairs,
        eval_pairs,
        pad_token_id=tokenizer.pad_token_id,
        out_dir=Path(results_root) / cfg.run_name,
        checkpoint_dir=Path("checkpoints") / cfg.run_name,
    )
