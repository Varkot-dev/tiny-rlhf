"""The same DPO run through Hugging Face TRL's DPOTrainer (milestone M2).

This file contains no DPO math. It takes the same config, the same starting weights
and the same pairs as the hand-written loop in train_dpo.py, hands them to TRL, and
writes TRL's own logged numbers in the same CSV layout. If the hand-written loop is
right, the two sets of curves agree.

Written against trl 1.18.0 and transformers 5.23.0. Settings that had to be set
explicitly to get a like-for-like run are marked "parity:" below.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from tiny_rlhf.data import (
    PreferenceCollator,
    TokenizedPair,
    pairs_fingerprint,
    tokenize_pair,
    truncation_stats,
)
from tiny_rlhf.train_dpo import (
    EVAL_COLUMNS,
    METRICS,
    PAIR_COLUMNS,
    TRAIN_COLUMNS,
    DPOConfig,
    clear_stale_outputs,
    evaluate,
    git_sha,
    library_versions,
    load_models,
    plan_steps,
    resolve_device,
    use_amp,
)

# Our column name -> the key TRL logs (an "eval_" prefix is added for held-out rows).
TRL_KEYS = {
    "loss": "loss",
    "reward_margin": "rewards/margins",
    "accuracy": "rewards/accuracies",
    "chosen_logp": "logps/chosen",
    "rejected_logp": "logps/rejected",
    "chosen_reward": "rewards/chosen",
    "rejected_reward": "rewards/rejected",
}


def select_rows(
    dataset_name: str, split: str, n: int, seed: int, tokenizer: Any, max_length: int | None
) -> tuple[Any, list[TokenizedPair]]:
    """The dataset rows and tokenized pairs that tiny_rlhf.data.load_pairs would return.

    Same shuffle, same first n rows, same drop rule (prompt alone fills max_length).
    TRL gets the raw rows and tokenizes them itself; the pairs are our tokenization
    of the same rows, used to check that the two agree.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split).shuffle(seed=seed)
    dataset = dataset.select(range(min(n, len(dataset))))
    return keep_rows_that_fit(dataset, tokenizer, max_length)


def keep_rows_that_fit(
    dataset: Any, tokenizer: Any, max_length: int | None
) -> tuple[Any, list[TokenizedPair]]:
    pairs = [tokenize_pair(tokenizer, row["chosen"], row["rejected"]) for row in dataset]
    if max_length is not None:
        kept = [i for i, p in enumerate(pairs) if len(p.prompt_ids) < max_length]
        dataset = dataset.select(kept)
        pairs = [pairs[i] for i in kept]
    return dataset, pairs


def tokenization_matches(trl_dataset: Any, pairs: list[TokenizedPair]) -> bool:
    """True if TRL's tokenized dataset holds exactly our token ids, row for row."""
    if len(trl_dataset) != len(pairs):
        return False
    return all(
        row["prompt_ids"] == pair.prompt_ids
        and row["chosen_ids"] == pair.chosen_ids
        and row["rejected_ids"] == pair.rejected_ids
        for row, pair in zip(trl_dataset, pairs, strict=True)
    )


def build_trl_args(cfg: DPOConfig, n_train_pairs: int, output_dir: str | Path) -> Any:
    """Translate our DPOConfig into TRL's DPOConfig, field by field."""
    from trl import DPOConfig as TRLConfig

    device = resolve_device(cfg.device)
    _, total_steps = plan_steps(n_train_pairs, cfg)
    return TRLConfig(
        output_dir=str(output_dir),
        beta=cfg.beta,
        # The plain DPO loss, plus TRL's "sft" term (mean NLL per preferred-response token)
        # when the config asks for it: the same objective as train_dpo with sft_weight.
        loss_type=["sigmoid", "sft"] if cfg.sft_weight > 0 else ["sigmoid"],
        loss_weights=[1.0, cfg.sft_weight] if cfg.sft_weight > 0 else None,
        max_length=cfg.max_length,
        truncation_mode="keep_start",
        learning_rate=cfg.lr,
        lr_scheduler_type="linear",
        # parity: an integer, computed the way our loop does it (ceil of ratio * steps).
        warmup_steps=math.ceil(cfg.warmup_ratio * total_steps),
        weight_decay=cfg.weight_decay,
        # parity: transformers 5 would otherwise pick the fused AdamW implementation.
        optim="adamw_torch",
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        max_grad_norm=cfg.max_grad_norm,
        num_train_epochs=cfg.epochs,
        max_steps=total_steps if cfg.max_steps is not None else -1,
        # parity: TRL turns bf16 on by default when fp16 is off. We use fp16 on cuda only.
        fp16=use_amp(cfg, device),
        bf16=False,
        # parity: TRL zeroes every dropout layer by default. Ours stay as configured.
        disable_dropout=False,
        # parity: dataset order instead of a random sampler when shuffle is false.
        train_sampling_strategy="random" if cfg.shuffle else "sequential",
        seed=cfg.seed,
        data_seed=cfg.seed,
        use_cpu=device.type == "cpu",
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=cfg.eval_every,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        dataloader_num_workers=0,
    )


def train_with_trl(
    cfg: DPOConfig,
    policy: nn.Module,
    reference: nn.Module,
    tokenizer: Any,
    train_rows: Any,
    eval_rows: Any,
    train_pairs: list[TokenizedPair],
    eval_pairs: list[TokenizedPair],
    out_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train `policy` with TRL's DPOTrainer and write the same files as train_dpo.train."""
    from transformers import PrinterCallback, TrainerCallback
    from trl import DPOTrainer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_stale_outputs(out_dir)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))
    work_dir = Path(checkpoint_dir) if checkpoint_dir is not None else out_dir / "_trl_work"

    device = resolve_device(cfg.device)
    amp = use_amp(cfg, device)
    steps_per_epoch, total_steps = plan_steps(len(train_pairs), cfg)
    args = build_trl_args(cfg, len(train_pairs), work_dir / "trl_output")

    # TRL logs one combined loss. With an sft term that is DPO + sft_weight * NLL, which
    # the hand-written loop writes as "total_loss" next to the DPO-only "loss". TRL has no
    # DPO-only number to report, so in that case "loss" is left empty and the combined
    # value goes to "total_loss" (train and eval alike).
    use_sft = cfg.sft_weight > 0
    loss_name = "total_loss" if use_sft else "loss"
    train_file = (out_dir / "train.csv").open("w", newline="")
    eval_file = (out_dir / "eval.csv").open("w", newline="")
    train_log = csv.DictWriter(
        train_file, fieldnames=TRAIN_COLUMNS + ("total_loss",) if use_sft else TRAIN_COLUMNS
    )
    eval_log = csv.DictWriter(
        eval_file, fieldnames=EVAL_COLUMNS + ("total_loss",) if use_sft else EVAL_COLUMNS
    )
    train_log.writeheader()
    eval_log.writeheader()
    eval_steps_seen: list[int] = []
    first_eval: dict[str, float] = {}
    skipped_steps: list[int] = []
    start = time.perf_counter()

    class CsvLogger(TrainerCallback):
        """Copy what TRL logs into our CSV layout. No number is recomputed here."""

        lr_used = float("nan")

        def on_pre_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
            # The Trainer logs the lr after the scheduler has moved on. Read the lr
            # this step is about to use, which is what the hand-written loop logs.
            if optimizer is not None:
                self.lr_used = optimizer.param_groups[0]["lr"]

        def on_optimizer_step(self, args, state, control, **kwargs):
            # Under fp16 the Trainer's gradient scaler skips an update when it sees
            # non-finite gradients. The Trainer then drops that batch and still counts
            # the step, while the hand-written loop redoes the same pairs, so exact
            # parity ends at the first skip. Record where that happens.
            if trainer.accelerator.optimizer_step_was_skipped:
                skipped_steps.append(state.global_step + 1)

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "eval_loss" in logs:
                row = {name: logs[f"eval_{key}"] for name, key in TRL_KEYS.items()}
                if use_sft:
                    row["total_loss"], row["loss"] = row["loss"], ""
                eval_log.writerow({"step": state.global_step, **row, "n_pairs": len(eval_pairs)})
                eval_file.flush()
                eval_steps_seen.append(state.global_step)
                if not first_eval:
                    first_eval.update(row)
                print(
                    f"eval  step {state.global_step:5d}  {loss_name} {row[loss_name]:.4f}  "
                    f"margin {row['reward_margin']:+.4f}  acc {row['accuracy']:.3f}  "
                    f"chosen_logp {row['chosen_logp']:.2f}"
                )
            elif "loss" in logs:
                row = {name: logs[key] for name, key in TRL_KEYS.items()}
                if use_sft:
                    row["total_loss"], row["loss"] = row["loss"], ""
                train_log.writerow(
                    {
                        "step": state.global_step,
                        "epoch": (state.global_step - 1) // steps_per_epoch,
                        **row,
                        "lr": self.lr_used,
                        "grad_norm": logs.get("grad_norm", ""),
                        "elapsed_s": round(time.perf_counter() - start, 3),
                    }
                )
                train_file.flush()

    trainer = DPOTrainer(
        model=policy,
        # parity: an explicit, separately loaded reference, as in the hand-written loop.
        # (TRL's own default with a LoRA policy is to switch the adapter off instead.)
        ref_model=reference,
        args=args,
        train_dataset=train_rows,
        eval_dataset=eval_rows,
        processing_class=tokenizer,
        callbacks=[CsvLogger()],
    )
    trainer.remove_callback(PrinterCallback)  # our CsvLogger prints the eval rows instead
    tokens_match = tokenization_matches(trainer.train_dataset, train_pairs) and (
        tokenization_matches(trainer.eval_dataset, eval_pairs)
    )
    if not tokens_match:
        raise RuntimeError(
            "TRL tokenized the selected rows differently from tiny_rlhf.data.tokenize_pair, "
            "so the two runs would not see the same token sequences."
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    trainer.evaluate()  # step 0, before any update
    trainer.train()
    if eval_steps_seen[-1] != trainer.state.global_step:
        trainer.evaluate()  # the last step was off the eval grid
    wall_time = time.perf_counter() - start
    train_file.close()
    eval_file.close()

    # Per-pair rows are not something TRL reports, so score the TRL-trained policy with
    # the hand-written evaluate(). Its means should reproduce TRL's last eval row, which
    # is one more cross-check between the two implementations.
    collate = PreferenceCollator(pad_token_id=tokenizer.pad_token_id, max_length=cfg.max_length)
    eval_batches = [
        collate(eval_pairs[i : i + cfg.eval_batch_size])
        for i in range(0, len(eval_pairs), cfg.eval_batch_size)
    ]
    recomputed, pair_rows = evaluate(
        trainer.model, trainer.ref_model, eval_batches, cfg.beta, device, amp
    )
    with (out_dir / "eval_pairs_final.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        writer.writerows(pair_rows)
    with (out_dir / "eval.csv").open() as f:
        last_trl_eval = list(csv.DictReader(f))[-1]
    # With an sft term TRL's eval loss is the combined one; compare the DPO metrics only.
    compared = [k for k in METRICS if not (use_sft and k == "loss")]

    if (
        checkpoint_dir is not None
        and cfg.save_adapter
        and hasattr(trainer.model, "save_pretrained")
    ):
        trainer.model.save_pretrained(str(checkpoint_dir))

    trainable = [p for p in trainer.model.parameters() if p.requires_grad]
    eval_sizes = {len(b["input_ids"]) // 2 for b in eval_batches}
    run_info = {
        "run_name": cfg.run_name,
        "implementation": "trl",
        "wall_time_s": round(wall_time, 2),
        "optimizer_steps": trainer.state.global_step,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": args.get_warmup_steps(total_steps),
        "first_eval_loss": first_eval["loss"] if not use_sft else None,
        "first_eval_total_loss": first_eval["total_loss"] if use_sft else None,
        "sft_weight": cfg.sft_weight,
        "device": str(device),
        "amp_fp16": amp,
        "param_dtype": str(next(trainer.model.parameters()).dtype),
        # MiB (bytes / 2**20), not MB or GB; the field name is kept for existing results.
        "peak_gpu_memory_mb": (
            round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
            if device.type == "cuda"
            else None
        ),
        "trainable_params": sum(p.numel() for p in trainable),
        "total_params": sum(p.numel() for p in trainer.model.parameters()),
        "n_train_pairs": len(train_pairs),
        "n_eval_pairs": len(eval_pairs),
        "truncation_train": truncation_stats(train_pairs, cfg.max_length),
        "truncation_eval": truncation_stats(eval_pairs, cfg.max_length),
        "seed": cfg.seed,
        "git_sha": git_sha(),
        "versions": library_versions(),
        "tokenization_matches_handwritten": tokens_match,
        "skipped_optimizer_steps": skipped_steps,  # fp16 scaler skips; empty in float32
        "eval_pairs_sha256": pairs_fingerprint(eval_pairs, cfg.max_length),
        "train_order": "random (TRL sampler)" if cfg.shuffle else "sequential",
        # TRL averages per-batch means; ours weights by pairs. They agree only when
        # every held-out batch has the same size.
        "eval_batches_equal_size": len(eval_sizes) == 1,
        "final_eval_by_trl": {k: float(last_trl_eval[k]) for k in compared},
        "final_eval_recomputed_by_handwritten_evaluate": {k: recomputed[k] for k in compared},
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    return run_info


def main(cfg: DPOConfig, results_root: str | Path = "results") -> dict[str, Any]:
    """Load the same tokenizer, weights and pairs as train_dpo.main, then train with TRL."""
    tokenizer, policy, reference = load_models(cfg)
    train_rows, train_pairs = select_rows(
        cfg.dataset_name, "train", cfg.n_train, cfg.seed, tokenizer, cfg.max_length
    )
    eval_rows, eval_pairs = select_rows(
        cfg.dataset_name, "test", cfg.n_eval, cfg.seed, tokenizer, cfg.max_length
    )
    return train_with_trl(
        cfg,
        policy,
        reference,
        tokenizer,
        train_rows,
        eval_rows,
        train_pairs,
        eval_pairs,
        out_dir=Path(results_root) / cfg.run_name,
        checkpoint_dir=Path("checkpoints") / cfg.run_name,
    )
