"""KTO (Kahneman-Tversky Optimization, Ethayarajh et al. 2024) written by hand (M7).

DPO learns from pairs: "for this prompt, answer A is better than answer B". KTO learns
from single answers with a yes/no label: "this answer is good" or "this answer is bad".
This module trains on the Track B data with every pair split into one good (desirable)
and one bad (undesirable) example, so the two methods see exactly the same answers and
differ only in whether the model is told which two answers belong together.

The math mirrors TRL 1.18.0's KTOTrainer (loss_type "kto"), read from
trl/trainer/kto_trainer.py, and a test checks the two agree on a tiny model.

    r(x, y)  = log pi(y|x) - log pi_ref(y|x)        the implied reward, before beta
    z0       = max(0, mean over the micro-batch of [log pi(y'|x) - log pi_ref(y'|x)])
               computed on MISMATCHED (x, y') pairs: each prompt with another
               example's answer. It estimates KL(pi || pi_ref), the reference point.
               No gradient flows through it.

    good answer:  loss = desirable_weight   * (1 - sigmoid(beta * (r - z0)))
    bad answer:   loss = undesirable_weight * (1 - sigmoid(beta * (z0 - r)))

    batch loss = mean of the per-answer losses (good and bad answers together)

Step 0: the policy equals the reference, so r = 0 and z0 = 0 for every answer, every
per-answer loss is 1 - sigmoid(0) = 0.5, and with both weights 1 the loss is exactly 0.5.
(The DPO loss starts at log 2 = 0.693 instead.)

Each per-answer loss lies between 0 and its weight. A good answer whose log-ratio is far
above z0 contributes almost nothing, and so does a bad answer far below it: the loss
saturates on both sides. DPO's -log sigmoid keeps growing without bound for pairs that
are ranked the wrong way and only flattens out for pairs ranked the right way.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from tiny_rlhf.data import PreferenceCollator, TokenizedPair, pairs_fingerprint, truncation_stats
from tiny_rlhf.losses import sequence_logprob
from tiny_rlhf.train_dpo import (
    PAIR_COLUMNS,
    accumulate_with_retries,
    clear_stale_outputs,
    git_sha,
    library_versions,
    lr_factor,
    pair_logps,
    resolve_device,
    to_device,
    use_amp,
)

TRAIN_COLUMNS = (
    "step",
    "epoch",
    "loss",  # the KTO loss, starts at 0.5 with both weights 1
    "kl",  # z0, the reference point
    "chosen_reward",  # mean of beta * r over the good answers of the step
    "rejected_reward",  # mean of beta * r over the bad answers of the step
    "chosen_logp",
    "rejected_logp",
    "n_desirable",
    "n_undesirable",
    "lr",
    "grad_norm",
    "elapsed_s",
)
# The first nine after "step" match the DPO loop's eval.csv (tiny_rlhf.train_dpo), so the
# two kinds of run can be plotted on the same axes. "loss" is the KTO loss here.
EVAL_COLUMNS = (
    "step",
    "loss",
    "reward_margin",
    "accuracy",
    "chosen_logp",
    "rejected_logp",
    "chosen_reward",
    "rejected_reward",
    "n_pairs",
    "kl",
    "dpo_loss",  # the DPO loss of the same pairs at the same beta, for comparison only
)


@dataclass
class KTOConfig:
    """Everything that defines a KTO run. Same fields and meanings as DPOConfig, except:

    batch_size        ANSWERS per micro-batch (DPO counts pairs). Must be at least 2:
                      the reference point z0 is estimated from the other answers of the
                      micro-batch, and with one answer the "mismatched" answer is the
                      answer itself.
    desirable_weight, undesirable_weight
                      multiply the good-answer and bad-answer losses (TRL's names). With
                      every pair split in two the data is balanced, so both stay 1.
    n_train, n_eval   PAIRS read from the file; each becomes two examples.
    """

    run_name: str = "kto"
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    dataset_name: str = "data/trackb"
    n_train: int = 200
    n_eval: int = 48
    max_length: int | None = 320
    beta: float = 0.1
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    epochs: int = 1
    batch_size: int = 4
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    eval_every: int = 10
    eval_batch_size: int = 8  # pairs per held-out batch
    seed: int = 0
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    amp: bool | str = "auto"
    device: str = "auto"
    max_steps: int | None = None
    save_adapter: bool = True


def load_kto_config(path: str | Path, overrides: dict[str, Any] | None = None) -> KTOConfig:
    """Read a YAML config and apply `key: value` overrides. Unknown keys are an error."""
    values = yaml.safe_load(Path(path).read_text()) or {}
    values.update(overrides or {})
    known = {f.name for f in dataclasses.fields(KTOConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    cfg = KTOConfig(**values)
    if cfg.amp not in ("auto", True, False):
        raise ValueError(f"amp must be auto, true or false, got {cfg.amp!r}")
    if cfg.batch_size < 2:
        raise ValueError(
            "batch_size must be at least 2: the KL reference point is estimated from the "
            "other answers in the micro-batch (TRL refuses a batch size of 1 for the same reason)"
        )
    if cfg.eval_every < 1 or cfg.grad_accum < 1:
        raise ValueError("eval_every and grad_accum must be at least 1")
    if cfg.beta <= 0 or cfg.desirable_weight <= 0 or cfg.undesirable_weight <= 0:
        raise ValueError("beta and both weights must be positive")
    return cfg


# --------------------------------------------------------------------------------------
# The loss
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KTOLossOutput:
    """KTO loss plus what is worth logging. Everything but `loss` is detached.

    loss             scalar, has gradients
    losses           per-answer losses, good answers first, then bad ones (as TRL orders them)
    kl               z0, the clamped reference point (a scalar)
    chosen_rewards   beta * r for each good answer
    rejected_rewards beta * r for each bad answer
    """

    loss: Tensor
    losses: Tensor
    kl: Tensor
    chosen_rewards: Tensor
    rejected_rewards: Tensor


def kto_loss(
    policy_logps: Tensor,
    ref_logps: Tensor,
    labels: Tensor,
    kl_policy_logps: Tensor,
    kl_ref_logps: Tensor,
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> KTOLossOutput:
    """The KTO loss for one micro-batch, exactly as TRL 1.18.0 computes it.

    Args:
        policy_logps, ref_logps: [n] summed log-probability of each answer given its own
            prompt, under the policy and under the frozen reference.
        labels: [n] bool, True for a good (desirable) answer.
        kl_policy_logps, kl_ref_logps: [n] the same for the mismatched (prompt, answer)
            sequences of the micro-batch. Used only for the reference point z0.
        beta: how strongly the policy is held near the reference.
        desirable_weight, undesirable_weight: multiply the two kinds of per-answer loss.

    Derivation in plain text math. KTO borrows prospect theory's value function: a
    person judges an outcome by its gain or loss relative to a reference point, not by
    its absolute value, and the curve flattens far from the reference point. Here the
    "outcome" is the implied reward r = log pi(y|x) - log pi_ref(y|x), and the reference
    point is z0, an estimate of KL(pi || pi_ref): how far the policy has moved on
    average. The value of a good answer is sigmoid(beta * (r - z0)): it is high when the
    policy raised that answer more than it moved on average. The value of a bad answer
    is sigmoid(beta * (z0 - r)): high when the policy lowered it relative to the average
    move. The loss is the weight minus the value, so minimising it maximises the value.

    z0 is detached and clamped at 0. Detached: it only sets the bar, the gradient does
    not try to change it. Clamped: an estimated KL below 0 is impossible for a true KL,
    so negative estimates (noise) are treated as 0. It is computed from mismatched pairs
    because on a matched good answer r is exactly what is being raised; using another
    prompt's answer gives an estimate of how much the policy moved in general.

    Note on the estimate: KL(pi || pi_ref) is the expectation of log pi - log pi_ref
    over answers SAMPLED FROM pi. Here the answers come from the dataset, not from pi, so
    z0 is a cheap proxy, as in the paper and in TRL.
    """
    if not (
        policy_logps.shape == ref_logps.shape == labels.shape
        and kl_policy_logps.shape == kl_ref_logps.shape
    ):
        raise ValueError("log-probability and label tensors must share a shape")
    labels = labels.bool()
    kl = (kl_policy_logps - kl_ref_logps).mean().detach().clamp(min=0)
    logratios = policy_logps - ref_logps.detach()
    chosen = logratios[labels]
    rejected = logratios[~labels]
    chosen_losses = 1 - torch.sigmoid(beta * (chosen - kl))
    rejected_losses = 1 - torch.sigmoid(beta * (kl - rejected))
    losses = torch.cat((desirable_weight * chosen_losses, undesirable_weight * rejected_losses))
    return KTOLossOutput(
        loss=losses.mean(),
        losses=losses.detach(),
        kl=kl,
        chosen_rewards=beta * chosen.detach(),
        rejected_rewards=beta * rejected.detach(),
    )


def step0_loss(
    desirable_weight: float, undesirable_weight: float, n_good: int, n_bad: int
) -> float:
    """The KTO loss when the policy equals the reference: every per-answer loss is 0.5."""
    total = desirable_weight * n_good + undesirable_weight * n_bad
    return 0.5 * total / max(1, n_good + n_bad)


# --------------------------------------------------------------------------------------
# From pairs to single answers, and the mismatched answers for z0
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class UnpairedExample:
    """One answer with a good/bad label. pair_index points back to the pair it came from."""

    prompt_ids: list[int]
    completion_ids: list[int]
    label: bool
    pair_index: int
    kl_completion_ids: list[int] | None = None  # another example's answer, for z0


def unpair(
    pairs: list[TokenizedPair], seed: int, batch_size: int
) -> tuple[list[UnpairedExample], dict[str, int]]:
    """Split every pair into a good and a bad example, in a seeded order.

    TRL feeds KTO examples in a FIXED order (it refuses a random sampler), because the
    mismatched answer used for z0 is precomputed from the neighbours in each micro-batch.
    So the order is decided once, here: a seeded shuffle, then a pass that keeps the two
    halves of a pair out of the same micro-batch. If they shared one, a prompt's
    "mismatched" answer could be its own pair-mate, an answer to the same prompt.

    Returns (examples, counts). counts records how many pairs went in and how many
    micro-batches still hold both halves of a pair (only possible in the last batches,
    when nothing else is left to swap in).
    """
    examples = [UnpairedExample(p.prompt_ids, p.chosen_ids, True, i) for i, p in enumerate(pairs)]
    examples += [
        UnpairedExample(p.prompt_ids, p.rejected_ids, False, i) for i, p in enumerate(pairs)
    ]
    rng = random.Random(seed)
    rng.shuffle(examples)

    ordered: list[UnpairedExample] = []
    remaining = examples
    conflicts = 0
    while remaining:
        chunk: list[UnpairedExample] = []
        seen: set[int] = set()
        rest: list[UnpairedExample] = []
        for ex in remaining:
            if len(chunk) < batch_size and ex.pair_index not in seen:
                chunk.append(ex)
                seen.add(ex.pair_index)
            else:
                rest.append(ex)
        # Nothing left that does not clash with this chunk: fill it anyway (end of data).
        while len(chunk) < batch_size and rest:
            conflicts += 1 if rest[0].pair_index in seen else 0
            chunk.append(rest.pop(0))
        ordered += chunk
        remaining = rest
    counts = {
        "n_pairs": len(pairs),
        "n_examples": len(ordered),
        "n_desirable": sum(e.label for e in ordered),
        "n_undesirable": sum(not e.label for e in ordered),
        "micro_batches_holding_both_halves_of_a_pair": conflicts,
    }
    return ordered, counts


def add_kl_completions(examples: list[UnpairedExample], batch_size: int) -> list[UnpairedExample]:
    """Give each example the answer of its predecessor in the same micro-batch.

    Exactly TRL's rule: within each consecutive block of batch_size examples,
    (x_1, y_1), ..., (x_n, y_n) gets the mismatched answers y_n, y_1, ..., y_{n-1}.
    The mismatched answers of a micro-batch are therefore the same set as its real
    answers, paired with different prompts. A final block of one example gets its own
    answer back (TRL behaves the same way).
    """
    out: list[UnpairedExample] = []
    for start in range(0, len(examples), batch_size):
        block = examples[start : start + batch_size]
        rotated = [block[-1].completion_ids] + [e.completion_ids for e in block[:-1]]
        out += [
            dataclasses.replace(e, kl_completion_ids=k) for e, k in zip(block, rotated, strict=True)
        ]
    return out


def order_fingerprint(examples: list[UnpairedExample]) -> str:
    """sha256 of the example order and labels, so a run's data order can be checked."""
    payload = json.dumps([[e.pair_index, e.label] for e in examples], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class UnpairedCollator:
    """Pad unpaired examples into a batch, mirroring TRL's DataCollatorForUnpairedPreference.

    Returns input_ids, attention_mask, response_mask for the real (prompt, answer)
    sequences and kl_input_ids, kl_attention_mask, kl_response_mask for the mismatched
    ones, plus labels. Right padding, and each sequence is cut to its first max_length
    tokens (the prompt is kept, a long answer loses its tail).
    """

    def __init__(self, pad_token_id: int, max_length: int | None = None) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def _pad(self, prompts: list[list[int]], answers: list[list[int]]) -> tuple[Tensor, ...]:
        rows, masks = [], []
        for prompt, answer in zip(prompts, answers, strict=True):
            rows.append((prompt + answer)[: self.max_length])
            masks.append(([False] * len(prompt) + [True] * len(answer))[: self.max_length])
        width = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(rows), width), dtype=torch.long)
        response = torch.zeros((len(rows), width), dtype=torch.bool)
        for i, (ids, mask) in enumerate(zip(rows, masks, strict=True)):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[i, : len(ids)] = 1
            response[i, : len(ids)] = torch.tensor(mask, dtype=torch.bool)
        return input_ids, attention, response

    def __call__(self, examples: list[UnpairedExample]) -> dict[str, Tensor]:
        prompts = [e.prompt_ids for e in examples]
        input_ids, attention, response = self._pad(prompts, [e.completion_ids for e in examples])
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention,
            "response_mask": response,
            "labels": torch.tensor([e.label for e in examples], dtype=torch.bool),
        }
        if all(e.kl_completion_ids is not None for e in examples):
            kl_ids, kl_attention, kl_response = self._pad(
                prompts, [e.kl_completion_ids for e in examples]
            )
            batch.update(
                kl_input_ids=kl_ids, kl_attention_mask=kl_attention, kl_response_mask=kl_response
            )
        return batch


def answer_logps(model: nn.Module, batch: dict[str, Tensor], prefix: str = "") -> Tensor:
    """Summed log-probability of each row's answer tokens. prefix "kl_" scores the
    mismatched sequences instead."""
    ids = batch[f"{prefix}input_ids"]
    logits = model(input_ids=ids, attention_mask=batch[f"{prefix}attention_mask"]).logits
    return sequence_logprob(logits, ids, batch[f"{prefix}response_mask"])


# --------------------------------------------------------------------------------------
# Evaluation on held-out pairs
# --------------------------------------------------------------------------------------


@torch.no_grad()
def reference_cache(
    reference: nn.Module,
    pair_batches: list[dict[str, Tensor]],
    kl_batches: list[dict[str, Tensor]],
    device: torch.device,
    amp: bool,
) -> tuple[list[tuple[Tensor, Tensor]], list[Tensor]]:
    """The frozen reference's log-probabilities on the held-out pairs and mismatched answers."""
    pairs_out, kl_out = [], []
    for b in pair_batches:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            c, r = pair_logps(reference, to_device(b, device))
        pairs_out.append((c.cpu(), r.cpu()))
    for b in kl_batches:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            kl_out.append(answer_logps(reference, to_device(b, device), "kl_").cpu())
    return pairs_out, kl_out


@torch.no_grad()
def evaluate(
    policy: nn.Module,
    pair_batches: list[dict[str, Tensor]],
    kl_batches: list[dict[str, Tensor]],
    ref_cache: tuple[list[tuple[Tensor, Tensor]], list[Tensor]],
    cfg: KTOConfig,
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Score held-out pairs both ways. Returns (metrics, one row per pair).

    The pairs are scored as pairs (accuracy, reward margin, DPO loss: directly
    comparable with a DPO run's eval.csv) and as single answers (the KTO loss, with one
    reference point z0 estimated from every held-out mismatched answer at once).
    """
    was_training = policy.training
    policy.eval()
    ref_pairs, ref_kl = ref_cache
    pol_c, pol_r, ref_c, ref_r = [], [], [], []
    for b, (rc, rr) in zip(pair_batches, ref_pairs, strict=True):
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            c, r = pair_logps(policy, to_device(b, device))
        pol_c.append(c.float().cpu())
        pol_r.append(r.float().cpu())
        ref_c.append(rc)
        ref_r.append(rr)
    kl_pol = []
    for b in kl_batches:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            kl_pol.append(answer_logps(policy, to_device(b, device), "kl_").float().cpu())
    policy.train(was_training)

    pol_c, pol_r, ref_c, ref_r = (torch.cat(t).double() for t in (pol_c, pol_r, ref_c, ref_r))
    kl_pol_all, kl_ref_all = torch.cat(kl_pol).double(), torch.cat(ref_kl).double()
    n = len(pol_c)
    out = kto_loss(
        torch.cat((pol_c, pol_r)),
        torch.cat((ref_c, ref_r)),
        torch.cat((torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool))),
        kl_pol_all,
        kl_ref_all,
        beta=cfg.beta,
        desirable_weight=cfg.desirable_weight,
        undesirable_weight=cfg.undesirable_weight,
    )
    chosen_rewards = cfg.beta * (pol_c - ref_c)
    rejected_rewards = cfg.beta * (pol_r - ref_r)
    margin = chosen_rewards - rejected_rewards
    metrics = {
        "loss": out.loss.item(),
        "reward_margin": margin.mean().item(),
        "accuracy": (margin > 0).double().mean().item(),
        "chosen_logp": pol_c.mean().item(),
        "rejected_logp": pol_r.mean().item(),
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
        "n_pairs": n,
        "kl": out.kl.item(),
        "dpo_loss": -F.logsigmoid(margin).mean().item(),
    }
    rows = [
        {
            "pair_index": i,
            "chosen_logp": pol_c[i].item(),
            "rejected_logp": pol_r[i].item(),
            "ref_chosen_logp": ref_c[i].item(),
            "ref_rejected_logp": ref_r[i].item(),
            "chosen_reward": chosen_rewards[i].item(),
            "rejected_reward": rejected_rewards[i].item(),
            "reward_margin": margin[i].item(),
        }
        for i in range(n)
    ]
    return metrics, rows


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def plan_steps(n_examples: int, cfg: KTOConfig) -> tuple[int, int]:
    """(optimizer steps per epoch, total optimizer steps after the max_steps cap)."""
    micro_batches = math.ceil(n_examples / cfg.batch_size)
    per_epoch = math.ceil(micro_batches / cfg.grad_accum)
    total = per_epoch * cfg.epochs
    if cfg.max_steps is not None:
        total = min(total, cfg.max_steps)
    return per_epoch, total


def epoch_steps(n_examples: int, cfg: KTOConfig) -> list[list[list[int]]]:
    """steps[s][m] = example indices of micro-batch m of optimizer step s, in fixed order.

    No reshuffling between epochs: the mismatched answers were built for these exact
    micro-batches (as in TRL, which uses a sequential sampler for KTO).
    """
    micro = [
        list(range(i, min(i + cfg.batch_size, n_examples)))
        for i in range(0, n_examples, cfg.batch_size)
    ]
    return [micro[i : i + cfg.grad_accum] for i in range(0, len(micro), cfg.grad_accum)]


def train(
    cfg: KTOConfig,
    policy: nn.Module,
    reference: nn.Module,
    train_pairs: list[TokenizedPair],
    eval_pairs: list[TokenizedPair],
    pad_token_id: int,
    out_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    """Train `policy` with KTO against the frozen `reference`. Writes logs to out_dir.

    Same outputs as tiny_rlhf.train_dpo.train (config.yaml, train.csv, eval.csv,
    eval_pairs_final.csv, run_info.json), with KTO's columns. Returns run_info.
    """
    if not train_pairs or not eval_pairs:
        raise ValueError("train_pairs and eval_pairs must both be non-empty")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_stale_outputs(out_dir)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))

    device = resolve_device(cfg.device)
    amp = use_amp(cfg, device)  # type: ignore[arg-type]  # same fields as DPOConfig
    torch.manual_seed(cfg.seed)  # LoRA dropout draws from the global generator

    examples, counts = unpair(train_pairs, cfg.seed, cfg.batch_size)
    examples = add_kl_completions(examples, cfg.batch_size)
    eval_examples, _ = unpair(eval_pairs, cfg.seed, cfg.batch_size)
    eval_examples = add_kl_completions(eval_examples, cfg.batch_size)

    policy.to(device).train()
    reference.to(device).eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    collate = UnpairedCollator(pad_token_id=pad_token_id, max_length=cfg.max_length)
    pair_collate = PreferenceCollator(pad_token_id=pad_token_id, max_length=cfg.max_length)
    eval_pair_batches = [
        pair_collate(eval_pairs[i : i + cfg.eval_batch_size])
        for i in range(0, len(eval_pairs), cfg.eval_batch_size)
    ]
    eval_kl_batches = [
        collate(eval_examples[i : i + cfg.batch_size])
        for i in range(0, len(eval_examples), cfg.batch_size)
    ]

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch, total_steps = plan_steps(len(examples), cfg)
    warmup_steps = math.ceil(cfg.warmup_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, total_steps, warmup_steps)
    )
    if scaler is None:
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
    ref_cache = reference_cache(reference, eval_pair_batches, eval_kl_batches, device, amp)

    def run_eval(step: int) -> tuple[dict[str, float], list[dict[str, float]]]:
        metrics, rows = evaluate(
            policy, eval_pair_batches, eval_kl_batches, ref_cache, cfg, device, amp
        )
        eval_log.writerow({"step": step, **metrics})
        eval_file.flush()
        print(
            f"eval  step {step:5d}  kto_loss {metrics['loss']:.4f}  kl {metrics['kl']:.4f}  "
            f"margin {metrics['reward_margin']:+.4f}  acc {metrics['accuracy']:.3f}  "
            f"chosen_logp {metrics['chosen_logp']:.2f}"
        )
        return metrics, rows

    # The KTO version of the plan's step-0 check: policy equals reference, so every
    # implied reward and the reference point are 0 and every per-answer loss is 0.5.
    first_metrics, pair_rows = run_eval(0)
    expected = step0_loss(
        cfg.desirable_weight, cfg.undesirable_weight, len(eval_pairs), len(eval_pairs)
    )
    worst_margin = max(abs(row["reward_margin"]) for row in pair_rows)
    if not (abs(first_metrics["loss"] - expected) <= 1e-3 and worst_margin <= 1e-2):
        raise RuntimeError(
            f"step-0 KTO loss is {first_metrics['loss']:.6f} (expected {expected:.6f}) and the "
            f"largest step-0 reward margin is {worst_margin:.6f} (expected 0). The policy and "
            "the reference disagree before training."
        )

    step = 0
    last_eval_step = 0
    scaler_retries = 0
    for epoch in range(cfg.epochs):
        for micro_batches in epoch_steps(len(examples), cfg):
            if step >= total_steps:
                break
            n_step = sum(len(m) for m in micro_batches)

            def accumulate(
                micro_batches: list[list[int]] = micro_batches, n_step: int = n_step
            ) -> dict[str, float]:
                sums = {
                    "loss": 0.0,
                    "kl": 0.0,
                    "cr": 0.0,
                    "rr": 0.0,
                    "cl": 0.0,
                    "rl": 0.0,
                    "nd": 0,
                    "nu": 0,
                }
                for indices in micro_batches:
                    batch = to_device(collate([examples[i] for i in indices]), device)
                    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                        # Reference first (no dropout, no randomness), then the policy on
                        # the mismatched answers without gradient, then the policy on the
                        # real answers. The policy's two passes happen in TRL's order, so a
                        # LoRA dropout layer draws its random numbers in the same order.
                        with torch.no_grad():
                            ref = answer_logps(reference, batch)
                            ref_kl = answer_logps(reference, batch, "kl_")
                            pol_kl = answer_logps(policy, batch, "kl_")
                        pol = answer_logps(policy, batch)
                    out = kto_loss(
                        pol,
                        ref,
                        batch["labels"],
                        pol_kl,
                        ref_kl,
                        beta=cfg.beta,
                        desirable_weight=cfg.desirable_weight,
                        undesirable_weight=cfg.undesirable_weight,
                    )
                    # Weight each micro-batch by its share of the step's answers: the mean
                    # loss over the step, as in the DPO loop.
                    share = len(indices) / n_step
                    scaler.scale(out.loss * share).backward()
                    labels = batch["labels"]
                    sums["loss"] += out.loss.item() * share
                    sums["kl"] += out.kl.item() * share
                    sums["cr"] += out.chosen_rewards.sum().item()
                    sums["rr"] += out.rejected_rewards.sum().item()
                    sums["cl"] += pol.detach()[labels].sum().item()
                    sums["rl"] += pol.detach()[~labels].sum().item()
                    sums["nd"] += int(labels.sum())
                    sums["nu"] += int((~labels).sum())
                return sums

            sums, grad_norm, retries = accumulate_with_retries(
                accumulate, scaler, optimizer, trainable, cfg.max_grad_norm, step + 1
            )
            scaler_retries += retries
            lr_used = scheduler.get_last_lr()[0]
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            nd, nu = max(sums["nd"], 1), max(sums["nu"], 1)
            train_log.writerow(
                {
                    "step": step,
                    "epoch": epoch,
                    "loss": sums["loss"],
                    "kl": sums["kl"],
                    "chosen_reward": sums["cr"] / nd if sums["nd"] else "",
                    "rejected_reward": sums["rr"] / nu if sums["nu"] else "",
                    "chosen_logp": sums["cl"] / nd if sums["nd"] else "",
                    "rejected_logp": sums["rl"] / nu if sums["nu"] else "",
                    "n_desirable": sums["nd"],
                    "n_undesirable": sums["nu"],
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
        "implementation": "handwritten_kto",
        "wall_time_s": round(wall_time, 2),
        "optimizer_steps": step,
        "scaler_retries": scaler_retries,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup_steps,
        "first_eval_loss": first_metrics["loss"],
        "expected_first_eval_loss": expected,
        "beta": cfg.beta,
        "desirable_weight": cfg.desirable_weight,
        "undesirable_weight": cfg.undesirable_weight,
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
        "unpaired": counts,
        "unpaired_order_sha256": order_fingerprint(examples),
        "truncation_train": truncation_stats(train_pairs, cfg.max_length),
        "truncation_eval": truncation_stats(eval_pairs, cfg.max_length),
        "eval_pairs_sha256": pairs_fingerprint(eval_pairs, cfg.max_length),
        "seed": cfg.seed,
        "git_sha": git_sha(),
        "versions": library_versions(),
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    return run_info


def main(cfg: KTOConfig, results_root: str | Path = "results") -> dict[str, Any]:
    """Load the models and the Track B pairs (same selection as the DPO runs), then train."""
    from tiny_rlhf.trackb.train import select_pairs
    from tiny_rlhf.train_dpo import load_models

    data_dir = Path(cfg.dataset_name)
    if not (data_dir / "train.jsonl").exists():
        raise FileNotFoundError(f"{data_dir}/train.jsonl not found (dataset_name is data/trackb)")
    tokenizer, policy, reference = load_models(cfg)  # type: ignore[arg-type]
    train_pairs = select_pairs(
        data_dir / "train.jsonl", tokenizer, cfg.max_length, cfg.n_train, cfg.seed
    )
    eval_pairs = select_pairs(
        data_dir / "val.jsonl", tokenizer, cfg.max_length, cfg.n_eval, cfg.seed
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
