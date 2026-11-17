"""REINFORCE with a baseline, by hand, on a toy task small enough to solve exactly.

This is the "actual RL part" of the plan's stretch item, at toy scale. It exists to make
three things from docs/PLAN.md section 2 visible as numbers instead of algebra:

1. The KL-constrained RL problem
       maximise  J(pi) = E_{y ~ pi}[ r(y) ] - beta * KL(pi || pi_ref)
   has the closed-form solution  pi*(y) = pi_ref(y) * exp(r(y) / beta) / Z.
   Here every sequence can be enumerated, so pi* is computed exactly and the learned
   policy is compared with it.
2. REINFORCE reaches that solution by sampling from the current policy (on-policy),
   and a baseline lowers the variance of its gradient estimate without biasing it.
3. DPO reaches the same solution from a fixed set of preference pairs sampled from the
   reference (offline), with no reward values and no sampling from the policy, provided
   the labels follow the Bradley-Terry model.

The toy: sequences of `seq_len` tokens from a vocabulary of `vocab_size`. The last
vocabulary id is a "stop" token. Reward (see `reward`):
    +1 for every adjacent pair (a, b) of non-stop tokens with b == (a + 1) mod (V - 1)
    +stop_bonus if the final token is the stop token
so a perfect sequence counts upward cyclically (0 1 2 0 1 ...) and then stops. The
reward depends on pairs of neighbouring tokens, so the policy has to be autoregressive.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from tiny_rlhf.losses import dpo_loss

BASELINES = ("none", "running_mean", "leave_one_out")
LABEL_MODES = ("bradley_terry", "hard")


@dataclass
class ToyConfig:
    """Everything that defines the toy experiment. Loaded from configs/reinforce_toy.yaml."""

    run_name: str = "reinforce_toy"
    vocab_size: int = 4  # the last id is the stop token
    seq_len: int = 6
    stop_bonus: float = 2.0
    beta: float = 0.5
    embed_dim: int = 16
    hidden_dim: int = 32
    init_seed: int = 0  # initial weights; the frozen reference is this initial policy
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # sampling seeds
    batch_size: int = 32  # sequences per REINFORCE step (DPO uses batch_size / 2 pairs)
    steps: int = 400
    lr: float = 0.01
    ema_decay: float = 0.9  # for the running-mean baseline
    eval_every: int = 10
    grad_variance_batches: int = 200
    grad_variance_steps: list[int] = field(default_factory=lambda: [0, 100, 400])
    # One longer DPO run per label mode (first seed only), to see where DPO goes with
    # more pairs than the equal-budget comparison allows. 0 turns it off.
    dpo_extended_steps: int = 4000


def load_toy_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ToyConfig:
    """Read a YAML config and apply overrides. Unknown keys are an error."""
    values = yaml.safe_load(Path(path).read_text()) or {}
    values.update(overrides or {})
    known = {f.name for f in dataclasses.fields(ToyConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    cfg = ToyConfig(**values)
    if cfg.vocab_size < 3 or cfg.seq_len < 1 or cfg.beta <= 0 or cfg.batch_size % 2:
        raise ValueError(
            "need vocab_size >= 3, seq_len >= 1, beta > 0 and an even batch_size "
            "(DPO turns a batch of sequences into batch_size / 2 pairs)"
        )
    return cfg


# --------------------------------------------------------------------------------------
# The task: reward function and exact enumeration
# --------------------------------------------------------------------------------------


def reward(seqs: Tensor, vocab_size: int, stop_bonus: float = 2.0) -> Tensor:
    """Reward of each sequence. seqs: [n, seq_len] long. Returns [n] float64.

    +1 per adjacent pair (a, b) where neither is the stop token and b follows a in the
    cycle 0 -> 1 -> ... -> V-2 -> 0, plus `stop_bonus` if the last token is the stop token.
    """
    stop = vocab_size - 1
    prev, nxt = seqs[:, :-1], seqs[:, 1:]
    counts_up = (prev != stop) & (nxt != stop) & (nxt == (prev + 1) % stop)
    ends_with_stop = seqs[:, -1] == stop
    return counts_up.sum(dim=1).double() + stop_bonus * ends_with_stop.double()


def enumerate_sequences(vocab_size: int, seq_len: int) -> Tensor:
    """Every possible sequence, shape [vocab_size ** seq_len, seq_len]."""
    return torch.tensor(list(itertools.product(range(vocab_size), repeat=seq_len)))


def optimal_log_probs(ref_logp: Tensor, rewards: Tensor, beta: float) -> Tensor:
    """log pi*(y) for every sequence, where pi*(y) = pi_ref(y) * exp(r(y) / beta) / Z.

    Args:
        ref_logp: [n] log pi_ref(y) over ALL sequences (so it sums to 1 in probability).
        rewards: [n] r(y) for the same sequences.
        beta: the KL penalty strength.
    """
    return torch.log_softmax(ref_logp + rewards / beta, dim=0)


# --------------------------------------------------------------------------------------
# The policy
# --------------------------------------------------------------------------------------


class TinyPolicy(nn.Module):
    """A small GRU language model over fixed-length sequences. No prompt: x is empty."""

    def __init__(self, vocab_size: int, seq_len: int, embed_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.vocab_size, self.seq_len = vocab_size, seq_len
        self.embed = nn.Embedding(vocab_size + 1, embed_dim)  # extra id = start-of-sequence
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def log_prob(self, seqs: Tensor) -> Tensor:
        """log pi(y) for each row of seqs [n, seq_len], summed over its tokens. Shape [n]."""
        start = torch.full_like(seqs[:, :1], self.vocab_size)
        inputs = torch.cat([start, seqs[:, :-1]], dim=1)  # the token at t is predicted from < t
        hidden, _ = self.gru(self.embed(inputs))
        token_logp = F.log_softmax(self.head(hidden).double(), dim=-1)
        return token_logp.gather(-1, seqs.unsqueeze(-1)).squeeze(-1).sum(dim=1)

    @torch.no_grad()
    def sample(self, n: int, generator: torch.Generator) -> Tensor:
        """Draw n sequences from the policy, one token at a time. Shape [n, seq_len]."""
        token = torch.full((n, 1), self.vocab_size, dtype=torch.long)
        state = None
        out = []
        for _ in range(self.seq_len):
            hidden, state = self.gru(self.embed(token), state)
            probs = F.softmax(self.head(hidden[:, -1]).double(), dim=-1)
            token = torch.multinomial(probs, 1, generator=generator)
            out.append(token)
        return torch.cat(out, dim=1)


@dataclass
class Task:
    """The fixed pieces of one experiment: all sequences, their rewards, the frozen
    reference, and the exact optimum computed from them."""

    cfg: ToyConfig
    all_seqs: Tensor  # [n_all, seq_len]
    rewards: Tensor  # [n_all]
    reference: TinyPolicy
    ref_logp: Tensor  # [n_all] log pi_ref
    opt_logp: Tensor  # [n_all] log pi*

    @property
    def optimal_expected_reward(self) -> float:
        return (self.opt_logp.exp() * self.rewards).sum().item()

    @property
    def optimal_objective(self) -> float:
        """J(pi*) = beta * log Z, with Z = sum_y pi_ref(y) exp(r(y) / beta)."""
        return (
            self.cfg.beta * torch.logsumexp(self.ref_logp + self.rewards / self.cfg.beta, 0).item()
        )


def make_policy(cfg: ToyConfig) -> TinyPolicy:
    torch.manual_seed(cfg.init_seed)
    return TinyPolicy(cfg.vocab_size, cfg.seq_len, cfg.embed_dim, cfg.hidden_dim)


def make_task(cfg: ToyConfig) -> Task:
    """Build the reference (the initial policy, frozen) and solve the task exactly."""
    reference = make_policy(cfg).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    all_seqs = enumerate_sequences(cfg.vocab_size, cfg.seq_len)
    rewards = reward(all_seqs, cfg.vocab_size, cfg.stop_bonus)
    with torch.no_grad():
        ref_logp = reference.log_prob(all_seqs)
    return Task(
        cfg, all_seqs, rewards, reference, ref_logp, optimal_log_probs(ref_logp, rewards, cfg.beta)
    )


@torch.no_grad()
def exact_metrics(policy: TinyPolicy, task: Task) -> dict[str, float]:
    """Exact quantities for a policy, by summing over every sequence (no sampling noise)."""
    logp = policy.log_prob(task.all_seqs)
    p = logp.exp()
    expected_reward = (p * task.rewards).sum().item()
    kl_from_ref = (p * (logp - task.ref_logp)).sum().item()
    return {
        "expected_reward": expected_reward,
        "kl_from_ref": kl_from_ref,
        # The objective being maximised. It equals optimal_objective - beta * kl_to_opt.
        "objective": expected_reward - task.cfg.beta * kl_from_ref,
        "kl_to_opt": (p * (logp - task.opt_logp)).sum().item(),  # KL(pi || pi*)
        "tv_to_opt": 0.5 * (p - task.opt_logp.exp()).abs().sum().item(),
    }


# --------------------------------------------------------------------------------------
# REINFORCE
# --------------------------------------------------------------------------------------


def kl_penalised_return(rewards: Tensor, logp: Tensor, ref_logp: Tensor, beta: float) -> Tensor:
    """Per-sample return R(y) = r(y) - beta * (log pi(y) - log pi_ref(y)), detached.

    Why this is the right "reward" for the KL-penalised objective:
        J = E_{y~pi}[ r(y) - beta * (log pi(y) - log pi_ref(y)) ] = E_{y~pi}[ R(y) ]
    R depends on the parameters through log pi(y), so differentiating gives two terms:
        grad J = E[ R(y) * grad log pi(y) ]  -  beta * E[ grad log pi(y) ]
    and the second is zero, because E[grad log pi] = sum_y grad pi(y) = grad 1 = 0.
    So R can be treated as a constant (detached) and the usual estimator still holds.
    """
    return (rewards - beta * (logp - ref_logp)).detach()


def reinforce_loss(logp: Tensor, returns: Tensor, baseline: Tensor | float = 0.0) -> Tensor:
    """Surrogate whose gradient is the REINFORCE estimate of -grad J.

    loss = -mean( (R - b) * log pi(y) ),  y sampled from pi, R and b held constant.
    Any b that does not depend on the sample it is applied to leaves the expected
    gradient unchanged (E[b * grad log pi] = b * 0) and can lower its variance.
    """
    return -((returns.detach() - baseline) * logp).mean()


def leave_one_out_baseline(returns: Tensor) -> Tensor:
    """For each sample, the mean return of the OTHER samples in the batch.

    Using the plain batch mean would include the sample's own return in its baseline,
    which makes the estimator slightly biased. Leaving it out keeps it unbiased.
    """
    n = returns.numel()
    return (returns.sum() - returns) / (n - 1)


class RunningMean:
    """Exponential moving average of past batch-mean returns. Starts at 0."""

    def __init__(self, decay: float) -> None:
        self.decay, self.value, self.seen = decay, 0.0, False

    def update(self, returns: Tensor) -> None:
        mean = returns.mean().item()
        self.value = mean if not self.seen else self.decay * self.value + (1 - self.decay) * mean
        self.seen = True


def reinforce_batch_loss(
    policy: TinyPolicy,
    task: Task,
    baseline: str,
    generator: torch.Generator,
    running: RunningMean | None = None,
) -> Tensor:
    """Sample one batch from the current policy and return its REINFORCE surrogate loss."""
    cfg = task.cfg
    seqs = policy.sample(cfg.batch_size, generator)  # on-policy: fresh samples every step
    logp = policy.log_prob(seqs)
    with torch.no_grad():
        ref_logp = task.reference.log_prob(seqs)
    returns = kl_penalised_return(
        reward(seqs, cfg.vocab_size, cfg.stop_bonus), logp, ref_logp, cfg.beta
    )
    if baseline == "none":
        b: Tensor | float = 0.0
    elif baseline == "running_mean":
        assert running is not None
        b = running.value  # built from earlier batches only, so independent of this one
        running.update(returns)
    elif baseline == "leave_one_out":
        b = leave_one_out_baseline(returns)
    else:
        raise ValueError(f"unknown baseline {baseline!r}, expected one of {BASELINES}")
    return reinforce_loss(logp, returns, b)


def train_reinforce(
    task: Task, baseline: str, seed: int, snapshot_steps: tuple[int, ...] = ()
) -> tuple[list[dict[str, float]], dict[int, dict[str, Tensor]]]:
    """Train a copy of the reference with REINFORCE. Returns (history, snapshots).

    history has one row per evaluation: step, sequences_seen and the exact metrics.
    snapshots maps a step number to the policy's state_dict at that step.
    """
    cfg = task.cfg
    policy = make_policy(cfg)  # same weights as the reference
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    generator = torch.Generator().manual_seed(seed)
    running = RunningMean(cfg.ema_decay)
    history, snapshots = [], {}
    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            history.append(
                {
                    "step": step,
                    "sequences_seen": step * cfg.batch_size,
                    **exact_metrics(policy, task),
                }
            )
        if step in snapshot_steps:
            snapshots[step] = copy.deepcopy(policy.state_dict())
        if step == cfg.steps:
            break
        loss = reinforce_batch_loss(policy, task, baseline, generator, running)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return history, snapshots


def gradient_variance(
    task: Task, state: dict[str, Tensor], baseline: str, seed: int, warmup_batches: int = 50
) -> dict[str, float]:
    """Spread of the REINFORCE gradient estimate at FIXED parameters.

    Draws `grad_variance_batches` independent batches, computes the gradient vector of
    each, and reports the trace of their covariance (the mean squared distance of a
    batch gradient from the average gradient) and the squared norm of that average.
    The running-mean baseline is first warmed up on `warmup_batches` batches.
    """
    cfg = task.cfg
    policy = make_policy(cfg)
    policy.load_state_dict(state)
    generator = torch.Generator().manual_seed(seed)
    running = RunningMean(cfg.ema_decay)
    params = list(policy.parameters())
    if baseline == "running_mean":
        with torch.no_grad():
            for _ in range(warmup_batches):
                seqs = policy.sample(cfg.batch_size, generator)
                logp = policy.log_prob(seqs)
                rewards = reward(seqs, cfg.vocab_size, cfg.stop_bonus)
                running.update(
                    kl_penalised_return(rewards, logp, task.reference.log_prob(seqs), cfg.beta)
                )
    grads = []
    for _ in range(cfg.grad_variance_batches):
        loss = reinforce_batch_loss(policy, task, baseline, generator, running)
        grads.append(torch.cat([g.flatten() for g in torch.autograd.grad(loss, params)]))
    stacked = torch.stack(grads).double()
    mean = stacked.mean(dim=0)
    return {
        "variance": (stacked - mean).pow(2).sum(dim=1).mean().item(),
        "mean_grad_sq_norm": mean.pow(2).sum().item(),
    }


# --------------------------------------------------------------------------------------
# DPO on the same toy
# --------------------------------------------------------------------------------------


def make_preference_pairs(
    task: Task, n_pairs: int, labels: str, generator: torch.Generator
) -> tuple[Tensor, Tensor]:
    """A fixed offline dataset: pairs of sequences sampled from the REFERENCE, labelled
    by the reward function. Returns (chosen, rejected), each [n_pairs, seq_len].

    labels="bradley_terry": the first sequence is preferred with probability
        sigmoid(r_a - r_b). This is the annotator the Bradley-Terry model assumes, and
        under it DPO's best fit is exactly pi* for the same beta.
    labels="hard": the higher-reward sequence is always preferred (ties broken by a coin
        flip). No finite reward difference explains labels that are never wrong, so the
        DPO loss keeps pushing margins apart.
    """
    if labels not in LABEL_MODES:
        raise ValueError(f"unknown labels {labels!r}, expected one of {LABEL_MODES}")
    cfg = task.cfg
    a = task.reference.sample(n_pairs, generator)
    b = task.reference.sample(n_pairs, generator)
    diff = reward(a, cfg.vocab_size, cfg.stop_bonus) - reward(b, cfg.vocab_size, cfg.stop_bonus)
    coin = torch.rand(n_pairs, generator=generator, dtype=torch.float64)
    if labels == "bradley_terry":
        a_wins = coin < torch.sigmoid(diff)
    else:
        a_wins = (diff > 0) | ((diff == 0) & (coin < 0.5))
    pick = a_wins.unsqueeze(1)
    return torch.where(pick, a, b), torch.where(pick, b, a)


def train_dpo_toy(task: Task, labels: str, seed: int) -> list[dict[str, float]]:
    """Train a copy of the reference with DPO on a fixed set of pairs, one pass.

    Each step uses batch_size / 2 pairs, which is batch_size sequences, so a DPO step
    and a REINFORCE step consume the same number of sequences. DPO never samples from
    the policy being trained and never sees a reward value, only which of two was preferred.
    """
    cfg = task.cfg
    pairs_per_step = cfg.batch_size // 2
    generator = torch.Generator().manual_seed(seed)
    chosen, rejected = make_preference_pairs(task, cfg.steps * pairs_per_step, labels, generator)
    with torch.no_grad():
        ref_chosen, ref_rejected = (
            task.reference.log_prob(chosen),
            task.reference.log_prob(rejected),
        )
    policy = make_policy(cfg)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    history = []
    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            history.append(
                {
                    "step": step,
                    "sequences_seen": step * cfg.batch_size,
                    **exact_metrics(policy, task),
                }
            )
        if step == cfg.steps:
            break
        rows = slice(step * pairs_per_step, (step + 1) * pairs_per_step)
        out = dpo_loss(
            policy.log_prob(chosen[rows]),
            policy.log_prob(rejected[rows]),
            ref_chosen[rows],
            ref_rejected[rows],
            beta=cfg.beta,
        )
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        optimizer.step()
    return history


# --------------------------------------------------------------------------------------
# The whole experiment
# --------------------------------------------------------------------------------------


def run_experiment(cfg: ToyConfig) -> dict[str, Any]:
    """Run every method for every seed. Returns curves, gradient variances and a summary."""
    task = make_task(cfg)
    curves: list[dict[str, Any]] = []
    snapshots: dict[int, dict[str, Tensor]] = {}
    for baseline in BASELINES:
        for seed in cfg.seeds:
            want = tuple(cfg.grad_variance_steps) if (baseline, seed) == (BASELINES[-1], 0) else ()
            history, snaps = train_reinforce(task, baseline, seed, snapshot_steps=want)
            snapshots.update(snaps)
            curves += [{"method": f"reinforce_{baseline}", "seed": seed, **row} for row in history]
    for labels in LABEL_MODES:
        for seed in cfg.seeds:
            history = train_dpo_toy(task, labels, seed)
            curves += [{"method": f"dpo_{labels}", "seed": seed, **row} for row in history]

    # Gradient variance at fixed parameters taken from one run (leave-one-out, seed 0).
    grad_variance = [
        {"at_step": step, "baseline": baseline, **gradient_variance(task, state, baseline, seed=0)}
        for step, state in sorted(snapshots.items())
        for baseline in BASELINES
    ]

    extended_curves: list[dict[str, Any]] = []
    if cfg.dpo_extended_steps > 0:
        long_task = dataclasses.replace(
            task, cfg=dataclasses.replace(cfg, steps=cfg.dpo_extended_steps, eval_every=100)
        )
        for labels in LABEL_MODES:
            history = train_dpo_toy(long_task, labels, cfg.seeds[0])
            extended_curves += [
                {"method": f"dpo_{labels}_extended", "seed": cfg.seeds[0], **row} for row in history
            ]

    final = [row for row in curves if row["step"] == cfg.steps]
    summary = {}
    for method in dict.fromkeys(row["method"] for row in curves):
        rows = [row for row in final if row["method"] == method]
        summary[method] = {
            metric: {
                "mean": sum(r[metric] for r in rows) / len(rows),
                "min": min(r[metric] for r in rows),
                "max": max(r[metric] for r in rows),
            }
            for metric in ("expected_reward", "kl_from_ref", "objective", "kl_to_opt", "tv_to_opt")
        }
    return {
        "curves": curves,
        "extended_curves": extended_curves,
        "grad_variance": grad_variance,
        "summary": summary,
        "task": {
            "n_sequences": int(task.all_seqs.shape[0]),
            "max_reward": task.rewards.max().item(),
            "reference_expected_reward": (task.ref_logp.exp() * task.rewards).sum().item(),
            "optimal_expected_reward": task.optimal_expected_reward,
            "optimal_objective": task.optimal_objective,
            "optimal_kl_from_ref": (task.opt_logp.exp() * (task.opt_logp - task.ref_logp))
            .sum()
            .item(),
            "log2": math.log(2.0),
            "n_parameters": sum(p.numel() for p in task.reference.parameters()),
        },
    }
