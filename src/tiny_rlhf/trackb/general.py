"""A small general regression check for a Track B adapter.

Track B training only ever shows the model short answers about templated documents.
This check asks whether that training disturbed the model on ordinary chat data it
was never trained on here: held-out conversations from the Track A dataset.

For each conversation it computes, over the tokens of the dataset's preferred response:

    nll_base      mean negative log-likelihood per token under the base model
    nll_trained   the same under the base model plus the trained adapter
    kl            mean over response positions of KL(base || trained), the divergence
                  between the two models' next-token distributions

What it shows: whether the trained model still assigns about the same probability to
ordinary human-preferred responses (nll_trained - nll_base near 0) and how far its
next-token distributions moved (kl near 0 means it behaves like the base model here).

What it does not show: that generated text is as good as before (nothing is generated
or judged), that any capability was kept or lost, or anything about prompts unlike this
dataset. A low KL on these conversations is necessary for "no regression", not proof.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from tiny_rlhf.data import PreferenceCollator, TokenizedPair
from tiny_rlhf.stats import DEFAULT_RESAMPLES, bootstrap_interval, paired_bootstrap_difference

# Response positions are scored this many at a time. Each chunk builds a few float32
# tensors of shape [chunk, vocab]; with a 152k vocabulary 256 rows is about 150 MB each.
ROW_CHUNK = 256


def _nll_and_kl_rows(
    base_rows: torch.Tensor, trained_rows: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-position (nll_base, nll_trained, kl) for logits of shape [n_positions, vocab]."""
    nll_base, nll_trained, kl = [], [], []
    for start in range(0, base_rows.shape[0], ROW_CHUNK):
        rows = slice(start, start + ROW_CHUNK)
        base_logp = F.log_softmax(base_rows[rows].float(), dim=-1)
        trained_logp = F.log_softmax(trained_rows[rows].float(), dim=-1)
        target = targets[rows].unsqueeze(-1)
        nll_base.append(-base_logp.gather(-1, target).squeeze(-1))
        nll_trained.append(-trained_logp.gather(-1, target).squeeze(-1))
        # KL(base || trained) = sum over the vocabulary of p_base * (log p_base - log p_trained).
        kl.append((base_logp.exp() * (base_logp - trained_logp)).sum(dim=-1))
    return torch.cat(nll_base), torch.cat(nll_trained), torch.cat(kl)


@torch.no_grad()
def response_nll_and_kl(
    base: Any,
    trained: Any,
    pairs: list[TokenizedPair],
    pad_token_id: int,
    max_length: int | None,
    batch_size: int = 2,
    device: torch.device | str = "cpu",
    amp: bool = False,
) -> dict[str, list[float]]:
    """Per-conversation nll_base, nll_trained and kl, over the chosen response's tokens."""
    device = torch.device(device)
    collate = PreferenceCollator(pad_token_id=pad_token_id, max_length=max_length)
    out: dict[str, list[float]] = {"nll_base": [], "nll_trained": [], "kl": [], "n_tokens": []}
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        batch = collate(chunk)
        # The collator stacks chosen rows first; only those are needed here.
        ids = batch["input_ids"][: len(chunk)].to(device)
        attention = batch["attention_mask"][: len(chunk)].to(device)
        mask = batch["response_mask"][: len(chunk)].to(device)[:, 1:]
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            base_logits = base(input_ids=ids, attention_mask=attention).logits
            trained_logits = trained(input_ids=ids, attention_mask=attention).logits
        # Position t-1 predicts token t, as in tiny_rlhf.losses.sequence_logprob. Keep
        # only the response positions before any softmax: a float32 log_softmax over
        # every position and a 152k vocabulary does not fit next to two models on a T4.
        nll_base, nll_trained, kl = _nll_and_kl_rows(
            base_logits[:, :-1][mask], trained_logits[:, :-1][mask], ids[:, 1:][mask]
        )
        sequence_index = mask.nonzero(as_tuple=True)[0]
        n_tokens = mask.sum(dim=1)

        def mean_over_response(
            per_position: torch.Tensor, index=sequence_index, n=n_tokens
        ) -> torch.Tensor:
            totals = torch.zeros(len(n), dtype=per_position.dtype, device=per_position.device)
            return totals.index_add(0, index, per_position) / n.clamp(min=1)

        out["nll_base"] += mean_over_response(nll_base).tolist()
        out["nll_trained"] += mean_over_response(nll_trained).tolist()
        out["kl"] += mean_over_response(kl).tolist()
        out["n_tokens"] += n_tokens.tolist()
    return out


def summarize_general(
    values: dict[str, list[float]], n_resamples: int = DEFAULT_RESAMPLES, seed: int = 0
) -> dict[str, Any]:
    """Means with bootstrap intervals over conversations, and the paired NLL change."""

    def mean_ci(xs: list[float]) -> dict[str, float]:
        low, high = bootstrap_interval(xs, n_resamples, seed)
        return {"mean": sum(xs) / len(xs), "ci_low": low, "ci_high": high}

    return {
        "n_conversations": len(values["kl"]),
        "mean_response_tokens_scored": sum(values["n_tokens"]) / len(values["n_tokens"]),
        "nll_per_token_base": mean_ci(values["nll_base"]),
        "nll_per_token_trained": mean_ci(values["nll_trained"]),
        "nll_per_token_change": paired_bootstrap_difference(
            values["nll_base"], values["nll_trained"], n_resamples, seed
        ),
        "kl_base_to_trained_per_token": mean_ci(values["kl"]),
        "interval": "95% percentile bootstrap over conversations",
        "n_resamples": n_resamples,
        "seed": seed,
    }
