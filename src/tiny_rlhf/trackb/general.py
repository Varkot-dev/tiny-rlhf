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
from tiny_rlhf.trackb.evaluate import (
    DEFAULT_RESAMPLES,
    bootstrap_interval,
    paired_bootstrap_difference,
)


@torch.no_grad()
def response_nll_and_kl(
    base: Any,
    trained: Any,
    pairs: list[TokenizedPair],
    pad_token_id: int,
    max_length: int | None,
    batch_size: int = 4,
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
        # Position t-1 predicts token t, as in tiny_rlhf.losses.sequence_logprob.
        base_logp = F.log_softmax(base_logits[:, :-1].float(), dim=-1)
        trained_logp = F.log_softmax(trained_logits[:, :-1].float(), dim=-1)
        targets = ids[:, 1:].unsqueeze(-1)
        n_tokens = mask.sum(dim=1).clamp(min=1)

        def mean_over_response(per_position: torch.Tensor, m=mask, n=n_tokens) -> torch.Tensor:
            return torch.where(m, per_position, torch.zeros_like(per_position)).sum(dim=1) / n

        nll_base = mean_over_response(-base_logp.gather(-1, targets).squeeze(-1))
        nll_trained = mean_over_response(-trained_logp.gather(-1, targets).squeeze(-1))
        # KL(base || trained) = sum over the vocabulary of p_base * (log p_base - log p_trained).
        kl = mean_over_response((base_logp.exp() * (base_logp - trained_logp)).sum(dim=-1))
        out["nll_base"] += nll_base.tolist()
        out["nll_trained"] += nll_trained.tolist()
        out["kl"] += kl.tolist()
        out["n_tokens"] += mask.sum(dim=1).tolist()
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
