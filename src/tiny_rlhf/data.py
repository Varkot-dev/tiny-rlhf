"""Turning preference pairs into tensors.

A pair is one prompt with two responses: the preferred one ("chosen") and the other
one ("rejected"). Three steps:

tokenize_pair       conversation messages -> prompt, chosen and rejected token ids
PreferenceCollator  a list of pairs -> one padded batch plus the response mask
load_pairs          a Hugging Face preference dataset -> a list of tokenized pairs

The layout mirrors TRL's DPOTrainer (trl 1.18.0) so that the M2 parity run feeds both
implementations the same token sequences:
- the prompt is rendered with the chat template and add_generation_prompt=True,
- a response is everything the template adds after that, which includes the
  end-of-turn token (the model must learn to stop) and anything following it,
- sequences are truncated from the right ("keep the start") and padded on the right,
- a batch stacks all chosen rows first and all rejected rows second.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

Message = dict[str, str]


@dataclass(frozen=True)
class TokenizedPair:
    """Token ids for one preference pair. The prompt is shared by both responses."""

    prompt_ids: list[int]
    chosen_ids: list[int]
    rejected_ids: list[int]


def _common_prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _template_ids(
    tokenizer: Any, messages: list[Message], add_generation_prompt: bool
) -> list[int]:
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt, return_dict=True
    )
    return list(out["input_ids"])


def tokenize_pair(tokenizer: Any, chosen: list[Message], rejected: list[Message]) -> TokenizedPair:
    """Tokenize one pair given as two full conversations that share every message but the last.

    Args:
        tokenizer: a Hugging Face tokenizer with a chat template.
        chosen: messages ending with the preferred assistant response.
        rejected: the same messages ending with the other assistant response.

    Returns:
        TokenizedPair where prompt_ids + chosen_ids is the full chosen conversation
        and prompt_ids + rejected_ids is the full rejected conversation.

    Raises:
        ValueError: if the two conversations do not share the same prompt messages,
            if the last message is not from the assistant, or if a response is empty.
    """
    if chosen[:-1] != rejected[:-1] or not chosen[:-1]:
        raise ValueError("chosen and rejected must share the same non-empty prompt messages")
    if chosen[-1]["role"] != "assistant" or rejected[-1]["role"] != "assistant":
        raise ValueError("the last message of each conversation must be the assistant response")

    prompt_ids = _template_ids(tokenizer, chosen[:-1], add_generation_prompt=True)
    full_chosen = _template_ids(tokenizer, chosen, add_generation_prompt=False)
    full_rejected = _template_ids(tokenizer, rejected, add_generation_prompt=False)

    # The response starts where prompt and prompt+response stop agreeing. For the chat
    # templates used here that is exactly len(prompt_ids), but some templates render a
    # lone prompt differently, so measure it instead of assuming (TRL does the same).
    prompt_len = min(
        _common_prefix_length(prompt_ids, full_chosen),
        _common_prefix_length(prompt_ids, full_rejected),
    )
    pair = TokenizedPair(
        prompt_ids=prompt_ids[:prompt_len],
        chosen_ids=full_chosen[prompt_len:],
        rejected_ids=full_rejected[prompt_len:],
    )
    if not pair.prompt_ids or not pair.chosen_ids or not pair.rejected_ids:
        raise ValueError("tokenization produced an empty prompt or an empty response")
    return pair


class PreferenceCollator:
    """Pad a list of pairs into one batch of 2B rows: B chosen rows, then B rejected rows.

    Args:
        pad_token_id: id used to fill the right side of shorter rows. Its value never
            matters to the loss because padded positions are masked out.
        max_length: if set, each row (prompt + response) is cut to its first
            max_length tokens. A long response loses its tail; the prompt is kept.
    """

    def __init__(self, pad_token_id: int, max_length: int | None = None) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, pairs: list[TokenizedPair]) -> dict[str, Tensor]:
        rows: list[list[int]] = []
        is_response: list[list[bool]] = []
        for response_field in ("chosen_ids", "rejected_ids"):
            for pair in pairs:
                response = getattr(pair, response_field)
                ids = (pair.prompt_ids + response)[: self.max_length]
                mask = ([False] * len(pair.prompt_ids) + [True] * len(response))[: self.max_length]
                rows.append(ids)
                is_response.append(mask)

        width = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        response_mask = torch.zeros((len(rows), width), dtype=torch.bool)
        for i, (ids, mask) in enumerate(zip(rows, is_response, strict=True)):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
            response_mask[i, : len(ids)] = torch.tensor(mask, dtype=torch.bool)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
        }


def truncation_stats(pairs: list[TokenizedPair], max_length: int | None) -> dict[str, float]:
    """How much of the data max_length cuts off. Recorded with every run.

    Returns the number of pairs, the fraction of chosen and of rejected responses that
    lose at least one token, and the mean number of response tokens actually scored.
    """
    n = len(pairs)
    if n == 0:
        return {"n_pairs": 0}
    limit = float("inf") if max_length is None else max_length

    def kept(pair: TokenizedPair, response: list[int]) -> int:
        return int(min(len(response), max(limit - len(pair.prompt_ids), 0)))

    chosen_kept = [kept(p, p.chosen_ids) for p in pairs]
    rejected_kept = [kept(p, p.rejected_ids) for p in pairs]
    return {
        "n_pairs": n,
        "frac_chosen_truncated": sum(
            k < len(p.chosen_ids) for k, p in zip(chosen_kept, pairs, strict=True)
        )
        / n,
        "frac_rejected_truncated": sum(
            k < len(p.rejected_ids) for k, p in zip(rejected_kept, pairs, strict=True)
        )
        / n,
        "mean_chosen_tokens_scored": sum(chosen_kept) / n,
        "mean_rejected_tokens_scored": sum(rejected_kept) / n,
        "mean_prompt_tokens": sum(len(p.prompt_ids) for p in pairs) / n,
    }


def load_pairs(
    dataset_name: str,
    split: str,
    n: int,
    seed: int,
    tokenizer: Any,
    max_length: int | None,
) -> list[TokenizedPair]:
    """Load n tokenized pairs from a Hugging Face dataset with `chosen` / `rejected` columns.

    The split is shuffled with `seed` and the first n rows are taken, so the same
    (dataset, split, n, seed) always gives the same pairs in the same order. Pairs
    whose prompt alone fills max_length are dropped, because truncation would leave
    no response token to score; the returned list can therefore be shorter than n.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split).shuffle(seed=seed)
    dataset = dataset.select(range(min(n, len(dataset))))
    pairs = [tokenize_pair(tokenizer, row["chosen"], row["rejected"]) for row in dataset]
    if max_length is not None:
        pairs = [p for p in pairs if len(p.prompt_ids) < max_length]
    return pairs
