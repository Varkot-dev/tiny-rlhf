"""Tests for tokenizing and batching preference pairs. All offline: the tokenizer is
a word-level toy with a ChatML-style chat template, built in memory."""

from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from tiny_rlhf.data import PreferenceCollator, TokenizedPair, tokenize_pair, truncation_stats
from tiny_rlhf.losses import sequence_logprob

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> "
    "{% endfor %}{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
)
WORDS = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>", "user", "assistant"] + list("abcdefghij")


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<|im_end|>",
        chat_template=CHAT_TEMPLATE,
    )


def conv(prompt: str, response: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]


def ids(tokenizer: PreTrainedTokenizerFast, text: str) -> list[int]:
    return [tokenizer.convert_tokens_to_ids(w) for w in text.split()]


def test_tokenize_pair_splits_prompt_from_response(tokenizer: PreTrainedTokenizerFast) -> None:
    pair = tokenize_pair(tokenizer, conv("a b", "c d e"), conv("a b", "f"))
    # The prompt ends with the assistant header; the response ends with the end-of-turn token.
    assert pair.prompt_ids == ids(
        tokenizer, "<|im_start|> user a b <|im_end|> <|im_start|> assistant"
    )
    assert pair.chosen_ids == ids(tokenizer, "c d e <|im_end|>")
    assert pair.rejected_ids == ids(tokenizer, "f <|im_end|>")


def test_prompt_plus_response_is_the_full_templated_conversation(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    chosen = conv("a b c", "d e")
    pair = tokenize_pair(tokenizer, chosen, conv("a b c", "g h i j"))
    full = tokenizer.apply_chat_template(chosen, tokenize=True, return_dict=True)["input_ids"]
    assert pair.prompt_ids + pair.chosen_ids == list(full)


def test_tokenize_pair_rejects_different_prompts(tokenizer: PreTrainedTokenizerFast) -> None:
    with pytest.raises(ValueError, match="same non-empty prompt"):
        tokenize_pair(tokenizer, conv("a", "c"), conv("b", "c"))


def test_tokenize_pair_rejects_non_assistant_last_message(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    bad = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    with pytest.raises(ValueError, match="assistant"):
        tokenize_pair(tokenizer, bad, bad)


PAIRS = [
    TokenizedPair(prompt_ids=[1, 2, 3], chosen_ids=[4, 5], rejected_ids=[6]),
    TokenizedPair(prompt_ids=[7, 8], chosen_ids=[9], rejected_ids=[10, 11]),
]


def test_collator_matches_hand_built_batch() -> None:
    # Same example as the docstring of TRL's DataCollatorForPreference (trl 1.18.0).
    batch = PreferenceCollator(pad_token_id=0)(PAIRS)
    assert batch["input_ids"].tolist() == [
        [1, 2, 3, 4, 5],  # pair 0, chosen
        [7, 8, 9, 0, 0],  # pair 1, chosen
        [1, 2, 3, 6, 0],  # pair 0, rejected
        [7, 8, 10, 11, 0],  # pair 1, rejected
    ]
    assert batch["attention_mask"].tolist() == [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0],
    ]
    assert batch["response_mask"].long().tolist() == [
        [0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0],
    ]
    assert batch["response_mask"].dtype == torch.bool
    assert batch["input_ids"].dtype == torch.long


def test_collator_truncates_from_the_right_and_keeps_the_prompt() -> None:
    batch = PreferenceCollator(pad_token_id=0, max_length=4)(PAIRS)
    assert batch["input_ids"].tolist() == [
        [1, 2, 3, 4],  # chosen response lost its last token
        [7, 8, 9, 0],
        [1, 2, 3, 6],
        [7, 8, 10, 11],
    ]
    assert batch["response_mask"].long().sum(dim=1).tolist() == [1, 1, 1, 2]


def test_response_mask_never_marks_padding_or_position_zero() -> None:
    batch = PreferenceCollator(pad_token_id=0)(PAIRS)
    assert not (batch["response_mask"] & (batch["attention_mask"] == 0)).any()
    assert not batch["response_mask"][:, 0].any()


def test_batch_logprob_equals_scoring_each_row_alone() -> None:
    # Padding and batching must not change any row's log-probability.
    torch.manual_seed(0)
    vocab = 12
    table = torch.randn(vocab, vocab, dtype=torch.float64)  # bigram "model": logits = table[token]
    batch = PreferenceCollator(pad_token_id=0)(PAIRS)
    together = sequence_logprob(
        table[batch["input_ids"]], batch["input_ids"], batch["response_mask"]
    )
    singles = []
    for response_field in ("chosen_ids", "rejected_ids"):
        for pair in PAIRS:
            one = PreferenceCollator(pad_token_id=0)(
                [TokenizedPair(pair.prompt_ids, getattr(pair, response_field), [0])]
            )
            row = one["input_ids"][:1]
            singles.append(sequence_logprob(table[row], row, one["response_mask"][:1]))
    assert torch.allclose(together, torch.cat(singles), atol=1e-12)


def test_pad_token_id_does_not_change_the_logprob() -> None:
    torch.manual_seed(0)
    table = torch.randn(12, 12, dtype=torch.float64)
    a = PreferenceCollator(pad_token_id=0)(PAIRS)
    b = PreferenceCollator(pad_token_id=5)(PAIRS)
    assert torch.equal(
        sequence_logprob(table[a["input_ids"]], a["input_ids"], a["response_mask"]),
        sequence_logprob(table[b["input_ids"]], b["input_ids"], b["response_mask"]),
    )


def test_truncation_stats() -> None:
    stats = truncation_stats(PAIRS, max_length=4)
    assert stats["n_pairs"] == 2
    assert stats["frac_chosen_truncated"] == 0.5  # pair 0 chosen: 3 prompt + 2 response > 4
    assert stats["frac_rejected_truncated"] == 0.0
    assert stats["mean_chosen_tokens_scored"] == 1.0
    assert stats["mean_rejected_tokens_scored"] == 1.5
    assert truncation_stats(PAIRS, max_length=None)["frac_chosen_truncated"] == 0.0
