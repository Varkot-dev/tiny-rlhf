"""Reading Track B files back in."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tiny_rlhf.data import TokenizedPair, tokenize_pair


def load_trackb_examples(path: str | Path) -> list[dict[str, Any]]:
    """Read one Track B JSON Lines file into a list of example dicts, in file order."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_trackb_pairs(
    path: str | Path,
    tokenizer: Any,
    max_length: int | None,
    n: int | None = None,
) -> list[TokenizedPair]:
    """Tokenize the first n examples of a Track B file (all of them if n is None).

    As in tiny_rlhf.data.load_pairs, a pair whose prompt alone fills max_length is
    dropped, so the result can be shorter than n. The files are already shuffled by
    construction (each example is drawn independently), so taking a prefix is unbiased.
    """
    examples = load_trackb_examples(path)
    if n is not None:
        examples = examples[:n]
    pairs = [tokenize_pair(tokenizer, e["chosen"], e["rejected"]) for e in examples]
    if max_length is not None:
        pairs = [p for p in pairs if len(p.prompt_ids) < max_length]
    return pairs
