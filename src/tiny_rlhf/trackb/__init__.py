"""Track B: a templated preference dataset for staying on the user's task when the
document being processed contains unrelated instructions. See docs/DATASHEET.md."""

from tiny_rlhf.trackb.checks import (
    flagged_document,
    followed_injection,
    refused,
    score,
    task_correct,
)
from tiny_rlhf.trackb.generate import HELD_OUT, generate, write_dataset
from tiny_rlhf.trackb.load import load_trackb_examples, load_trackb_pairs

__all__ = [
    "HELD_OUT",
    "flagged_document",
    "followed_injection",
    "generate",
    "load_trackb_examples",
    "load_trackb_pairs",
    "refused",
    "score",
    "task_correct",
    "write_dataset",
]
