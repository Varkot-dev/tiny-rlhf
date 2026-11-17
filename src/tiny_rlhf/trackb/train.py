"""Track B training: the hand-written DPO loop, pointed at the Track B files.

Nothing about the method changes. tiny_rlhf.train_dpo supplies the config, the model
loading and the loop; this module only swaps where the pairs come from:

    training pairs   <dataset_name>/train.jsonl
    held-out pairs   <dataset_name>/val.jsonl   (same categories as training, new instances)

The test files are never read here. They are for scripts/eval_trackb.py, which scores
generated text, not log-probabilities.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from tiny_rlhf.data import TokenizedPair, tokenize_pair
from tiny_rlhf.trackb.load import load_trackb_examples
from tiny_rlhf.train_dpo import DPOConfig, load_models, train


def select_pairs(
    path: str | Path, tokenizer: Any, max_length: int | None, n: int, seed: int
) -> list[TokenizedPair]:
    """The first n examples of a Track B file after a seeded shuffle, tokenized.

    Same contract as tiny_rlhf.data.load_pairs: the same (file, n, seed) always gives
    the same pairs in the same order, and a pair whose prompt alone fills max_length is
    dropped, so the list can be shorter than n.
    """
    examples = load_trackb_examples(path)
    random.Random(seed).shuffle(examples)
    pairs = [tokenize_pair(tokenizer, e["chosen"], e["rejected"]) for e in examples[:n]]
    if max_length is not None:
        pairs = [p for p in pairs if len(p.prompt_ids) < max_length]
    return pairs


def main(cfg: DPOConfig, results_root: str | Path = "results") -> dict[str, Any]:
    """Load models and Track B pairs, then run the hand-written DPO loop."""
    data_dir = Path(cfg.dataset_name)
    if not (data_dir / "train.jsonl").exists():
        raise FileNotFoundError(
            f"{data_dir}/train.jsonl not found. For Track B, dataset_name is the directory "
            "holding the generated files (data/trackb)."
        )
    tokenizer, policy, reference = load_models(cfg)
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
