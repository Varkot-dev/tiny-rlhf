"""Bootstrap confidence intervals, shared by every evaluation in the repo.

A bootstrap interval answers: how much would this number move if a different sample of
the same size had been drawn? It treats the examples we have as the population,
redraws from them with replacement many times, and looks at how the statistic varies.

tiny_rlhf.reward_model.bootstrap_ci is an older, separate implementation of the same
percentile interval that draws its resamples with torch instead of numpy. It is kept
as it is so that reward-model runs already saved under results/ stay reproducible
from their seed; new code should use the functions here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_RESAMPLES = 2000


def bootstrap_interval(
    values: Sequence[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of `values`.

    Resample the examples with replacement n_resamples times, take the mean of each
    resample, and report the middle `level` share of those means.
    """
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, data.size, size=(n_resamples, data.size))
    means = data[idx].mean(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return (float(low), float(high))


def paired_bootstrap_difference(
    before: Sequence[float],
    after: Sequence[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float | None]:
    """Mean of (after - before) on the same examples, with a paired bootstrap interval.

    Pairing matters: the two models answer the same prompts, so most of the
    example-to-example variation is shared and cancels in the difference. Resampling
    the two lists independently would throw that away and give a wider interval.
    """
    a = np.asarray(before, dtype=np.float64)
    b = np.asarray(after, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired lists differ in length: {a.size} vs {b.size}")
    if a.size == 0:
        return {"difference": None, "ci_low": None, "ci_high": None, "n": 0}
    diffs = b - a
    low, high = bootstrap_interval(diffs, n_resamples, seed, level)
    return {"difference": float(diffs.mean()), "ci_low": low, "ci_high": high, "n": int(a.size)}
