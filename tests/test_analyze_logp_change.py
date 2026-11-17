"""Tests for the per-pair log-probability change summary, on hand-built numbers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "analyze_logp_change",
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_logp_change.py",
)
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)


def test_quadrants_accuracy_and_per_token_values() -> None:
    # Four pairs, one in each quadrant.
    chosen = np.array([-2.0, 3.0, 1.0, -1.0])
    rejected = np.array([-6.0, -1.0, 2.0, 4.0])
    tokens_c = np.array([10, 10, 10, 10])
    tokens_r = np.array([20, 20, 20, 20])
    stats = analyze.logp_change_stats(
        chosen, rejected, tokens_c, tokens_r, np.array([False, True, False, False])
    )
    assert stats["frac_both_down"] == 0.25
    assert stats["frac_chosen_up_rejected_down"] == 0.25
    assert stats["frac_both_up"] == 0.25
    assert stats["frac_chosen_down_rejected_up"] == 0.25
    # chosen - rejected: +4, +4, -1, -5 -> two of four ranked right
    assert stats["preference_accuracy"] == 0.5
    assert stats["mean_change_chosen"] == pytest.approx(0.25)
    assert stats["mean_change_per_token_chosen"] == pytest.approx(0.025)
    assert stats["mean_change_per_token_rejected"] == pytest.approx(-0.0125)
    assert stats["n_chosen_truncated"] == 1
    assert stats["mean_change_per_token_chosen_truncated"] == pytest.approx(0.3)


def test_share_of_drop_from_the_largest_drops() -> None:
    # Ten pairs: one rejected response lost 91 nats, nine lost 1 each -> 91 / 100.
    rejected = np.array([-91.0] + [-1.0] * 9)
    chosen = np.zeros(10)
    ones = np.ones(10)
    stats = analyze.logp_change_stats(chosen + 0.5, rejected, ones, ones, np.zeros(10, bool))
    assert stats["share_of_rejected_drop_from_top_10pct_pairs"] == pytest.approx(0.91)


def test_correlation_with_length() -> None:
    tokens = np.array([10, 20, 30, 40])
    stats = analyze.logp_change_stats(
        -0.1 * tokens, -0.2 * tokens, tokens, tokens, np.zeros(4, bool)
    )
    assert stats["corr_change_vs_tokens_chosen"] == pytest.approx(-1.0)
