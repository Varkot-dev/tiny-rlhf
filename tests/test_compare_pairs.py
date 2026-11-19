"""Tests for scripts/compare_pairs.py: paired comparison of two runs on the same pairs."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_pairs", ROOT / "scripts/compare_pairs.py")
compare_pairs_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_pairs_module)


def test_counts_and_changes_on_hand_built_margins() -> None:
    a = {0: 1.0, 1: -1.0, 2: 0.0, 3: 2.0}  # right, wrong, tie (wrong), right
    b = {0: 1.5, 1: 0.5, 2: -0.5, 3: -1.0}  # right, right, wrong, wrong
    out = compare_pairs_module.compare_pairs(a, b, n_resamples=200)
    assert out["accuracy_a"] == pytest.approx(0.5)
    assert out["accuracy_b"] == pytest.approx(0.5)
    assert out["accuracy_change_b_minus_a"]["difference"] == pytest.approx(0.0)
    assert out["margin_change_b_minus_a"]["difference"] == pytest.approx((0.5 + 1.5 - 0.5 - 3) / 4)
    assert (out["both_right"], out["only_a_right"], out["only_b_right"], out["both_wrong"]) == (
        1,
        1,
        1,
        1,
    )


def test_refuses_runs_that_scored_different_pairs() -> None:
    with pytest.raises(ValueError):
        compare_pairs_module.compare_pairs({0: 1.0, 1: 1.0}, {0: 1.0, 2: 1.0})


def write_run(path: Path, margins: list[float], sha: str) -> None:
    path.mkdir(parents=True)
    (path / "run_info.json").write_text(json.dumps({"eval_pairs_sha256": sha}))
    rows = "".join(f"{i},{m}\n" for i, m in enumerate(margins))
    (path / "eval_pairs_final.csv").write_text("pair_index,reward_margin\n" + rows)


def test_main_checks_fingerprints_and_writes_the_result(tmp_path: Path, monkeypatch) -> None:
    write_run(tmp_path / "a", [1.0, -1.0, 1.0, -1.0], "same")
    write_run(tmp_path / "b", [1.0, 1.0, 1.0, -1.0], "same")
    write_run(tmp_path / "c", [1.0, 1.0, 1.0, -1.0], "other")
    out = tmp_path / "out" / "cmp.json"
    argv = ["compare_pairs.py", "--a", str(tmp_path / "a"), "--b", str(tmp_path / "b")]
    monkeypatch.setattr(sys, "argv", [*argv, "--out", str(out), "--resamples", "100"])
    compare_pairs_module.main()
    saved = json.loads(out.read_text())
    assert saved["accuracy_change_b_minus_a"]["difference"] == pytest.approx(0.25)
    assert saved["eval_pairs_sha256"] == "same"
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_pairs.py", "--a", str(tmp_path / "a"), "--b", str(tmp_path / "c"), "--out", "x"],
    )
    with pytest.raises(SystemExit):
        compare_pairs_module.main()


@pytest.mark.skipif(
    not (ROOT / "results/sweeps/label_noise/robust_vs_ordinary.json").exists(),
    reason="no saved comparison",
)
def test_saved_comparison_agrees_with_the_sweep_summary() -> None:
    saved = json.loads((ROOT / "results/sweeps/label_noise/robust_vs_ordinary.json").read_text())
    summary = json.loads((ROOT / "results/sweeps/label_noise/summary.json").read_text())
    accuracy = {r["label"]: r["accuracy"]["mean"] for r in summary["runs"]}
    assert saved["accuracy_a"] == pytest.approx(accuracy["0.3"])
    assert saved["accuracy_b"] == pytest.approx(accuracy["0.3_robust"])
