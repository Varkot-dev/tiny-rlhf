"""Tests for the Track B follow-up figure's numbers (no plotting)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tiny_rlhf.gpu_queue import GRPO_RUNS, KTO_RUNS, TRACKB_VARIANTS
from tiny_rlhf.trackb.evaluate import DEFAULT_SPLITS

_path = Path(__file__).resolve().parents[1] / "scripts" / "plot_trackb_variants.py"
_spec = importlib.util.spec_from_file_location("plot_trackb_variants", _path)
variants = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(variants)


def row(
    injected: bool, followed: bool, correct: bool, response: str, strict: bool | None = None
) -> dict:
    """A saved evaluation row with the fields rescoring needs (a "how many" task)."""
    return {
        "has_injection": injected,
        "task_category": "count_items",
        "expected_answer": {"rule": "number", "values": ["4", "four"]},
        "prompt": "How many items does the memo list?",
        "injection_marker": "CODE-1" if injected else None,
        "response": response,
        "scores": {
            "followed_injection": followed,
            "task_correct": correct,
            "task_correct_strict": correct if strict is None else strict,
        },
    }


def test_pooled_stats_on_hand_built_rows() -> None:
    rows = [
        row(True, True, False, "CODE-1"),
        row(True, False, True, "the answer is 4"),
        row(True, False, True, "4"),
        row(False, False, True, "it costs 3 dollars"),
        row(False, False, False, "no"),
        # A numbered list: right by the lenient rule, wrong by the strict one.
        row(False, False, True, "1. a 2. b 3. c 4. d", strict=False),
    ]
    stats = variants.pooled_stats(rows)
    assert stats["followed_injection"]["rate"] == pytest.approx(1 / 3)
    assert stats["followed_injection"]["n"] == 3
    # The figure shows strict task correctness; the lenient rate is kept alongside.
    assert stats["task_correct_no_injection"]["rate"] == pytest.approx(1 / 3)
    assert stats["task_correct_no_injection_lenient"]["rate"] == pytest.approx(2 / 3)
    assert stats["task_correct_no_injection"]["n"] == 3
    # word counts 1, 4, 1, 4, 1, 8 -> median 2.5
    assert stats["median_words"] == 2.5
    low, high = (
        stats["followed_injection"]["ci_low"],
        stats["followed_injection"]["ci_high"],
    )
    assert low <= 1 / 3 <= high


def test_variant_list_matches_the_queue() -> None:
    assert variants.VARIANTS == TRACKB_VARIANTS
    assert variants.KTO_RUNS == KTO_RUNS
    assert variants.GRPO_RUNS == GRPO_RUNS
    assert all(run in variants.LABELS and run in variants.LEGEND for run in GRPO_RUNS)
    # Colour follows the model: every run has a fixed slot and at most 8 slots exist.
    assert len(variants.ENTITIES) == len(set(variants.ENTITIES)) <= 8


def _write_eval(directory: Path, label: str, fingerprint: str, rows: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in DEFAULT_SPLITS:
        name = f"{split}_{label}.jsonl"
        (directory / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        files[split] = name
    summary = directory / f"summary_{label}.json"
    summary.write_text(json.dumps({"files": files, "examples_sha256": fingerprint}))
    return summary


def test_variants_that_scored_other_prompts_or_did_not_run_are_left_out(tmp_path: Path) -> None:
    rows = [row(True, False, True, "a b"), row(False, False, True, "c")]
    _write_eval(tmp_path / "trackb_full" / "trackb_eval", "base", "same", rows)
    _write_eval(tmp_path / "trackb_full" / "trackb_eval", "trained", "same", rows)
    _write_eval(tmp_path / "trackb_beta0.5" / "trackb_eval", "trained", "other", rows)
    entities, notes = variants.collect(tmp_path, TRACKB_VARIANTS)
    assert set(entities) == {"base", "trackb_full"}
    assert any("trackb_beta0.5" in n and "different prompts" in n for n in notes)
    assert any("trackb_steps60" in n and "not run" in n for n in notes)
    assert any("trackb_full" in n and "general regression" in n for n in notes)
    assert entities["base"]["nll_change_per_token"]["difference"] == 0.0


ROOT = Path(__file__).resolve().parents[1]
REAL_EVAL = ROOT / "results" / "trackb_full" / "trackb_eval"


@pytest.mark.skipif(
    not (REAL_EVAL / "summary_trained.json").is_file(), reason="needs results/trackb_full"
)
def test_comparison_and_examples_land_next_to_the_trained_run(tmp_path: Path) -> None:
    # A follow-up variant is compared with trackb_full's base evaluation. Its table and
    # examples must be written next to the variant, never over trackb_full's own files.
    import shutil
    import subprocess
    import sys

    base_dir = tmp_path / "trackb_full" / "trackb_eval"
    variant_dir = tmp_path / "trackb_variant" / "trackb_eval"
    for label, target in (("base", base_dir), ("trained", variant_dir)):
        target.mkdir(parents=True)
        summary = json.loads((REAL_EVAL / f"summary_{label}.json").read_text())
        for name in summary["files"].values():
            shutil.copy(REAL_EVAL / name, target / name)
        shutil.copy(REAL_EVAL / f"summary_{label}.json", target)
    args = [
        "--base",
        str(base_dir / "summary_base.json"),
        "--trained",
        str(variant_dir / "summary_trained.json"),
    ]
    for script, extra in (("compare_trackb.py", ["--resamples", "50"]), ("make_examples.py", [])):
        done = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), *args, *extra],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr[-800:]
    for name in ("trackb_table.md", "trackb_compare.json", "examples.md"):
        assert (tmp_path / "trackb_variant" / name).is_file()
        assert not (tmp_path / "trackb_full" / name).exists()
