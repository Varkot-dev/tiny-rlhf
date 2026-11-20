"""Tests for scripts/plot_hero.py: the README's opening figure reads only saved results."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
spec = importlib.util.spec_from_file_location("plot_hero", ROOT / "scripts/plot_hero.py")
hero = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hero)

needs_results = pytest.mark.skipif(
    not (RESULTS / "seeds_summary.json").exists()
    or not (RESULTS / "trackb_grpo" / "general_regression.json").exists(),
    reason="saved results not present",
)


@needs_results
def test_every_point_comes_from_the_saved_files() -> None:
    values = hero.collect(RESULTS)
    seeds = json.loads((RESULTS / "seeds_summary.json").read_text())
    points = values["points"]
    # Four seeded groups of three, plus the one GRPO run.
    assert len(points) == 13
    assert [p["method"] for p in points].count("GRPO") == 1
    for group, method, beta in hero.SEED_GROUPS:
        mine = [p for p in points if p["method"] == method and p["beta"] == beta]
        assert len(mine) == 3
        for p in mine:
            saved = seeds["groups"][group]["seeds"][str(p["seed"])]
            assert p["on_task"] == saved["on_task_strict"]
            assert p["drift"] == saved["nll_change_per_token"]
    assert values["base"]["on_task"] == seeds["base"]["on_task_strict"]
    general = json.loads((RESULTS / "trackb_grpo" / "general_regression.json").read_text())
    (grpo,) = (p for p in points if p["method"] == "GRPO")
    assert grpo["drift"] == general["nll_per_token_change"]


@needs_results
def test_figure_is_drawn(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")  # CI does not install matplotlib
    out = tmp_path / "hero.png"
    hero.plot(hero.collect(RESULTS), out)
    assert out.stat().st_size > 20_000
