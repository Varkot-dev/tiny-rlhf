"""Tests for the seeds stage: the seed configs, and the numbers of the across-seeds figure."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import yaml

from tiny_rlhf.gpu_queue import SEED_RUNS, seed_run_name

ROOT = Path(__file__).resolve().parents[1]


def _load(script: str):
    path = ROOT / "scripts" / script
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seeds_plot = _load("plot_seeds.py")
strict_plot = _load("plot_trackb_strict.py")


# --------------------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("run", "seed"), SEED_RUNS)
def test_seed_config_differs_from_its_original_only_in_seed_and_run_name(
    run: str, seed: int
) -> None:
    original = yaml.safe_load((ROOT / f"configs/{run}.yaml").read_text())
    variant = yaml.safe_load((ROOT / f"configs/seeds/{seed_run_name(run, seed)}.yaml").read_text())
    differ = {k for k in original.keys() | variant.keys() if original.get(k) != variant.get(k)}
    assert differ == {"seed", "run_name"}
    assert original["seed"] == 0 and variant["seed"] == seed
    assert variant["run_name"] == seed_run_name(run, seed)


def test_every_seed_config_belongs_to_the_queue() -> None:
    on_disk = {p.stem for p in (ROOT / "configs/seeds").glob("*.yaml")}
    assert on_disk == {seed_run_name(run, seed) for run, seed in SEED_RUNS}


def test_whole_training_split_is_used_so_seeds_change_order_not_pairs() -> None:
    # The config headers say the seed changes only the order of the training pairs. That
    # holds because n_train is the whole of train.jsonl (and n_eval the whole of val.jsonl).
    n_train = sum(1 for _ in (ROOT / "data/trackb/train.jsonl").open())
    n_val = sum(1 for _ in (ROOT / "data/trackb/val.jsonl").open())
    for run, _ in SEED_RUNS:
        cfg = yaml.safe_load((ROOT / f"configs/{run}.yaml").read_text())
        assert cfg["n_train"] == n_train and cfg["n_eval"] == n_val


# --------------------------------------------------------------------------------------
# Figure numbers
# --------------------------------------------------------------------------------------


def test_spread_of_per_seed_values() -> None:
    assert seeds_plot.spread([0.2, 0.4, 0.9]) == {
        "mean": pytest.approx(0.5),
        "min": 0.2,
        "max": 0.9,
        "n_seeds": 3,
    }
    assert seeds_plot.spread([]) == {"mean": None, "min": None, "max": None, "n_seeds": 0}


def _copy_run(src: str, dst: str, results: Path) -> None:
    (results / dst).mkdir(parents=True)
    shutil.copytree(ROOT / "results" / src / "trackb_eval", results / dst / "trackb_eval")
    shutil.copy(ROOT / "results" / src / "general_regression.json", results / dst)


def test_collect_reuses_the_strict_rescoring_and_names_what_is_missing(tmp_path: Path) -> None:
    if not (ROOT / "results/trackb_kto/trackb_eval/summary_trained.json").is_file():
        pytest.skip("needs the saved Track B evaluations")
    results = tmp_path / "results"
    for run in ("trackb_full", "trackb_kto"):
        _copy_run(run, run, results)
    # A stand-in for a second seed of KTO beta 0.1, and one that scored other prompts.
    _copy_run("trackb_kto_beta0.5", "trackb_kto_seed1", results)
    _copy_run("trackb_beta0.5", "trackb_beta0.5", results)
    summary = results / "trackb_beta0.5" / "trackb_eval" / "summary_trained.json"
    data = json.loads(summary.read_text())
    data["examples_sha256"] = "something else"
    summary.write_text(json.dumps(data))

    out = seeds_plot.collect(results, n_resamples=50)
    kto = out["groups"]["trackb_kto"]
    assert set(kto["seeds"]) == {"0", "1"}
    # Seed 0 is the original run and must equal the strict rescoring saved in the repo.
    saved = json.loads((ROOT / "results/trackb_kto/trackb_eval/rescored_trained.json").read_text())[
        "pooled_test_splits"
    ]["groups"]
    for key, group in (
        ("on_task_strict", "injected"),
        ("followed_injection", "injected"),
        ("task_correct_strict", "no_injection"),
    ):
        assert kto["seeds"]["0"][key]["rate"] == pytest.approx(saved[group]["metrics"][key]["rate"])
    values = [kto["seeds"][s]["on_task_strict"]["rate"] for s in ("0", "1")]
    assert kto["on_task_strict"]["mean"] == pytest.approx(sum(values) / 2)
    assert kto["on_task_strict"]["min"] == min(values)
    assert kto["nll_change_per_token"]["n_seeds"] == 2
    assert out["groups"]["trackb_beta0.5"]["seeds"] == {}
    assert "trackb_beta0.5: scored different prompts from the base" in out["missing"]
    assert "trackb_kto_seed2: not run yet" in out["missing"]
    assert out["base"]["on_task_strict"]["n"] == 343


def test_strict_all_runs_figure_leaves_seed_repeats_out(tmp_path: Path) -> None:
    results = tmp_path / "results"
    for run in ("trackb_full", "trackb_kto", "trackb_kto_seed1"):
        (results / run / "trackb_eval").mkdir(parents=True)
        (results / run / "trackb_eval" / "rescored_trained.json").write_text("{}")
    (results / "trackb_full" / "trackb_eval" / "rescored_base.json").write_text("{}")
    compare = {"runs": {"trackb_full": {}, "trackb_kto": {}, "trackb_kto_seed1": {}}}
    (results / "trackb_rescored_compare.json").write_text(json.dumps(compare))
    assert list(strict_plot.load_entities(results)) == ["base", "trackb_full", "trackb_kto"]
