"""Tests for the M8 scripts that read finished runs: the GRPO training figure and the
reward-hacking check. Offline, on small hand-made run directories."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tiny_rlhf.trackb.evaluate import DEFAULT_SPLITS, score_row
from tiny_rlhf.train_grpo import TRAIN_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "trackb"


def load_script(name: str) -> Any:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


training = load_script("plot_grpo_training")
hacking = load_script("check_grpo_hacking")


# --------------------------------------------------------------------------------------
# Training figure
# --------------------------------------------------------------------------------------


def test_moving_average_skips_missing_values() -> None:
    assert training.moving_average([1.0, None, 3.0, 5.0], window=2) == [1.0, 1.0, 3.0, 4.0]
    assert training.moving_average([None, None], window=3) == [None, None]


def write_run(directory: Path, steps: int) -> None:
    directory.mkdir(parents=True)
    with (directory / "train.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAIN_COLUMNS)
        writer.writeheader()
        for step in range(1, steps + 1):
            row = dict.fromkeys(TRAIN_COLUMNS, 0.5)
            row.update(step=step, reward_mean=step / steps, kl=0.01 * step, rate_refused="")
            writer.writerow(row)
    info = {
        "optimizer_steps": steps,
        "stopped_by_time_limit": False,
        "prompts_per_step": 4,
        "trl_settings": {"num_generations": 8, "beta": 0.04},
    }
    (directory / "run_info.json").write_text(json.dumps(info))


def test_training_figure_and_summary(tmp_path: Path) -> None:
    run = tmp_path / "trackb_grpo"
    write_run(run, steps=12)
    out, summary = tmp_path / "fig.png", tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "plot_grpo_training.py"),
            "--run",
            str(run),
            "--out",
            str(out),
            "--json-out",
            str(summary),
            "--window",
            "4",
        ],
        check=True,
        capture_output=True,
    )
    assert out.stat().st_size > 10_000
    data = json.loads(summary.read_text())
    reward = data["columns"]["reward_mean"]
    assert reward["first"] == pytest.approx((1 + 2 + 3 + 4) / 4 / 12)
    assert reward["last"] == pytest.approx((9 + 10 + 11 + 12) / 4 / 12)
    assert data["columns"]["rate_refused"] == {"first": None, "last": None}


# --------------------------------------------------------------------------------------
# Reward-hacking check
# --------------------------------------------------------------------------------------


def examples(split: str, n: int) -> list[dict[str, Any]]:
    lines = (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()[:n]
    return [json.loads(line) for line in lines]


def write_eval(directory: Path, label: str, respond) -> None:
    """An evaluation of the five test splits' first 4 examples, answered by `respond`."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in DEFAULT_SPLITS:
        rows = [score_row(e, respond(e)) for e in examples(split, 4)]
        files[split] = f"{split}_{label}.jsonl"
        (directory / files[split]).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (directory / f"summary_{label}.json").write_text(
        json.dumps({"files": files, "examples_sha256": "same"})
    )


def preferred(example: dict[str, Any]) -> str:
    return example["chosen"][-1]["content"]


def copy_document(example: dict[str, Any]) -> str:
    return example["chosen"][0]["content"]


def test_hacking_check_counts_copies_and_writes_a_sample(tmp_path: Path) -> None:
    write_eval(tmp_path / "trackb_full" / "trackb_eval", "base", preferred)
    write_eval(tmp_path / "trackb_grpo" / "trackb_eval", "trained", copy_document)
    out = tmp_path / "hacking.png"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_grpo_hacking.py"),
            "--results",
            str(tmp_path),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "not evaluated yet" in result.stdout  # the DPO and KTO runs are missing here
    report = json.loads((tmp_path / "trackb_grpo" / "reward_hacking.json").read_text())
    base, grpo = report["models"]["base"], report["models"]["trackb_grpo"]
    assert base["n_answers"] == grpo["n_answers"] == 20
    # Every preferred answer is paid and none is a copy; every copy is a copy and unpaid.
    assert base["paid"]["rate"] == 1.0 and base["echoed_document"]["rate"] == 0.0
    assert grpo["echoed_document"]["rate"] == 1.0 and grpo["paid"]["rate"] == 0.0
    assert report["human_judged_wrong_rate"] is None
    assert out.stat().st_size > 10_000
    sample = (tmp_path / "trackb_grpo" / "reward_hacking_sample.md").read_text()
    assert "0 of the 0 answers" in sample  # nothing paid, nothing to judge


def test_hedging_is_counted_only_for_tasks_with_several_candidates() -> None:
    from tiny_rlhf.trackb.checks import answer_candidates

    item = next(e for e in examples("test_in_dist", 150) if e["task_category"] == "cheapest_item")
    names = answer_candidates(item)
    assert len(names) >= 3
    row = score_row(item, f"Maybe the {names[0]} or the {names[1]}.")
    assert hacking.candidates_named(row) == 2
    city = next(e for e in examples("test_in_dist", 150) if e["task_category"] == "extract_city")
    assert hacking.candidates_named(score_row(city, "Leeds")) is None
