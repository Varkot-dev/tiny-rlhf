"""Tests for the sweep runner: parsing, run names, skipping, failure isolation, summaries."""

from __future__ import annotations

import csv
import json
import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tiny_rlhf.sweeps import (
    format_label,
    index_path,
    load_sweep,
    run_is_complete,
    run_summary,
    run_sweep,
    sweep_summary,
)
from tiny_rlhf.train_dpo import load_config

REPO = Path(__file__).resolve().parents[1]
SHIPPED = sorted((REPO / "configs" / "sweeps").glob("*.yaml"))

# A stand-in for scripts/train_dpo.py: writes results/<run>/run_info.json like a real run,
# unless told to fail (exit code 3) or to exit 0 without writing anything.
STUB = """
import json, pathlib, sys
args = sys.argv[1:]
name = args[args.index("--run-name") + 1]
if "--set" in args and 'mode="fail"' in args:
    sys.exit(3)
if "--set" in args and 'mode="silent"' in args:
    sys.exit(0)
out = pathlib.Path("results") / name
out.mkdir(parents=True, exist_ok=True)
(out / "run_info.json").write_text(json.dumps({"args": args}))
"""


def write_sweep(tmp_path: Path, body: str) -> Path:
    (tmp_path / "stub.py").write_text(STUB)
    path = tmp_path / "sweep.yaml"
    path.write_text(textwrap.dedent(body))
    return path


BASIC = """
    name: demo
    base_config: configs/base.yaml
    field: beta
    script: stub.py
    run_name: "demo_{label}"
    overrides: {n_train: 16}
    values: [0.01, 0.1, 0.5]
"""


def test_format_label() -> None:
    assert format_label(0.01) == "0.01"
    assert format_label(16) == "16"
    assert format_label(True) == "true"
    assert format_label("abc") == "abc"


def test_load_sweep_expands_values_into_runs(tmp_path: Path) -> None:
    sweep = load_sweep(write_sweep(tmp_path, BASIC))
    assert sweep.name == "demo" and sweep.field == "beta"
    assert [r.run_name for r in sweep.runs] == ["demo_0.01", "demo_0.1", "demo_0.5"]
    assert [r.value for r in sweep.runs] == [0.01, 0.1, 0.5]
    assert sweep.runs[1].command == (
        "stub.py",
        "--config",
        "configs/base.yaml",
        "--run-name",
        "demo_0.1",
        "--set",
        "n_train=16",
        "--set",
        "beta=0.1",
    )


def test_entry_overrides_labels_flags_and_extra_args(tmp_path: Path) -> None:
    sweep = load_sweep(
        write_sweep(
            tmp_path,
            """
            name: noise
            base_config: configs/base.yaml
            field: noise_fraction
            flag: --noise-fraction
            script: stub.py
            overrides: {n_train: 16}
            values:
              - 0.1
              - value: 0.3
                label: "0.3_robust"
                series: robust
                overrides: {n_train: 8, lora_target_modules: [q_proj, v_proj]}
                extra_args: ["--label-smoothing", 0.3]
            """,
        )
    )
    plain, robust = sweep.runs
    assert plain.run_name == "sweep_noise_0.1"  # the default pattern
    assert plain.command[-4:] == ("--noise-fraction", "0.1", "--set", "n_train=16")
    assert robust.run_name == "sweep_noise_0.3_robust" and robust.series == "robust"
    # The per-entry override wins, the swept value goes through the flag and not --set,
    # and a list override is written in a form the training script's YAML parser reads.
    assert robust.command[5:] == (
        "--noise-fraction",
        "0.3",
        "--set",
        "n_train=8",
        "--set",
        'lora_target_modules=["q_proj", "v_proj"]',
        "--label-smoothing",
        "0.3",
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("name: x\nbase_config: c.yaml\nfield: beta\n", "values"),
        (
            "name: x\nbase_config: c.yaml\nfield: beta\nvalues: [1]\nbogus: 1\n",
            "unknown sweep keys",
        ),
        ("name: x\nbase_config: c.yaml\nfield: beta\nvalues: [{label: a}]\n", "bad sweep entry"),
        ("name: x\nbase_config: c.yaml\nfield: beta\nvalues: [1, 1]\n", "duplicate run names"),
        ("name: x\nbase_config: c.yaml\nfield: beta\nvalues: [1]\nx_scale: cubic\n", "x_scale"),
    ],
)
def test_load_sweep_rejects_bad_files(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(body)
    with pytest.raises(ValueError, match=message):
        load_sweep(path)


def test_run_sweep_runs_everything_and_writes_the_index(tmp_path: Path) -> None:
    sweep = load_sweep(write_sweep(tmp_path, BASIC))
    index = run_sweep(sweep, root=tmp_path, python=sys.executable)
    assert [r["status"] for r in index["runs"]] == ["done", "done", "done"]
    assert all(run_is_complete(r.run_name, tmp_path) for r in sweep.runs)
    on_disk = json.loads(index_path(sweep, tmp_path).read_text())
    assert on_disk == index
    assert on_disk["field"] == "beta"
    assert [r["value"] for r in on_disk["runs"]] == [0.01, 0.1, 0.5]
    # The stub recorded the arguments it was given: the swept value reached the script.
    args = json.loads((tmp_path / "results" / "demo_0.5" / "run_info.json").read_text())["args"]
    assert "beta=0.5" in args and "n_train=16" in args


def test_skip_existing_does_not_rerun_finished_runs(tmp_path: Path) -> None:
    sweep = load_sweep(write_sweep(tmp_path, BASIC))
    done = tmp_path / "results" / "demo_0.1"
    done.mkdir(parents=True)
    (done / "run_info.json").write_text('{"marker": "from an earlier session"}')
    calls: list[list[str]] = []

    def runner(command: list[str], cwd: Path):
        calls.append(command)
        import subprocess

        return subprocess.run(command, cwd=cwd)

    index = run_sweep(sweep, root=tmp_path, skip_existing=True, runner=runner)
    assert [r["status"] for r in index["runs"]] == ["done", "skipped_existing", "done"]
    assert len(calls) == 2
    assert json.loads((done / "run_info.json").read_text()) == {"marker": "from an earlier session"}


def test_without_skip_existing_a_finished_run_is_rerun(tmp_path: Path) -> None:
    sweep = load_sweep(write_sweep(tmp_path, BASIC))
    done = tmp_path / "results" / "demo_0.1"
    done.mkdir(parents=True)
    (done / "run_info.json").write_text('{"marker": "old"}')
    run_sweep(sweep, root=tmp_path)
    assert "args" in json.loads((done / "run_info.json").read_text())


def test_a_failed_run_is_recorded_and_the_sweep_continues(tmp_path: Path) -> None:
    path = write_sweep(
        tmp_path,
        """
        name: demo
        base_config: configs/base.yaml
        field: mode
        script: stub.py
        run_name: "demo_{label}"
        values: [ok, fail, silent, alsook]
        """,
    )
    index = run_sweep(load_sweep(path), root=tmp_path)
    by_label = {r["label"]: r for r in index["runs"]}
    assert by_label["ok"]["status"] == "done" and by_label["alsook"]["status"] == "done"
    assert by_label["fail"]["status"] == "failed" and by_label["fail"]["returncode"] == 3
    # Exit code 0 without the final file is still a failure.
    assert by_label["silent"]["status"] == "failed" and by_label["silent"]["returncode"] == 0


def test_a_missing_script_is_a_failed_run_not_a_crash(tmp_path: Path) -> None:
    path = write_sweep(tmp_path, BASIC.replace("script: stub.py", "script: no_such_script.py"))
    index = run_sweep(load_sweep(path), root=tmp_path)
    assert [r["status"] for r in index["runs"]] == ["failed"] * 3


def test_only_runs_the_named_entries_and_keeps_earlier_records(tmp_path: Path) -> None:
    sweep = load_sweep(write_sweep(tmp_path, BASIC))
    first = run_sweep(sweep, root=tmp_path, only=["0.1"])
    assert [r["status"] for r in first["runs"]] == ["not_run", "done", "not_run"]
    second = run_sweep(sweep, root=tmp_path, only=["demo_0.5"])  # by run name this time
    assert [r["status"] for r in second["runs"]] == ["not_run", "done", "done"]
    assert second["runs"][1]["seconds"] is not None  # carried over from the first call
    with pytest.raises(ValueError, match="unknown entries"):
        run_sweep(sweep, root=tmp_path, only=["0.7"])


def test_dry_run_changes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sweep = load_sweep(write_sweep(tmp_path, BASIC))
    run_sweep(sweep, root=tmp_path, dry_run=True)
    assert not (tmp_path / "results").exists()
    printed = capsys.readouterr().out.strip().splitlines()
    assert len(printed) == 3 and printed[0].startswith("python stub.py --config")


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.stem)
def test_shipped_sweeps_load_and_their_overrides_are_valid_config(path: Path) -> None:
    sweep = load_sweep(path)
    assert (REPO / sweep.base_config).exists()
    assert sweep.runs
    for run in sweep.runs:
        assert (REPO / run.command[0]).exists(), run.command[0]
        # Every --set KEY=VALUE must be accepted by the real config loader.
        sets = [run.command[i + 1] for i, a in enumerate(run.command) if a == "--set"]
        overrides = {}
        import yaml

        for item in sets:
            key, _, value = item.partition("=")
            overrides[key] = yaml.safe_load(value)
        cfg = load_config(REPO / sweep.base_config, overrides)
        assert cfg.n_eval == 300  # every shipped sweep scores the same held-out pairs


def test_shipped_sweep_names() -> None:
    assert {p.stem for p in SHIPPED} == {"beta", "data_size", "label_noise", "lora_rank"}


def test_lora_rank_sweep_keeps_alpha_at_twice_the_rank() -> None:
    sweep = load_sweep(REPO / "configs" / "sweeps" / "lora_rank.yaml")
    for run in sweep.runs:
        assert f"lora_alpha={2 * run.value}" in run.command
        assert f"lora_r={run.value}" in run.command


def test_data_size_sweep_holds_optimizer_steps_fixed() -> None:
    sweep = load_sweep(REPO / "configs" / "sweeps" / "data_size.yaml")
    for run in sweep.runs:
        assert "max_steps=340" in run.command
        epochs = next(int(a.split("=")[1]) for a in run.command if a.startswith("epochs="))
        # Even if a quarter of the pairs were dropped for over-long prompts, the run
        # still has enough optimizer steps (8 pairs each) to reach the cap.
        steps_available = (int(run.value * 0.75) // 8) * epochs
        assert steps_available >= 340, (run.value, epochs, steps_available)


# --------------------------------------------------------------------------------------
# summaries used by the figures
# --------------------------------------------------------------------------------------


def write_pairs(run_dir: Path, rows: list[tuple[float, float, float, float, float]]) -> None:
    run_dir.mkdir(parents=True)
    with (run_dir / "eval_pairs_final.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pair_index",
                "chosen_logp",
                "rejected_logp",
                "ref_chosen_logp",
                "ref_rejected_logp",
                "chosen_reward",
                "rejected_reward",
                "reward_margin",
            ]
        )
        for i, (c, r, rc, rr, margin) in enumerate(rows):
            writer.writerow([i, c, r, rc, rr, 0, 0, margin])


def test_run_summary_matches_hand_computed_numbers(tmp_path: Path) -> None:
    # Four pairs: margins +0.2, +0.1, 0 (a tie counts as wrong), -0.1.
    write_pairs(
        tmp_path / "run",
        [
            (-10.0, -20.0, -11.0, -18.0, 0.2),
            (-10.0, -20.0, -10.5, -19.0, 0.1),
            (-10.0, -20.0, -10.0, -20.0, 0.0),
            (-12.0, -20.0, -10.0, -21.0, -0.1),
        ],
    )
    (tmp_path / "run" / "run_info.json").write_text(json.dumps({"optimizer_steps": 7}))
    summary = run_summary(tmp_path / "run", n_resamples=200, seed=0)
    assert summary["n_pairs"] == 4
    assert summary["accuracy"]["mean"] == 0.5
    assert summary["reward_margin"]["mean"] == pytest.approx(0.05)
    assert summary["chosen_logp_change"]["mean"] == pytest.approx((1 + 0.5 + 0 - 2) / 4)
    assert summary["rejected_logp_change"]["mean"] == pytest.approx((-2 - 1 + 0 + 1) / 4)
    for key in ("accuracy", "reward_margin", "chosen_logp_change"):
        assert summary[key]["ci_low"] <= summary[key]["mean"] <= summary[key]["ci_high"]
    assert summary["optimizer_steps"] == 7
    assert run_summary(tmp_path / "run", 200, 0) == summary  # seeded, so reproducible


def test_sweep_summary_lists_unfinished_runs_as_missing(tmp_path: Path) -> None:
    sweep = load_sweep(write_sweep(tmp_path, BASIC))
    write_pairs(tmp_path / "results" / "demo_0.1", [(-1.0, -2.0, -1.5, -1.5, 0.3)])
    (tmp_path / "results" / "demo_0.1" / "drift.json").write_text('{"kl": 0.01}')
    # Pair rows without run_info.json are a run that did not finish (or a crashed rerun).
    assert sweep_summary(sweep, tmp_path, n_resamples=50)["missing_runs"] == [
        "demo_0.01",
        "demo_0.1",
        "demo_0.5",
    ]
    (tmp_path / "results" / "demo_0.1" / "run_info.json").write_text("{}")
    summary = sweep_summary(sweep, tmp_path, n_resamples=50)
    assert summary["missing_runs"] == ["demo_0.01", "demo_0.5"]
    (only,) = summary["runs"]
    assert only["label"] == "0.1" and only["value"] == 0.1
    assert only["run_dir"] == "results/demo_0.1"  # relative, so summaries are portable
    assert only["drift"] == {"kl": 0.01}


def test_a_rerun_that_writes_nothing_is_failed_even_if_an_old_run_info_is_there(
    tmp_path: Path,
) -> None:
    # The stub exits 0 without writing anything, like a run that died quietly. A
    # run_info.json left by an earlier session must not make it count as done.
    path = write_sweep(
        tmp_path,
        """
        name: demo
        base_config: configs/base.yaml
        field: mode
        script: stub.py
        run_name: "demo_{label}"
        values: [silent]
        """,
    )
    old = tmp_path / "results" / "demo_silent"
    old.mkdir(parents=True)
    (old / "run_info.json").write_text('{"from": "an earlier session"}')
    two_hours_ago = time.time() - 7200
    os.utime(old / "run_info.json", (two_hours_ago, two_hours_ago))

    assert run_is_complete("demo_silent", tmp_path)
    assert not run_is_complete("demo_silent", tmp_path, written_after=time.time())
    index = run_sweep(load_sweep(path), root=tmp_path)
    assert [r["status"] for r in index["runs"]] == ["failed"]
