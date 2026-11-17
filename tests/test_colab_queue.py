"""Tests for the GPU job queue. Offline: jobs are tiny stub commands in a temporary git repo."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tiny_rlhf.gpu_queue import (
    FAILURE_DIR,
    KTO_RUNS,
    KTO_SMALL_BATCH,
    STAGES,
    STATUS_FILE,
    TAIL_LINES,
    TRACKB_BASE_SUMMARY,
    TRACKB_EVAL_ARGS,
    TRACKB_VARIANTS,
    Job,
    build_queue,
    describe,
    format_table,
    run_queue,
    run_streaming,
)
from tiny_rlhf.kto import KTOConfig
from tiny_rlhf.reward_model import RewardConfig
from tiny_rlhf.sweeps import load_sweep
from tiny_rlhf.train_dpo import DPOConfig

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.name", "test")
    git(path, "config", "user.email", "test@example.com")
    (path / "README.md").write_text("x\n")
    git(path, "add", "README.md")
    git(path, "commit", "-q", "-m", "init")
    return path


def stub(code: str) -> tuple[str, ...]:
    return (PY, "-c", code)


def writes(path: str, text: str = "ok") -> tuple[str, ...]:
    return stub(
        f"import pathlib; p = pathlib.Path({path!r}); p.parent.mkdir(parents=True, exist_ok=True); "
        f"p.write_text({text!r}); print('wrote', p)"
    )


def job(name: str, marker: str, *commands: tuple[str, ...], **kwargs: object) -> Job:
    kwargs.setdefault("outputs", (marker,))
    kwargs.setdefault("estimate_min", 1)
    stage = kwargs.pop("stage", "s")
    return Job(name=name, stage=stage, commands=commands, marker=marker, **kwargs)


def run(repo: Path, jobs: list[Job], **kwargs: object) -> dict:
    kwargs.setdefault("do_push", False)
    kwargs.setdefault("gpu", "Test GPU")
    kwargs.setdefault("echo", lambda text: None)
    return run_queue(jobs, root=repo, **kwargs)


def committed(repo: Path) -> set[str]:
    return set(git(repo, "ls-files").split())


# --------------------------------------------------------------------------------------
# The real queue, as data
# --------------------------------------------------------------------------------------

QUEUE = build_queue(ROOT)
BY_NAME = {j.name: j for j in QUEUE}


def test_queue_has_every_expected_job_exactly_once_in_priority_order() -> None:
    names = [j.name for j in QUEUE]
    assert len(names) == len(set(names))
    expected = {
        "m5_eval_base",
        "m5_train",
        "m5_eval_trained",
        "m5_compare",
        "m5_followup_figure",
        "m5_rescore_strict",
        "m3_reward",
        "m3_reward_vs_dpo",
        "m2_parity_ours",
        "m2_parity_trl",
        "m2_parity_figure",
        "m2_parity_ours_fp16",
        "m2_parity_trl_fp16",
        "m2_parity_figure_fp16",
        "beta_figure",
        "lora_rank_figure",
        "data_size_figure",
        "label_noise_figure",
    }
    for stage in ("beta", "lora_rank", "data_size", "label_noise"):
        sweep = load_sweep(ROOT / f"configs/sweeps/{stage}.yaml")
        expected |= {f"{stage}_{run.label}" for run in sweep.runs}
    for run in (*TRACKB_VARIANTS, *KTO_RUNS):
        expected |= {f"{run}_train", f"{run}_eval", f"{run}_compare"}
    expected.add("m7_figure")
    assert set(names) == expected
    assert [j.stage for j in QUEUE] == sorted((j.stage for j in QUEUE), key=STAGES.index)
    assert STAGES[:5] == ("m5", "m5_followup", "m7_kto", "m3", "m2")


def test_dependencies_exist_point_backwards_and_are_acyclic() -> None:
    position = {j.name: i for i, j in enumerate(QUEUE)}
    for j in QUEUE:
        for other in (*j.depends_on, *j.fallback_for):
            assert other in position, f"{j.name} refers to unknown job {other}"
            # Every reference points to an earlier job, so there can be no cycle.
            assert position[other] < position[j.name], f"{j.name} runs before {other}"


def test_every_job_has_an_estimate_and_specific_outputs() -> None:
    markers = [j.marker for j in QUEUE]
    assert len(markers) == len(set(markers))
    for j in QUEUE:
        assert j.estimate_min > 0
        assert j.marker.startswith("results/")
        for path in j.outputs:
            # Never a bare top-level directory: a failed job's outputs are discarded.
            assert path.startswith(("results/", "assets/")) and path.count("/") >= 1
            assert path.rstrip("/") not in ("results", "assets")


def test_every_script_config_and_sweep_file_exists() -> None:
    for j in QUEUE:
        for command in j.commands:
            if command[0] == "python" and command[1].endswith(".py"):
                assert (ROOT / command[1]).is_file(), f"{j.name}: {command[1]} is missing"
            for flag, value in zip(command, command[1:], strict=False):
                if flag in ("--config", "--sweep") and value.endswith(".yaml"):
                    assert (ROOT / value).is_file(), f"{j.name}: {value} is missing"
                if flag == "--sweep" and not value.endswith(".yaml"):
                    assert (ROOT / f"configs/sweeps/{value}.yaml").is_file()


CONFIG_FIELDS = {
    "scripts/train_reward.py": {f.name for f in dataclasses.fields(RewardConfig)},
    "scripts/train_dpo.py": {f.name for f in dataclasses.fields(DPOConfig)},
    "scripts/train_trl.py": {f.name for f in dataclasses.fields(DPOConfig)},
    "scripts/train_trackb.py": {f.name for f in dataclasses.fields(DPOConfig)},
    "scripts/train_kto.py": {f.name for f in dataclasses.fields(KTOConfig)},
}


def _set_keys(args: tuple[str, ...]) -> list[str]:
    return [v.partition("=")[0] for f, v in zip(args, args[1:], strict=False) if f == "--set"]


def test_every_set_override_names_a_real_config_field() -> None:
    seen = 0
    for j in QUEUE:
        for command in j.commands:
            keys = _set_keys(command) + (
                _set_keys(j.oom_retry_args) if "--config" in command else []
            )
            if not keys:
                continue
            assert command[1] in CONFIG_FIELDS, f"{j.name}: --set used with {command[1]}"
            for key in keys:
                assert key in CONFIG_FIELDS[command[1]], f"{j.name}: {key} is not a config field"
                seen += 1
    assert seen >= 9  # eval_every, the out-of-memory retries and the fp16 fallbacks


def test_m5_matches_the_one_hour_cell() -> None:
    cell = (ROOT / "docs/COLAB_ONE_HOUR.md").read_text()
    assert f'EVAL_ARGS = "{" ".join(TRACKB_EVAL_ARGS)}"' in cell
    train = BY_NAME["m5_train"].commands[0]
    assert "eval_every=125" in train and "configs/trackb_full.yaml" in train
    assert "--reuse-if-current" in BY_NAME["m5_eval_base"].commands[0]
    assert BY_NAME["m5_eval_base"].always_run
    assert BY_NAME["m5_train"].oom_retry_args and BY_NAME["m3_reward"].oom_retry_args


def test_followup_variants_are_evaluated_and_compared_exactly_like_m5() -> None:
    # The comparison refuses mismatched settings, and each variant is compared with the one
    # base evaluation in trackb_full, so the trained evaluations must use m5's settings.
    m5_eval = BY_NAME["m5_eval_trained"].commands[0]
    assert BY_NAME["m5_eval_base"].marker == TRACKB_BASE_SUMMARY
    for run in TRACKB_VARIANTS:
        assert (ROOT / f"configs/{run}.yaml").is_file()
        evaluation = BY_NAME[f"{run}_eval"].commands[0]
        assert evaluation[evaluation.index("--with-probes") :] == TRACKB_EVAL_ARGS
        # Identical to the m5 command except the adapter and the output directory.
        differ = [i for i, (a, b) in enumerate(zip(evaluation, m5_eval, strict=True)) if a != b]
        assert [evaluation[i - 1] for i in differ] == ["--adapter", "--out"]
        assert evaluation[evaluation.index("--out") + 1] == f"results/{run}/trackb_eval"
        train = BY_NAME[f"{run}_train"]
        assert train.commands[0][-1] == f"configs/{run}.yaml"
        assert train.oom_retry_args == BY_NAME["m5_train"].oom_retry_args
        compare = BY_NAME[f"{run}_compare"]
        assert "m5_eval_base" in compare.depends_on
        for command in compare.commands[1:]:
            assert command[command.index("--base") + 1] == TRACKB_BASE_SUMMARY
            assert command[command.index("--trained") + 1] == (
                f"results/{run}/trackb_eval/summary_trained.json"
            )
        # Nothing a variant writes may land in (and overwrite) trackb_full's directory.
        assert all("trackb_full" not in path for path in compare.outputs + train.outputs)
    figure = BY_NAME["m5_followup_figure"]
    assert figure.min_deps_done == 1 and figure.always_run
    assert set(figure.depends_on) == {f"{run}_compare" for run in TRACKB_VARIANTS}


def test_kto_runs_are_evaluated_and_compared_exactly_like_the_dpo_runs() -> None:
    # M7 compares KTO with DPO on the same base evaluation, so every setting that the
    # comparison checks must be m5's, and nothing may overwrite trackb_full's files.
    m5_eval = BY_NAME["m5_eval_trained"].commands[0]
    for run in KTO_RUNS:
        assert (ROOT / f"configs/{run}.yaml").is_file()
        train = BY_NAME[f"{run}_train"]
        assert train.stage == "m7_kto"
        assert train.commands[0][:2] == ("python", "scripts/train_kto.py")
        assert train.commands[0][-1] == f"configs/{run}.yaml"
        # KTO needs at least 2 answers per micro-batch, so its retry cannot use batch 1.
        assert train.oom_retry_args == KTO_SMALL_BATCH
        assert "batch_size=2" in KTO_SMALL_BATCH and "batch_size=1" not in KTO_SMALL_BATCH
        evaluation = BY_NAME[f"{run}_eval"].commands[0]
        assert evaluation[evaluation.index("--with-probes") :] == TRACKB_EVAL_ARGS
        differ = [i for i, (a, b) in enumerate(zip(evaluation, m5_eval, strict=True)) if a != b]
        assert [evaluation[i - 1] for i in differ] == ["--adapter", "--out"]
        assert evaluation[evaluation.index("--out") + 1] == f"results/{run}/trackb_eval"
        compare = BY_NAME[f"{run}_compare"]
        assert "m5_eval_base" in compare.depends_on
        for command in compare.commands[1:4]:
            assert command[command.index("--base") + 1] == TRACKB_BASE_SUMMARY
        scripts = [c[1] for c in compare.commands]
        assert scripts[-2:] == ["scripts/rescore_trackb.py", "scripts/plot_trackb_strict.py"]
        assert f"results/{run}/trackb_eval/rescored_trained.json" in compare.outputs
        assert all("trackb_full" not in path for path in compare.outputs + train.outputs)
    figure = BY_NAME["m7_figure"]
    assert figure.min_deps_done == 1 and figure.always_run
    assert set(figure.depends_on) == {f"{run}_compare" for run in KTO_RUNS}
    variants_cmd = figure.commands[0]
    runs = variants_cmd[variants_cmd.index("--runs") + 1 : variants_cmd.index("--out")]
    assert tuple(runs) == ("trackb_beta0.5", *KTO_RUNS)
    assert figure.marker in figure.outputs


def test_fp16_fallback_is_wired_to_both_float32_parity_runs() -> None:
    for name, run_name in (
        ("m2_parity_ours_fp16", "parity_full_fp16"),
        ("m2_parity_trl_fp16", "parity_full_fp16_trl"),
    ):
        fallback = BY_NAME[name]
        assert fallback.fallback_for == ("m2_parity_ours", "m2_parity_trl")
        train = fallback.commands[-1]
        assert train[train.index("--run-name") + 1] == run_name
        assert train[train.index("--set") + 1] == "amp=true"
        assert fallback.marker == f"results/{run_name}/run_info.json"
    assert "--set" not in BY_NAME["m2_parity_ours"].commands[-1]
    assert "amp: false" in (ROOT / "configs/parity_full.yaml").read_text()
    assert BY_NAME["m2_parity_figure_fp16"].depends_on == ("m2_parity_trl_fp16",)


def test_beta_drift_runs_in_the_same_job_as_its_training() -> None:
    for j in QUEUE:
        if j.stage == "beta" and not j.name.endswith("figure"):
            scripts = [c[1] for c in j.commands]
            assert scripts == ["scripts/run_sweep.py", "scripts/eval_drift.py"]
            assert "--skip-existing" not in j.commands[0]
            assert j.marker.endswith("/drift.json")


def test_sweep_figures_tolerate_missing_values() -> None:
    for stage in ("beta", "lora_rank", "data_size", "label_noise"):
        fig = BY_NAME[f"{stage}_figure"]
        assert fig.min_deps_done == 1 and fig.always_run
        assert len(fig.depends_on) == len(load_sweep(ROOT / f"configs/sweeps/{stage}.yaml").runs)


def test_jobs_that_read_an_adapter_name_the_job_that_writes_it() -> None:
    produced = {j.produces_adapter: j.name for j in QUEUE if j.produces_adapter}
    readers = [j for j in QUEUE if j.needs_adapter]
    assert {j.name for j in readers} == {"m5_eval_trained", "m5_compare"} | {
        f"{run}_{step}" for run in (*TRACKB_VARIANTS, *KTO_RUNS) for step in ("eval", "compare")
    }
    for j in readers:
        producer = produced[j.needs_adapter]
        position = [q.name for q in QUEUE]
        assert position.index(producer) < position.index(j.name)


def test_reward_comparison_uses_the_full_dpo_run_in_the_repo() -> None:
    compare = BY_NAME["m3_reward_vs_dpo"].commands[0]
    assert compare[compare.index("--dpo") + 1] == "results/dpo_full_t4"
    assert (ROOT / "results/dpo_full_t4/eval_pairs_final.csv").is_file()


def test_describe_lists_every_job_with_its_estimate(tmp_path: Path) -> None:
    rows = describe(QUEUE, tmp_path)
    assert [r["job"] for r in rows] == [j.name for j in QUEUE]
    assert all(r["status"].startswith("pending") and r["estimate_min"] > 0 for r in rows)


# --------------------------------------------------------------------------------------
# The streaming runner
# --------------------------------------------------------------------------------------


def test_run_streaming_shows_lines_and_reports_the_exit_status(tmp_path: Path) -> None:
    seen: list[str] = []
    code, tail = run_streaming(
        stub("import sys; print('one'); print('two', file=sys.stderr); sys.exit(3)"),
        cwd=tmp_path,
        echo=seen.append,
    )
    assert code == 3
    assert [line.strip() for line in tail] == ["one", "two"]  # stderr is merged in
    assert seen == tail


def test_run_streaming_keeps_only_the_end_of_a_long_output(tmp_path: Path) -> None:
    code, tail = run_streaming(
        stub("[print(i) for i in range(1000)]"), cwd=tmp_path, echo=lambda text: None
    )
    assert code == 0 and len(tail) == TAIL_LINES and tail[-1] == "999\n"


def test_run_streaming_reports_a_program_that_cannot_start(tmp_path: Path) -> None:
    code, tail = run_streaming(
        ["this-program-does-not-exist-xyz"], cwd=tmp_path, echo=lambda text: None
    )
    assert code == 127 and "could not start" in tail[0]


# --------------------------------------------------------------------------------------
# The queue runner on stub jobs
# --------------------------------------------------------------------------------------


def test_dry_run_prints_commands_and_runs_nothing(repo: Path) -> None:
    out: list[str] = []
    jobs = [job("a", "results/a/done.json", writes("results/a/done.json"))]
    results = run(repo, jobs, dry_run=True, echo=out.append)
    text = "".join(out)
    assert results == {}
    assert "# a [s] pending" in text and "results/a/done.json" in text
    assert not (repo / "results").exists()
    assert git(repo, "status", "--porcelain") == ""


def test_successful_job_commits_only_its_outputs_and_the_status(repo: Path) -> None:
    write_both = stub(
        "import pathlib; pathlib.Path('results/a').mkdir(parents=True); "
        "pathlib.Path('results/a/done.json').write_text('ok'); "
        "pathlib.Path('stray.txt').write_text('not an output')"
    )
    jobs = [job("a", "results/a/done.json", write_both)]
    results = run(repo, jobs)
    assert results["a"]["status"] == "done"
    files = committed(repo)
    assert {"results/a/done.json", STATUS_FILE} <= files
    assert "stray.txt" not in files
    assert git(repo, "log", "-1", "--format=%s").strip() == "Add results for a (Colab Test GPU)"
    assert git(repo, "log", "-1", "--format=%an <%ae>").strip() == (
        "Varshith Kotagiri <Varkot-dev@users.noreply.github.com>"
    )
    assert "Co-Authored-By" not in git(repo, "log", "-1", "--format=%B")
    status = json.loads((repo / STATUS_FILE).read_text())
    assert status["jobs"]["a"]["status"] == "done" and status["gpu"] == "Test GPU"


def test_job_whose_marker_is_already_in_the_tree_is_skipped(repo: Path) -> None:
    (repo / "results/a").mkdir(parents=True)
    (repo / "results/a/done.json").write_text("old")
    jobs = [job("a", "results/a/done.json", writes("results/a/done.json", "new"))]
    results = run(repo, jobs)
    assert results["a"]["status"] == "skipped-existing"
    assert (repo / "results/a/done.json").read_text() == "old"  # the command did not run


def test_failure_is_logged_partials_discarded_and_only_dependents_skipped(repo: Path) -> None:
    fails = stub(
        "import pathlib, sys; d = pathlib.Path('results/bad'); d.mkdir(parents=True); "
        "(d / 'train.csv').write_text('half a run'); print('boom: something broke'); sys.exit(2)"
    )
    jobs = [
        job("bad", "results/bad/run_info.json", fails, outputs=("results/bad",)),
        job("child", "results/child/x.json", writes("results/child/x.json"), depends_on=("bad",)),
        job("other", "results/other/x.json", writes("results/other/x.json")),
    ]
    results = run(repo, jobs)
    assert results["bad"]["status"] == "failed"
    assert results["bad"]["exit_code"] == 2 and results["bad"]["oom"] is False
    assert any("boom" in line for line in results["bad"]["last_lines"])
    assert results["child"]["status"] == "skipped-dependency"
    assert results["child"]["blocked_by"] == "bad"
    assert results["other"]["status"] == "done"
    assert not (repo / "results/bad").exists()
    files = committed(repo)
    assert not any(f.startswith("results/bad") for f in files)
    assert "results/other/x.json" in files and "results/child/x.json" not in files
    log = f"{FAILURE_DIR}/bad.log"
    assert log in files and "boom: something broke" in (repo / log).read_text()
    assert git(repo, "status", "--porcelain") == ""
    status = json.loads((repo / STATUS_FILE).read_text())["jobs"]
    assert status["bad"]["status"] == "failed"
    assert "failed" in format_table(results)


def test_exit_status_zero_without_a_fresh_marker_is_a_failure(repo: Path) -> None:
    jobs = [job("a", "results/a/run_info.json", stub("print('did nothing')"))]
    results = run(repo, jobs)
    assert results["a"]["status"] == "failed"
    assert "did not write" in results["a"]["reason"]


def test_a_stale_marker_from_before_the_launch_is_not_a_success(repo: Path) -> None:
    # The job is forced to run (its adapter is gone) but crashes without touching the old
    # completion file: an old file must not make it look finished.
    (repo / "results/t").mkdir(parents=True)
    (repo / "results/t/run_info.json").write_text("old")
    os.utime(repo / "results/t/run_info.json", (1_000_000_000, 1_000_000_000))  # long ago
    git(repo, "add", "results")
    git(repo, "commit", "-q", "-m", "earlier run")
    jobs = [
        job(
            "t",
            "results/t/run_info.json",
            stub("print('exits 0 but trains nothing')"),
            outputs=("results/t",),
            produces_adapter="checkpoints/t",
        ),
        job(
            "use",
            "results/t/drift.json",
            writes("results/t/drift.json"),
            depends_on=("t",),
            needs_adapter="checkpoints/t",
        ),
    ]
    results = run(repo, jobs)
    assert results["t"]["status"] == "failed"
    assert results["use"]["status"] == "skipped-dependency"


def test_failed_rerun_restores_the_committed_copy(repo: Path) -> None:
    (repo / "results/t").mkdir(parents=True)
    (repo / "results/t/run_info.json").write_text("committed")
    git(repo, "add", "results")
    git(repo, "commit", "-q", "-m", "earlier run")
    crash = stub(
        "import pathlib, sys; d = pathlib.Path('results/t'); (d / 'run_info.json').unlink(); "
        "(d / 'train.csv').write_text('partial'); sys.exit(1)"
    )
    jobs = [
        job(
            "t",
            "results/t/run_info.json",
            crash,
            outputs=("results/t",),
            produces_adapter="checkpoints/t",
        ),
        job(
            "use",
            "results/t/drift.json",
            writes("results/t/drift.json"),
            depends_on=("t",),
            needs_adapter="checkpoints/t",
        ),
    ]
    results = run(repo, jobs)
    assert results["t"]["status"] == "failed"
    assert (repo / "results/t/run_info.json").read_text() == "committed"
    assert not (repo / "results/t/train.csv").exists()


def test_producer_is_run_again_when_its_adapter_is_missing(repo: Path) -> None:
    (repo / "results/t").mkdir(parents=True)
    (repo / "results/t/run_info.json").write_text("{}")
    train = stub(
        "import pathlib; pathlib.Path('results/t/run_info.json').write_text("
        '\'{"wall_time_s": 1.5, "scaler_retries": 2}\'); '
        "c = pathlib.Path('checkpoints/t'); c.mkdir(parents=True); "
        "(c / 'adapter_config.json').write_text('{}')"
    )
    jobs = [
        job(
            "t",
            "results/t/run_info.json",
            train,
            outputs=("results/t",),
            produces_adapter="checkpoints/t",
        ),
        job(
            "use",
            "results/t/drift.json",
            writes("results/t/drift.json"),
            depends_on=("t",),
            needs_adapter="checkpoints/t",
        ),
    ]
    first = run(repo, jobs)
    assert first["t"]["status"] == "done" and first["t"]["wall_time_s"] == 1.5
    assert first["t"]["scaler_retries"] == 2
    assert first["use"]["status"] == "done"
    second = run(repo, jobs)
    assert {r["status"] for r in second.values()} == {"skipped-existing"}


def test_dependent_is_redone_when_its_dependency_was_redone(repo: Path) -> None:
    for path in ("results/t/run_info.json", "results/e/summary.json"):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text("old")
    jobs = [
        job(
            "t",
            "results/t/run_info.json",
            writes("results/t/run_info.json", "new"),
            always_run=True,
        ),
        job(
            "e",
            "results/e/summary.json",
            writes("results/e/summary.json", "new"),
            depends_on=("t",),
            rerun_if_deps_ran=True,
        ),
        job("f", "results/f/x.json", writes("results/f/x.json"), depends_on=("t",)),
    ]
    results = run(repo, jobs)
    assert results["t"]["status"] == "done" and results["e"]["status"] == "done"
    assert (repo / "results/e/summary.json").read_text() == "new"


def test_always_run_job_with_an_unchanged_result_counts_as_existing(repo: Path) -> None:
    (repo / "results/e").mkdir(parents=True)
    (repo / "results/e/summary.json").write_text("current")
    jobs = [
        job(
            "e",
            "results/e/summary.json",
            stub("print('summary is current; nothing to do')"),
            always_run=True,
        ),
        job(
            "after",
            "results/g/x.json",
            writes("results/g/x.json"),
            depends_on=("e",),
            rerun_if_deps_ran=True,
        ),
    ]
    results = run(repo, jobs)
    assert results["e"]["status"] == "skipped-existing"
    assert results["after"]["status"] == "done"
    assert (repo / "results/e/summary.json").read_text() == "current"


def test_figure_is_drawn_from_the_values_that_finished(repo: Path) -> None:
    fail = stub("import sys; print('crashed'); sys.exit(1)")
    jobs = [
        job("v1", "results/v1/run_info.json", writes("results/v1/run_info.json")),
        job("v2", "results/v2/run_info.json", fail, outputs=("results/v2",)),
        job(
            "fig",
            "results/sweeps/s/summary.json",
            writes("results/sweeps/s/summary.json"),
            depends_on=("v1", "v2"),
            min_deps_done=1,
            always_run=True,
        ),
    ]
    results = run(repo, jobs)
    assert results["v2"]["status"] == "failed" and results["fig"]["status"] == "done"
    # With no finished value the figure is skipped, not failed.
    jobs_none = [
        job("w1", "results/w1/run_info.json", fail, outputs=("results/w1",)),
        job(
            "fig2",
            "results/sweeps/t/summary.json",
            fail,
            depends_on=("w1",),
            min_deps_done=1,
            always_run=True,
        ),
    ]
    results = run(repo, jobs_none)
    assert results["fig2"]["status"] == "skipped-dependency"


OOM_THEN_OK = stub(
    "import sys, pathlib; "
    "small = sys.argv[-1] == 'batch_size=1'; "
    "print('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1 GiB') if not small "
    "else pathlib.Path('results/r').mkdir(parents=True, exist_ok=True); "
    "small and pathlib.Path('results/r/run_info.json').write_text('{}'); "
    "sys.exit(0 if small else 1)"
)


def test_out_of_memory_retries_once_with_the_small_batch(repo: Path) -> None:
    jobs = [
        job(
            "r",
            "results/r/run_info.json",
            OOM_THEN_OK,
            outputs=("results/r",),
            oom_retry_args=("--set", "batch_size=1"),
        )
    ]
    results = run(repo, jobs)
    assert results["r"]["status"] == "done"
    assert results["r"]["retried_with_small_batch"] is True


def test_a_failure_that_is_not_out_of_memory_is_not_retried(repo: Path) -> None:
    counter = repo / "count.txt"
    other_error = stub(
        f"import pathlib, sys; p = pathlib.Path({str(counter)!r}); "
        "p.write_text(p.read_text() + 'x' if p.exists() else 'x'); "
        "print('ValueError: something else'); sys.exit(1)"
    )
    jobs = [
        job(
            "r",
            "results/r/run_info.json",
            other_error,
            outputs=("results/r",),
            oom_retry_args=("--set", "batch_size=1"),
        )
    ]
    results = run(repo, jobs)
    assert results["r"]["status"] == "failed" and results["r"]["oom"] is False
    assert counter.read_text() == "x"  # ran once


def test_out_of_memory_triggers_the_fallback_and_is_remembered(repo: Path) -> None:
    oom = stub(
        "import sys; print('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1 GiB'); "
        "sys.exit(1)"
    )
    jobs = [
        job("big", "results/big/run_info.json", oom, outputs=("results/big",)),
        job("big_next", "results/big2/x.json", writes("results/big2/x.json"), depends_on=("big",)),
        job(
            "small",
            "results/small/run_info.json",
            writes("results/small/run_info.json", "{}"),
            fallback_for=("big",),
        ),
        job(
            "small_next",
            "results/small2/x.json",
            writes("results/small2/x.json"),
            depends_on=("small",),
            fallback_for=("big",),
        ),
    ]
    first = run(repo, jobs)
    assert first["big"]["status"] == "failed" and first["big"]["oom"] is True
    assert first["big"]["fallback_jobs"] == ["small", "small_next"]
    assert first["big_next"]["status"] == "skipped-dependency"
    assert first["small"]["status"] == "done" and first["small_next"]["status"] == "done"
    second = run(repo, jobs)
    assert second["big"]["status"] == "skipped-oom-earlier"
    assert second["small"]["status"] == "skipped-existing"
    third = run(repo, jobs, retry_oom=True)
    assert third["big"]["status"] == "failed"


def test_fallback_is_not_needed_when_the_primary_works_or_fails_otherwise(repo: Path) -> None:
    for i, (primary, expected) in enumerate(
        (
            (writes("results/big0/run_info.json", "{}"), "done"),
            (stub("import sys; print('typo'); sys.exit(1)"), "failed"),
        )
    ):
        jobs = [
            job(f"big{i}", f"results/big{i}/run_info.json", primary, outputs=(f"results/big{i}",)),
            job(
                f"small{i}",
                f"results/small{i}/run_info.json",
                writes(f"results/small{i}/run_info.json"),
                fallback_for=(f"big{i}",),
            ),
        ]
        results = run(repo, jobs)
        assert results[f"big{i}"]["status"] == expected
        assert results[f"small{i}"]["status"] == "not-needed"
        assert not (repo / f"results/small{i}").exists()


def test_budget_does_not_start_a_job_that_will_not_fit(repo: Path) -> None:
    minutes = iter([0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    jobs = [
        job("a", "results/a/x.json", writes("results/a/x.json"), estimate_min=5),
        job("big", "results/b/x.json", writes("results/b/x.json"), estimate_min=30),
        job("small", "results/c/x.json", writes("results/c/x.json"), estimate_min=2),
    ]
    results = run(repo, jobs, budget_min=20, clock=lambda: next(minutes, 10.0) * 60)
    assert results["a"]["status"] == "done"
    assert results["big"]["status"] == "skipped-budget"  # 30 > the 10 minutes left
    assert results["small"]["status"] == "done"  # a later, smaller job still fits
    assert not (repo / "results/b").exists()


def test_stage_selection_and_unknown_stage(repo: Path) -> None:
    jobs = [
        job("a", "results/a/x.json", writes("results/a/x.json"), stage="one"),
        job("b", "results/b/x.json", writes("results/b/x.json"), stage="two"),
    ]
    assert list(run(repo, jobs, stages=["two"])) == ["b"]
    with pytest.raises(ValueError, match="unknown stages"):
        run(repo, jobs, stages=["three"])


def test_dependency_outside_the_selected_stages_counts_only_if_finished(repo: Path) -> None:
    jobs = [
        job("a", "results/a/x.json", writes("results/a/x.json"), stage="one"),
        job("b", "results/b/x.json", writes("results/b/x.json"), stage="two", depends_on=("a",)),
    ]
    assert run(repo, jobs, stages=["two"])["b"]["status"] == "skipped-dependency"
    run(repo, jobs, stages=["one"])
    assert run(repo, jobs, stages=["two"])["b"]["status"] == "done"


def test_results_are_pushed_after_each_job_and_others_work_is_merged(
    repo: Path, tmp_path: Path
) -> None:
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-q", "origin", "main")
    other = tmp_path / "other"
    git(tmp_path, "clone", "-q", str(remote), str(other))
    git(other, "config", "user.name", "other")
    git(other, "config", "user.email", "other@example.com")
    (other / "notes.txt").write_text("from elsewhere\n")
    git(other, "add", "notes.txt")
    git(other, "commit", "-q", "-m", "other work")
    git(other, "push", "-q", "origin", "main")

    fails = stub("import sys; print('bad'); sys.exit(1)")
    jobs = [
        job("a", "results/a/x.json", writes("results/a/x.json")),
        job("b", "results/b/run_info.json", fails, outputs=("results/b",)),
    ]
    results = run(repo, jobs, do_push=True)
    assert results["a"]["status"] == "done" and results["b"]["status"] == "failed"
    remote_files = set(git(remote, "ls-tree", "-r", "--name-only", "main").split())
    assert {"results/a/x.json", "notes.txt", STATUS_FILE, f"{FAILURE_DIR}/b.log"} <= remote_files
    assert not any(f.startswith("results/b/") for f in remote_files)


def test_strict_rescoring_runs_last_in_the_followup_stage_and_commits_every_run() -> None:
    followup = [j for j in QUEUE if j.stage == "m5_followup"]
    rescore = BY_NAME["m5_rescore_strict"]
    assert followup[-1] is rescore
    assert rescore.always_run
    assert [c[1] for c in rescore.commands] == [
        "scripts/rescore_trackb.py",
        "scripts/plot_trackb_strict.py",
    ]
    assert rescore.marker in rescore.outputs
    for run in ("trackb_full", *TRACKB_VARIANTS):
        assert f"results/{run}/trackb_eval/rescored_trained.json" in rescore.outputs
    assert "results/trackb_full/trackb_eval/rescored_base.json" in rescore.outputs
    assert "assets/m5_trackb_strict.png" in rescore.outputs
    # Nothing it writes may land on an original evaluation file.
    assert not any("summary_" in path or path.endswith(".jsonl") for path in rescore.outputs)
