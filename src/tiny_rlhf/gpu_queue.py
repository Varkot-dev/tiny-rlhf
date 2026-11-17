"""The queue of GPU jobs still to run, and the runner that works through it.

One Colab cell (docs/COLAB_QUEUE.md) starts `scripts/colab_queue.py`, which runs the
remaining GPU work of the project back to back. The design answers problems met on
Colab so far:

- output must be visible while a job runs, so every command is streamed line by line;
- a failure must never be mistaken for a result, so a job counts as done only if every
  command exited with status 0 AND the job's completion file was written during this
  launch; the partial files of a failed job are discarded before anything is committed;
- a disconnect must not lose work, so each finished job is committed and pushed at once,
  and on a fresh clone every job whose completion file is already in the repo is skipped;
- a free session is short, so a job is not started unless its (rough) time estimate fits
  in what is left of the budget.

The queue is plain data (a list of Job records) so tests can check it without a GPU.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Priority order: the Track B result first (the hero figure), then the follow-up to it,
# then the reward model, then TRL parity on the full model, then the four ablation sweeps.
STAGES = ("m5", "m5_followup", "m3", "m2", "beta", "lora_rank", "data_size", "label_noise")

# Follow-up to the first full Track B run (results/trackb_full), which over-optimised:
# preference accuracy on validation pairs was 0.993 by step 125, the answers shrank to a
# median of 1 word, and task-correct on clean documents fell from 0.74 to 0.37. Each
# variant changes one thing (see the header of its config) and is compared with the same
# base evaluation, results/trackb_full/trackb_eval/summary_base.json.
TRACKB_VARIANTS = ("trackb_beta0.5", "trackb_steps60", "trackb_sft1")
TRACKB_BASE_SUMMARY = "results/trackb_full/trackb_eval/summary_base.json"
SWEEP_STAGES = ("beta", "lora_rank", "data_size", "label_noise")

STATUS_FILE = "results/queue_status.json"
FAILURE_DIR = "results/colab_failures"
TAIL_LINES = 80  # lines kept from a failed job's output, as in docs/COLAB_ONE_HOUR.md
STATUS_LINES = 30  # of those, how many also go into queue_status.json

GIT_NAME = "Varshith Kotagiri"
GIT_EMAIL = "Varkot-dev@users.noreply.github.com"

# Statuses that satisfy a dependency.
OK_STATUSES = ("done", "skipped-existing")

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
# Same micro-batch as the only measured full run (dpo_full: 1 pair, 8.8 GB peak on a T4).
SMALL_BATCH = ("--set", "batch_size=1", "--set", "grad_accum=8", "--set", "eval_batch_size=2")
# Must stay identical to docs/COLAB_ONE_HOUR.md: the comparison script refuses to compare a
# base and a trained evaluation made with different settings.
TRACKB_EVAL_ARGS = ("--with-probes", "--batch-size", "16", "--limit", "150")

# Installs TRL (the `parity` extra of pyproject.toml) only when it is missing. It runs
# before BOTH parity runs, so the hand-written and the TRL run see the same versions of
# the libraries TRL may upgrade.
ENSURE_TRL = (
    "python",
    "-c",
    "import importlib.util, subprocess, sys; "
    "sys.exit(0) if importlib.util.find_spec('trl') else "
    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[parity]'])",
)

# Time estimates, in minutes, on a free Colab T4. THESE ARE ROUGH GUESSES, not
# measurements. The only measured full run is dpo_full: 463 optimizer steps of 8 pairs at
# up to 512 tokens, policy plus reference, fp16, in 1,737 s (29 minutes). Everything else
# is scaled from that by eye. They only decide whether a job is started near the end of
# the budget; replace them as real timings arrive in results/queue_status.json.
ESTIMATE_MIN = {
    # Measured on a Colab T4 on 2026-11-17 (results/trackb_full), plus a margin for model loading.
    "m5_eval_base": 5,  # 1,050 generations scored in 221 s
    "m5_train": 11,  # 375 steps in 585 s
    "m5_eval_trained": 4,  # 1,050 generations scored in 131 s (shorter answers)
    "m5_compare": 2,  # general regression check and tables took under a minute
    # Measured for trackb_full on a T4: 375 training steps in 585 s with a 7.3 GB peak, and
    # the base evaluation (1,050 generations) in about 4 to 5 minutes with model loading.
    "m5f_train": 12,  # 375 steps, as trackb_full, plus loading two models
    "m5f_train_short": 4,  # 60 steps
    "m5f_eval": 6,
    "m5f_compare": 5,  # general regression check on 200 conversations, then tables
    "m5f_figure": 2,
    "m3_reward": 25,  # like dpo_full but one model and no vocabulary head
    "m3_reward_vs_dpo": 2,
    "m2_parity_ours": 75,  # float32 is several times slower than fp16 on a T4
    "m2_parity_trl": 75,
    "m2_parity_figure": 2,
    "m2_parity_ours_fp16": 32,
    "m2_parity_trl_fp16": 32,
    "m2_parity_figure_fp16": 2,
    "sweep_run": 16,  # 2,000 pairs: about half of dpo_full
    "sweep_run_beta": 19,  # plus the drift measurement
    "sweep_run_data_size": 20,  # 340 steps
    "sweep_figure": 2,
}


@dataclass(frozen=True)
class Job:
    """One step of the queue.

    name             unique id; also the name of its failure log
    stage            which --stages entry selects it
    commands         argv tuples run in order; a leading "python" means this interpreter
    marker           file whose presence means the job is finished (relative to the repo)
    outputs          paths committed when the job succeeds and discarded when it fails
    estimate_min     rough guess of its run time on a T4, used by the budget
    depends_on       jobs that must be finished first
    min_deps_done    if set, the job may run when at least this many of depends_on are
                     finished (a sweep figure is drawn from the values that did finish)
    needs_adapter    a LoRA adapter directory under checkpoints/ this job reads. Adapters
                     are not in git, so on a fresh machine the job that writes it
                     (produces_adapter) is run again even if its results are in the repo
    produces_adapter the adapter directory this job writes
    always_run       run even when the marker exists: the command itself decides whether
                     anything needs redoing (base evaluation, figures). If the marker's
                     contents are unchanged afterwards the job counts as skipped-existing
    rerun_if_deps_ran run again, even with the marker present, if a dependency was redone
                     during this launch (its result would be stale otherwise)
    fallback_for     this job runs only if one of the named jobs ran out of GPU memory
    oom_retry_args   if a command fails with an out-of-memory error, it is run once more
                     with these arguments appended (a smaller micro-batch)
    """

    name: str
    stage: str
    commands: tuple[tuple[str, ...], ...]
    marker: str
    outputs: tuple[str, ...]
    estimate_min: float
    depends_on: tuple[str, ...] = ()
    min_deps_done: int | None = None
    needs_adapter: str | None = None
    produces_adapter: str | None = None
    always_run: bool = False
    rerun_if_deps_ran: bool = False
    fallback_for: tuple[str, ...] = ()
    oom_retry_args: tuple[str, ...] = ()


def _m5_jobs() -> list[Job]:
    """Track B, split as in docs/COLAB_ONE_HOUR.md stage 1."""
    run = "trackb_full"
    evals = f"results/{run}/trackb_eval"
    adapter = f"checkpoints/{run}"
    summaries = (
        "--base",
        f"{evals}/summary_base.json",
        "--trained",
        f"{evals}/summary_trained.json",
    )
    return [
        Job(
            name="m5_eval_base",
            stage="m5",
            commands=(
                (
                    "python",
                    "scripts/eval_trackb.py",
                    "--model",
                    QWEN,
                    "--label",
                    "base",
                    "--out",
                    evals,
                    *TRACKB_EVAL_ARGS,
                    "--reuse-if-current",
                ),
            ),
            marker=f"{evals}/summary_base.json",
            outputs=(evals,),
            estimate_min=ESTIMATE_MIN["m5_eval_base"],
            # The script checks itself whether a saved base evaluation matches the current
            # data and settings, and evaluates again only if it does not.
            always_run=True,
        ),
        Job(
            name="m5_train",
            stage="m5",
            commands=(
                (
                    "python",
                    "scripts/train_trackb.py",
                    "--config",
                    "configs/trackb_full.yaml",
                    "--set",
                    "eval_every=125",
                ),
                (
                    "python",
                    "scripts/plot_training.py",
                    "--run",
                    f"results/{run}",
                    "--out",
                    f"assets/m5_training_{run}.png",
                ),
            ),
            marker=f"results/{run}/run_info.json",
            outputs=(f"results/{run}", f"assets/m5_training_{run}.png"),
            estimate_min=ESTIMATE_MIN["m5_train"],
            produces_adapter=adapter,
            oom_retry_args=SMALL_BATCH,
        ),
        Job(
            name="m5_eval_trained",
            stage="m5",
            commands=(
                (
                    "python",
                    "scripts/eval_trackb.py",
                    "--model",
                    QWEN,
                    "--adapter",
                    adapter,
                    "--label",
                    "trained",
                    "--out",
                    evals,
                    *TRACKB_EVAL_ARGS,
                ),
            ),
            marker=f"{evals}/summary_trained.json",
            outputs=(evals,),
            estimate_min=ESTIMATE_MIN["m5_eval_trained"],
            depends_on=("m5_train",),
            needs_adapter=adapter,
            rerun_if_deps_ran=True,
        ),
        Job(
            name="m5_compare",
            stage="m5",
            commands=(
                (
                    "python",
                    "scripts/eval_general.py",
                    "--model",
                    QWEN,
                    "--adapter",
                    adapter,
                    "--out",
                    f"results/{run}/general_regression.json",
                ),
                ("python", "scripts/compare_trackb.py", *summaries),
                ("python", "scripts/plot_trackb.py", *summaries),
                ("python", "scripts/make_examples.py", *summaries),
            ),
            marker=f"results/{run}/trackb_table.md",
            outputs=(
                f"results/{run}/general_regression.json",
                f"results/{run}/trackb_table.md",
                f"results/{run}/trackb_compare.json",
                f"results/{run}/examples.md",
                f"assets/m5_trackb_{run}.png",
            ),
            estimate_min=ESTIMATE_MIN["m5_compare"],
            depends_on=("m5_eval_base", "m5_eval_trained"),
            needs_adapter=adapter,
            rerun_if_deps_ran=True,
        ),
    ]


def _m5_followup_jobs() -> list[Job]:
    """Per variant: train, evaluate with the m5 settings, compare with trackb_full's base."""
    jobs: list[Job] = []
    for run in TRACKB_VARIANTS:
        evals = f"results/{run}/trackb_eval"
        adapter = f"checkpoints/{run}"
        summaries = ("--base", TRACKB_BASE_SUMMARY, "--trained", f"{evals}/summary_trained.json")
        short = "steps" in run
        jobs += [
            Job(
                name=f"{run}_train",
                stage="m5_followup",
                commands=(
                    ("python", "scripts/train_trackb.py", "--config", f"configs/{run}.yaml"),
                    (
                        "python",
                        "scripts/plot_training.py",
                        "--run",
                        f"results/{run}",
                        "--out",
                        f"assets/m5_training_{run}.png",
                    ),
                ),
                marker=f"results/{run}/run_info.json",
                outputs=(f"results/{run}", f"assets/m5_training_{run}.png"),
                estimate_min=ESTIMATE_MIN["m5f_train_short" if short else "m5f_train"],
                produces_adapter=adapter,
                oom_retry_args=SMALL_BATCH,
            ),
            Job(
                name=f"{run}_eval",
                stage="m5_followup",
                commands=(
                    (
                        "python",
                        "scripts/eval_trackb.py",
                        "--model",
                        QWEN,
                        "--adapter",
                        adapter,
                        "--label",
                        "trained",
                        "--out",
                        evals,
                        *TRACKB_EVAL_ARGS,
                    ),
                ),
                marker=f"{evals}/summary_trained.json",
                outputs=(evals,),
                estimate_min=ESTIMATE_MIN["m5f_eval"],
                depends_on=(f"{run}_train",),
                needs_adapter=adapter,
                rerun_if_deps_ran=True,
            ),
            Job(
                name=f"{run}_compare",
                stage="m5_followup",
                commands=(
                    (
                        "python",
                        "scripts/eval_general.py",
                        "--model",
                        QWEN,
                        "--adapter",
                        adapter,
                        "--out",
                        f"results/{run}/general_regression.json",
                    ),
                    # These three write next to the TRAINED run, not next to trackb_full.
                    ("python", "scripts/compare_trackb.py", *summaries),
                    ("python", "scripts/plot_trackb.py", *summaries),
                    ("python", "scripts/make_examples.py", *summaries),
                ),
                marker=f"results/{run}/trackb_table.md",
                outputs=(
                    f"results/{run}/general_regression.json",
                    f"results/{run}/trackb_table.md",
                    f"results/{run}/trackb_compare.json",
                    f"results/{run}/examples.md",
                    f"assets/m5_trackb_{run}.png",
                ),
                estimate_min=ESTIMATE_MIN["m5f_compare"],
                depends_on=("m5_eval_base", f"{run}_eval"),
                needs_adapter=adapter,
                rerun_if_deps_ran=True,
            ),
        ]
    compares = tuple(f"{run}_compare" for run in TRACKB_VARIANTS)
    jobs.append(
        Job(
            name="m5_followup_figure",
            stage="m5_followup",
            commands=(("python", "scripts/plot_trackb_variants.py"),),
            marker="results/trackb_variants_summary.json",
            outputs=("results/trackb_variants_summary.json", "assets/m5_trackb_variants.png"),
            estimate_min=ESTIMATE_MIN["m5f_figure"],
            # trackb_full itself is in the repo; the figure is drawn from the variants that
            # finished (it names the missing ones) and redrawn each launch.
            depends_on=compares,
            min_deps_done=1,
            always_run=True,
        )
    )
    return jobs


def _m3_jobs() -> list[Job]:
    return [
        Job(
            name="m3_reward",
            stage="m3",
            commands=(
                ("python", "scripts/train_reward.py", "--config", "configs/reward_full.yaml"),
            ),
            marker="results/reward_full/run_info.json",
            outputs=("results/reward_full",),
            estimate_min=ESTIMATE_MIN["m3_reward"],
            oom_retry_args=SMALL_BATCH,
        ),
        Job(
            name="m3_reward_vs_dpo",
            stage="m3",
            commands=(
                (
                    "python",
                    "scripts/plot_reward_vs_dpo.py",
                    "--dpo",
                    "results/dpo_full_t4",
                    "--reward",
                    "results/reward_full",
                ),
            ),
            marker="results/reward_full/reward_vs_dpo.json",
            outputs=(
                "results/reward_full/reward_vs_dpo.json",
                "assets/m3_reward_vs_dpo_reward_full.png",
            ),
            estimate_min=ESTIMATE_MIN["m3_reward_vs_dpo"],
            depends_on=("m3_reward",),
            always_run=True,
        ),
    ]


def _parity_jobs(fp16: bool) -> list[Job]:
    """The three M2 jobs: hand-written run, TRL run, overlay figure.

    configs/parity_full.yaml is float32 so that exact agreement can hold (under fp16 the
    Trainer drops an overflowed batch while the hand-written loop redoes it). If float32
    does not fit the GPU, both runs are done again in fp16 under their own names.
    """
    suffix = "_fp16" if fp16 else ""
    ours, trl = f"parity_full{suffix}", f"parity_full{suffix}_trl"
    config = ("--config", "configs/parity_full.yaml")
    extra = ("--set", "amp=true") if fp16 else ()
    fallback_for = ("m2_parity_ours", "m2_parity_trl") if fp16 else ()
    return [
        Job(
            name=f"m2_parity_ours{suffix}",
            stage="m2",
            commands=(
                ENSURE_TRL,
                ("python", "scripts/train_dpo.py", *config, "--run-name", ours, *extra),
            ),
            marker=f"results/{ours}/run_info.json",
            outputs=(f"results/{ours}",),
            estimate_min=ESTIMATE_MIN[f"m2_parity_ours{suffix}"],
            fallback_for=fallback_for,
        ),
        Job(
            name=f"m2_parity_trl{suffix}",
            stage="m2",
            commands=(
                ENSURE_TRL,
                ("python", "scripts/train_trl.py", *config, "--run-name", trl, *extra),
            ),
            marker=f"results/{trl}/run_info.json",
            outputs=(f"results/{trl}",),
            estimate_min=ESTIMATE_MIN[f"m2_parity_trl{suffix}"],
            depends_on=(f"m2_parity_ours{suffix}",),
            fallback_for=fallback_for,
        ),
        Job(
            name=f"m2_parity_figure{suffix}",
            stage="m2",
            commands=(
                (
                    "python",
                    "scripts/plot_parity.py",
                    "--ours",
                    f"results/{ours}",
                    "--trl",
                    f"results/{trl}",
                ),
            ),
            marker=f"results/{ours}/parity_summary.json",
            outputs=(f"results/{ours}/parity_summary.json", f"assets/m2_parity_{ours}.png"),
            estimate_min=ESTIMATE_MIN[f"m2_parity_figure{suffix}"],
            depends_on=(f"m2_parity_trl{suffix}",),
            always_run=True,
            fallback_for=fallback_for,
        ),
    ]


def _sweep_jobs(root: Path, stage: str) -> list[Job]:
    """One job per sweep value (so each is pushed on its own), then the sweep's figure."""
    from tiny_rlhf.sweeps import load_sweep

    path = f"configs/sweeps/{stage}.yaml"
    sweep = load_sweep(root / path)
    index = f"results/sweeps/{sweep.name}/index.json"
    beta = stage == "beta"
    jobs: list[Job] = []
    for run in sweep.runs:
        train = ("python", "scripts/run_sweep.py", "--sweep", path, "--only", run.label)
        if beta:
            # The drift measurement reads the LoRA adapter, which exists only on the machine
            # that trained it, so it runs in the same job. No --skip-existing here: if a
            # session finished training but not drift, the next one must train again.
            commands: tuple[tuple[str, ...], ...] = (
                train,
                ("python", "scripts/eval_drift.py", "--run", f"results/{run.run_name}"),
            )
            marker = f"results/{run.run_name}/drift.json"
            estimate = ESTIMATE_MIN["sweep_run_beta"]
        else:
            commands = ((*train, "--skip-existing"),)
            marker = f"results/{run.run_name}/run_info.json"
            key = "sweep_run_data_size" if stage == "data_size" else "sweep_run"
            estimate = ESTIMATE_MIN[key]
        jobs.append(
            Job(
                name=f"{stage}_{run.label}",
                stage=stage,
                commands=commands,
                marker=marker,
                outputs=(f"results/{run.run_name}", index),
                estimate_min=estimate,
            )
        )
    if beta:
        figure_cmd: tuple[str, ...] = ("python", "scripts/plot_beta_sweep.py")
        figure = "assets/m6_beta_sweep.png"
    else:
        figure_cmd = ("python", "scripts/plot_sweep.py", "--sweep", stage)
        figure = f"assets/m6_sweep_{sweep.name}.png"
    jobs.append(
        Job(
            name=f"{stage}_figure",
            stage=stage,
            commands=(figure_cmd,),
            marker=f"results/sweeps/{sweep.name}/summary.json",
            outputs=(f"results/sweeps/{sweep.name}/summary.json", figure),
            estimate_min=ESTIMATE_MIN["sweep_figure"],
            depends_on=tuple(j.name for j in jobs),
            # The figure scripts leave out unfinished runs and name them in the footer, so
            # the figure is drawn from whatever values did finish, and redrawn each launch.
            min_deps_done=1,
            always_run=True,
        )
    )
    return jobs


def build_queue(root: str | Path = ".") -> list[Job]:
    """Every remaining GPU job of the project, in the order it is run."""
    root = Path(root)
    jobs = (
        _m5_jobs()
        + _m5_followup_jobs()
        + _m3_jobs()
        + _parity_jobs(fp16=False)
        + _parity_jobs(fp16=True)
    )
    for stage in SWEEP_STAGES:
        jobs += _sweep_jobs(root, stage)
    return jobs


# --------------------------------------------------------------------------------------
# Running commands and git
# --------------------------------------------------------------------------------------


def run_streaming(
    argv: Sequence[str],
    cwd: str | Path = ".",
    env: dict[str, str] | None = None,
    echo: Callable[[str], None] | None = None,
) -> tuple[int, list[str]]:
    """Run a command, showing its output line by line as it is produced.

    stderr is merged into stdout. Returns (exit status, the last TAIL_LINES lines).
    """
    if echo is None:

        def echo(text: str) -> None:
            print(text, end="", flush=True)

    tail: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)
    # A Python child writing to a pipe buffers stdout but not stderr, so lines would
    # arrive out of order and progress would appear in bursts. Unbuffered output keeps
    # both streams in the order they were written and shows progress live.
    child_env = dict(os.environ if env is None else env)
    child_env["PYTHONUNBUFFERED"] = "1"
    try:
        proc = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
    except OSError as err:  # for example a missing program
        line = f"could not start {argv[0]}: {err}\n"
        echo(line)
        tail.append(line)
        return 127, list(tail)
    assert proc.stdout is not None
    for line in proc.stdout:
        echo(line)
        tail.append(line)
    return proc.wait(), list(tail)


def is_oom(lines: Iterable[str]) -> bool:
    return "out of memory" in "".join(lines).lower()


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, check=check, capture_output=True, text=True)


def is_tracked(root: Path, path: str) -> bool:
    return bool(git(root, "ls-files", "--", path, check=False).stdout.strip())


def discard_partial(root: Path, paths: Iterable[str]) -> None:
    """Put a failed job's outputs back to their committed state.

    Files the repo already has are restored (a rerun may have deleted or overwritten
    them); files it does not have are removed. Nothing half-written can be committed.
    """
    for path in paths:
        if is_tracked(root, path):
            git(root, "checkout", "-q", "--", path, check=False)
            git(root, "clean", "-fdq", "--", path, check=False)
        else:
            target = root / path
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif target.exists():
                target.unlink()


def commit(root: Path, paths: Iterable[str], message: str) -> bool:
    """Stage the given paths (those that exist or are tracked) and commit. True if committed."""
    for path in paths:
        if (root / path).exists() or is_tracked(root, path):
            git(root, "add", "-A", "--", path)
    if git(root, "diff", "--cached", "--quiet", check=False).returncode == 0:
        return False
    identity = ("-c", f"user.name={GIT_NAME}", "-c", f"user.email={GIT_EMAIL}")
    git(root, *identity, "commit", "-q", "-m", message)
    return True


def push(
    root: Path, branch: str, retries: int = 4, sleep: Callable[[float], None] = time.sleep
) -> bool:
    """Merge whatever is new on the branch, then push. Retries with backoff."""
    identity = ("-c", f"user.name={GIT_NAME}", "-c", f"user.email={GIT_EMAIL}")
    for attempt in range(retries):
        fetched = git(root, "fetch", "-q", "origin", branch, check=False).returncode == 0
        merged = False
        if fetched:
            result = git(root, *identity, "merge", "-q", "--no-edit", "FETCH_HEAD", check=False)
            merged = result.returncode == 0
            if not merged:
                git(root, "merge", "--abort", check=False)
        if (
            merged
            and git(root, "push", "-q", "origin", f"HEAD:{branch}", check=False).returncode == 0
        ):
            return True
        if attempt < retries - 1:
            sleep(2.0 * 2**attempt)
    return False


def gpu_name() -> str:
    """The GPU's name from nvidia-smi, without creating a CUDA context in this process."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "no GPU"
    return out.splitlines()[0] if out else "no GPU"


def current_branch(root: Path) -> str:
    return git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


# --------------------------------------------------------------------------------------
# The queue runner
# --------------------------------------------------------------------------------------


def marker_done(root: Path, job: Job, written_after: float | None = None) -> bool:
    path = root / job.marker
    if not path.exists():
        return False
    # One second of slack for file systems with coarse timestamps.
    return written_after is None or path.stat().st_mtime >= written_after - 1.0


def file_digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def load_status(root: Path) -> dict[str, Any]:
    path = root / STATUS_FILE
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {"jobs": {}}


def select(jobs: Sequence[Job], stages: Sequence[str] | None) -> list[Job]:
    """The jobs of the given stages, stage by stage in the order given."""
    if not stages:
        return list(jobs)
    known = {j.stage for j in jobs}
    unknown = [s for s in stages if s not in known]
    if unknown:
        raise ValueError(f"unknown stages {unknown}; known: {sorted(known)}")
    return [j for stage in stages for j in jobs if j.stage == stage]


def resolve_argv(command: Sequence[str], python: str) -> list[str]:
    return [python, *command[1:]] if command[0] == "python" else list(command)


def forced_producers(root: Path, selected: Sequence[Job]) -> set[str]:
    """Jobs to run again because an unfinished job needs their adapter and it is gone.

    Adapters live under checkpoints/, which is not in git. After a disconnect the
    training results are in the repo but the adapter is not on the new machine.
    """
    forced: set[str] = set()
    by_adapter = {j.produces_adapter: j.name for j in selected if j.produces_adapter}
    for job in selected:
        if job.needs_adapter is None or marker_done(root, job):
            continue
        if not (root / job.needs_adapter / "adapter_config.json").exists():
            producer = by_adapter.get(job.needs_adapter)
            if producer:
                forced.add(producer)
    return forced


def describe(jobs: Sequence[Job], root: str | Path = ".") -> list[dict[str, Any]]:
    """What --list shows: each job, its estimate, and whether its completion file exists."""
    root = Path(root)
    rows = []
    for job in jobs:
        if marker_done(root, job):
            status = "done"
        elif job.fallback_for:
            status = "pending (only if float32 runs out of memory)"
        else:
            status = "pending"
        rows.append(
            {
                "job": job.name,
                "stage": job.stage,
                "status": status,
                "estimate_min": job.estimate_min,
            }
        )
    return rows


def run_queue(
    jobs: Sequence[Job],
    root: str | Path = ".",
    stages: Sequence[str] | None = None,
    dry_run: bool = False,
    do_push: bool = True,
    budget_min: float | None = None,
    python: str = sys.executable,
    branch: str | None = None,
    gpu: str | None = None,
    clock: Callable[[], float] = time.time,
    echo: Callable[[str], None] | None = None,
    retry_oom: bool = False,
) -> dict[str, dict[str, Any]]:
    """Work through the queue. Returns {job name: {"status": ..., "minutes": ...}}.

    Statuses:
        done                 ran and wrote its completion file during this launch
        skipped-existing     already finished (completion file in the repo, or an
                             always-run job whose result did not change)
        failed               non-zero exit, or exit 0 without a fresh completion file
        skipped-dependency   a job it needs did not finish
        skipped-budget       its time estimate is more than the budget left
        not-needed           a fallback whose trigger (out of memory) did not happen
        skipped-oom-earlier  an earlier launch saw it run out of memory; its fallback is
                             used instead (pass retry_oom=True to try it again)
    """
    root = Path(root)
    if echo is None:

        def echo(text: str) -> None:
            print(text, end="", flush=True)

    by_name = {j.name: j for j in jobs}
    selected = select(jobs, stages)
    earlier = load_status(root).get("jobs", {})
    if retry_oom:  # forget earlier out-of-memory failures and try those jobs again
        earlier = {k: v for k, v in earlier.items() if not v.get("oom")}
    results: dict[str, dict[str, Any]] = {}

    if dry_run:
        for job in selected:
            state = "done" if marker_done(root, job) else "pending"
            note = (
                " (fallback: runs only if float32 runs out of memory)" if job.fallback_for else ""
            )
            echo(f"# {job.name} [{job.stage}] {state}, about {job.estimate_min:g} min{note}\n")
            for command in job.commands:
                echo(shlex.join(command) + "\n")
        return results

    branch = branch or current_branch(root)
    gpu = gpu or gpu_name()
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        # Progress bars redraw one line thousands of times; keep the output readable.
        "TQDM_DISABLE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    }
    forced = forced_producers(root, selected)
    started = clock()

    def oom_before(name: str) -> bool:
        row = results.get(name) or earlier.get(name) or {}
        return bool(row.get("oom")) and row.get("status") in ("failed", "skipped-oom-earlier")

    def dep_finished(name: str) -> bool:
        if name in results:
            return results[name]["status"] in OK_STATUSES
        return marker_done(root, by_name[name])

    def save_status() -> None:
        path = root / STATUS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"updated_unix": round(clock()), "gpu": gpu, "branch": branch}
        payload["jobs"] = {**earlier, **results}
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def record(job: Job, status: str, **extra: Any) -> None:
        results[job.name] = {"status": status, "stage": job.stage, "minutes": 0.0, **extra}

    for job in selected:
        has_marker = marker_done(root, job)
        # 1. A fallback runs only when its trigger happened.
        if job.fallback_for and not has_marker and not any(map(oom_before, job.fallback_for)):
            record(job, "not-needed")
            continue
        # 2. A job that already ran out of memory in an earlier launch is not tried again.
        if (
            earlier.get(job.name, {}).get("oom")
            and not has_marker
            and any(job.name in other.fallback_for for other in jobs)
        ):
            record(job, "skipped-oom-earlier", oom=True)
            continue
        # 3. Dependencies. One outside the selected stages counts if it is in the repo.
        for dep in job.depends_on:
            if dep not in by_name:
                raise ValueError(f"{job.name} depends on unknown job {dep}")
        finished = [d for d in job.depends_on if dep_finished(d)]
        needed = len(job.depends_on) if job.min_deps_done is None else job.min_deps_done
        if len(finished) < needed:
            blocked = [d for d in job.depends_on if d not in finished]
            record(job, "skipped-dependency", blocked_by=blocked[0])
            continue
        # 4. Already finished.
        deps_ran = any(results.get(d, {}).get("status") == "done" for d in job.depends_on)
        rerun = job.always_run or job.name in forced or (job.rerun_if_deps_ran and deps_ran)
        if has_marker and not rerun:
            record(job, "skipped-existing")
            continue
        # 5. Budget: do not start a job that probably cannot finish in the time left.
        if budget_min is not None:
            left = budget_min - (clock() - started) / 60.0
            if job.estimate_min > left:
                record(job, "skipped-budget", minutes_left=round(left, 1))
                continue

        echo(f"\n===== {job.name} ({job.stage}, estimate {job.estimate_min:g} min) =====\n")
        job_start = clock()
        digest_before = file_digest(root / job.marker)
        code, tail, retried = 0, [], False
        for command in job.commands:
            argv = resolve_argv(command, python)
            echo(f"$ {shlex.join(argv)}\n")
            code, tail = run_streaming(argv, cwd=root, env=env, echo=echo)
            if code != 0 and job.oom_retry_args and not retried and is_oom(tail):
                retried = True
                argv = [*argv, *job.oom_retry_args]
                echo("\nOut of GPU memory. Retrying with a smaller micro-batch.\n")
                echo(f"$ {shlex.join(argv)}\n")
                code, tail = run_streaming(argv, cwd=root, env=env, echo=echo)
            if code != 0:
                break
        minutes = round((clock() - job_start) / 60.0, 1)
        if job.always_run:
            # The command may legitimately leave an up-to-date file alone.
            ok = code == 0 and marker_done(root, job)
        else:
            ok = code == 0 and marker_done(root, job, written_after=job_start)

        if ok and job.always_run and digest_before == file_digest(root / job.marker):
            record(job, "skipped-existing", note="checked: already current")
            results[job.name]["minutes"] = minutes
            paths: list[str] = [STATUS_FILE]
            message = f"Update queue status after {job.name} (Colab {gpu})"
        elif ok:
            extra: dict[str, Any] = {"retried_with_small_batch": True} if retried else {}
            if job.marker.endswith("run_info.json"):
                try:
                    info = json.loads((root / job.marker).read_text())
                except ValueError:
                    info = {}
                keys = ("wall_time_s", "peak_gpu_memory_mb", "optimizer_steps", "scaler_retries")
                extra.update({k: info[k] for k in keys if info.get(k) is not None})
            record(job, "done", **extra)
            results[job.name]["minutes"] = minutes
            paths = [STATUS_FILE, *job.outputs]
            message = f"Add results for {job.name} (Colab {gpu})"
        else:
            oom = is_oom(tail)
            reason = (
                f"exit status {code}"
                if code != 0
                else f"exited with status 0 but did not write {job.marker} during this run"
            )
            fallbacks = [other.name for other in jobs if job.name in other.fallback_for]
            record(
                job,
                "failed",
                exit_code=code,
                reason=reason,
                oom=oom,
                last_lines=[line.rstrip("\n") for line in tail[-STATUS_LINES:]],
                **({"fallback_jobs": fallbacks} if oom and fallbacks else {}),
            )
            results[job.name]["minutes"] = minutes
            echo(f"\n*** {job.name} FAILED ({reason}). Its partial files are discarded.\n")
            discard_partial(root, job.outputs)
            log = Path(FAILURE_DIR) / f"{job.name}.log"
            (root / log).parent.mkdir(parents=True, exist_ok=True)
            (root / log).write_text(f"{reason}\nGPU: {gpu}\nminutes: {minutes}\n\n" + "".join(tail))
            paths = [STATUS_FILE, str(log)]
            message = f"Record failure of {job.name} (Colab {gpu})"
        save_status()
        if commit(root, paths, message) and do_push and not push(root, branch):
            echo(f"WARNING: could not push after {job.name}; it stays committed locally.\n")

    save_status()
    if commit(root, [STATUS_FILE], f"Update queue status (Colab {gpu})") and do_push:
        push(root, branch)
    return results


def format_table(results: dict[str, dict[str, Any]]) -> str:
    width = max([len(name) for name in results] + [3])
    lines = [f"{'job':<{width}}  {'status':<20}  minutes"]
    for name, row in results.items():
        lines.append(f"{name:<{width}}  {row['status']:<20}  {row.get('minutes', 0.0):>7.1f}")
    return "\n".join(lines)
