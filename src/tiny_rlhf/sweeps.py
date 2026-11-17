"""Running and summarising ablation sweeps (M6).

A sweep is one YAML file under configs/sweeps/: a base config, the one setting being
varied, and the values to try. Each value becomes one ordinary training run in its own
process, so a sweep is nothing more than a list of commands you could type by hand.

    load_sweep     read the YAML into a list of runs, each with its full command
    run_sweep      run them one after another, recording what happened in index.json
    run_summary    final held-out numbers of one run, with bootstrap intervals
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SCRIPT = "scripts/train_dpo.py"
SWEEP_KEYS = {
    "name",
    "description",
    "base_config",
    "field",
    "flag",
    "script",
    "run_name",
    "overrides",
    "values",
    "x_label",
    "x_scale",
}
ENTRY_KEYS = {"value", "label", "overrides", "extra_args", "script", "series"}


def format_label(value: Any) -> str:
    """Text form of a swept value for run names: 0.1 -> '0.1', 16 -> '16', True -> 'true'."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return repr(value)
    return str(value)


@dataclass(frozen=True)
class SweepRun:
    """One training run of a sweep."""

    label: str
    value: Any
    run_name: str
    command: tuple[str, ...]  # everything after the Python interpreter
    series: str | None = None


@dataclass(frozen=True)
class Sweep:
    name: str
    base_config: str
    field: str
    runs: tuple[SweepRun, ...]
    description: str = ""
    x_label: str | None = None
    x_scale: str = "linear"  # "linear", "log" or "categorical": a hint for the figure
    source: str = ""
    raw: dict[str, Any] = field(default_factory=dict, compare=False)


def _set_args(overrides: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in overrides.items():
        # json.dumps writes lists, numbers, true/false and null in a form YAML reads back.
        args += ["--set", f"{key}={json.dumps(value)}"]
    return args


def load_sweep(path: str | Path) -> Sweep:
    """Read a sweep file and expand it into runs.

    File format (all keys other than name, base_config, field and values are optional):

        name: beta                          # also the results/sweeps/<name>/ directory
        base_config: configs/dpo_full.yaml
        field: beta                         # the config field being varied
        flag: --noise-fraction              # pass the value with this CLI flag instead
                                            # of `--set field=value`
        script: scripts/train_dpo.py        # the training script to run
        run_name: "sweep_beta_{label}"      # {label}, {value} and {name} are filled in
        overrides: {n_train: 2000}          # applied to every run
        values:
          - 0.01                            # a bare value, or
          - value: 0.5                      # a mapping with per-run extras
            label: "0.5"
            overrides: {lora_alpha: 32}
            extra_args: ["--label-smoothing", "0.3"]
            series: robust loss             # groups runs in the figure
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    unknown = sorted(set(raw) - SWEEP_KEYS)
    if unknown:
        raise ValueError(f"unknown sweep keys in {path}: {unknown}")
    for key in ("name", "base_config", "field", "values"):
        if not raw.get(key):
            raise ValueError(f"sweep file {path} needs a non-empty '{key}'")
    x_scale = raw.get("x_scale", "linear")
    if x_scale not in ("linear", "log", "categorical"):
        raise ValueError(f"x_scale must be linear, log or categorical, got {x_scale!r}")

    name = str(raw["name"])
    pattern = raw.get("run_name", "sweep_{name}_{label}")
    shared = dict(raw.get("overrides") or {})
    runs: list[SweepRun] = []
    for item in raw["values"]:
        entry = item if isinstance(item, dict) else {"value": item}
        bad = sorted(set(entry) - ENTRY_KEYS)
        if bad or "value" not in entry:
            raise ValueError(f"bad sweep entry in {path}: {item!r}")
        value = entry["value"]
        label = str(entry.get("label", format_label(value)))
        run_name = pattern.format(label=label, value=format_label(value), name=name)
        overrides = {**shared, **(entry.get("overrides") or {})}
        command = [
            str(entry.get("script") or raw.get("script") or DEFAULT_SCRIPT),
            "--config",
            str(raw["base_config"]),
            "--run-name",
            run_name,
        ]
        if raw.get("flag"):
            command += [str(raw["flag"]), format_label(value)]
        else:
            overrides = {**overrides, raw["field"]: value}
        command += _set_args(overrides)
        command += [str(a) for a in entry.get("extra_args") or []]
        runs.append(SweepRun(label, value, run_name, tuple(command), entry.get("series")))

    names = [r.run_name for r in runs]
    if len(set(names)) != len(names):
        raise ValueError(f"sweep {path} produces duplicate run names: {names}")
    return Sweep(
        name=name,
        base_config=str(raw["base_config"]),
        field=str(raw["field"]),
        runs=tuple(runs),
        description=str(raw.get("description", "")),
        x_label=raw.get("x_label"),
        x_scale=x_scale,
        source=str(path),
        raw=raw,
    )


def index_path(sweep: Sweep, root: str | Path = ".") -> Path:
    return Path(root) / "results" / "sweeps" / sweep.name / "index.json"


def run_is_complete(run_name: str, root: str | Path = ".") -> bool:
    """A run counts as finished when it wrote run_info.json, the last file train() writes."""
    return (Path(root) / "results" / run_name / "run_info.json").exists()


def run_sweep(
    sweep: Sweep,
    root: str | Path = ".",
    only: Sequence[str] | None = None,
    skip_existing: bool = False,
    dry_run: bool = False,
    python: str = sys.executable,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Run the sweep's commands one after another and keep index.json up to date.

    Each run is a separate process, so GPU memory is fully released between runs and a
    crash in one run cannot take the others down. A failed run is recorded and the
    sweep moves on. index.json is rewritten after every run, and entries from earlier
    invocations are kept, so the sweep can be done in pieces with --only.

    Args:
        only: if given, run only the entries whose label or run name is listed.
        skip_existing: do not rerun entries that already have results/<run>/run_info.json.
        dry_run: print the commands and change nothing.

    Returns:
        The index that was written: sweep metadata plus one record per run with its
        status, one of "done", "failed", "skipped_existing" or "not_run".
    """
    root = Path(root)
    path = index_path(sweep, root)
    previous: dict[str, dict[str, Any]] = {}
    if path.exists():
        previous = {r["run_name"]: r for r in json.loads(path.read_text()).get("runs", [])}

    selected = set(only) if only else None
    if selected is not None:
        known = {r.label for r in sweep.runs} | {r.run_name for r in sweep.runs}
        missing = sorted(selected - known)
        if missing:
            raise ValueError(
                f"--only names unknown entries {missing}; known labels: {sorted(known)}"
            )

    records: dict[str, dict[str, Any]] = {}
    for run in sweep.runs:
        records[run.run_name] = {
            "label": run.label,
            "value": run.value,
            "series": run.series,
            "run_name": run.run_name,
            "command": " ".join(["python", *run.command]),
            "status": "done" if run_is_complete(run.run_name, root) else "not_run",
            **{k: previous.get(run.run_name, {}).get(k) for k in ("returncode", "seconds")},
        }
        if previous.get(run.run_name, {}).get("status") == "failed" and not run_is_complete(
            run.run_name, root
        ):
            records[run.run_name]["status"] = "failed"

    def write_index() -> dict[str, Any]:
        index = {
            "sweep": sweep.name,
            "source": sweep.source,
            "base_config": sweep.base_config,
            "field": sweep.field,
            "runs": [records[r.run_name] for r in sweep.runs],
        }
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(index, indent=2) + "\n")
        return index

    for run in sweep.runs:
        if selected is not None and not ({run.label, run.run_name} & selected):
            continue
        record = records[run.run_name]
        if dry_run:
            print(record["command"])
            continue
        if skip_existing and run_is_complete(run.run_name, root):
            record["status"] = "skipped_existing"
            print(f"[sweep {sweep.name}] {run.run_name}: already complete, skipped")
            write_index()
            continue
        print(f"[sweep {sweep.name}] {run.run_name}: {record['command']}", flush=True)
        start = time.perf_counter()
        try:
            returncode = runner([python, *run.command], cwd=root).returncode
        except OSError as err:  # for example a missing script
            print(f"[sweep {sweep.name}] {run.run_name}: could not start: {err}")
            returncode = -1
        record["returncode"] = returncode
        record["seconds"] = round(time.perf_counter() - start, 1)
        # Exit code 0 is not enough: the run must also have written its final file.
        ok = returncode == 0 and run_is_complete(run.run_name, root)
        record["status"] = "done" if ok else "failed"
        print(f"[sweep {sweep.name}] {run.run_name}: {record['status']}", flush=True)
        write_index()
    return write_index()


# --------------------------------------------------------------------------------------
# Summarising finished runs for the figures
# --------------------------------------------------------------------------------------


def _mean_ci(values: Sequence[float], n_resamples: int, seed: int) -> dict[str, float]:
    from tiny_rlhf.trackb.evaluate import bootstrap_interval

    low, high = bootstrap_interval(values, n_resamples, seed)
    return {"mean": sum(values) / len(values), "ci_low": low, "ci_high": high}


def run_summary(run_dir: str | Path, n_resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Final held-out numbers of one run, from its eval_pairs_final.csv.

    Each comes with a 95% percentile bootstrap interval over held-out pairs:
        accuracy            share of pairs with reward margin > 0 (ties count as wrong)
        reward_margin       mean of beta * [(log pi - log pi_ref)(chosen) - same(rejected)]
        chosen_logp_change  mean of log pi(chosen) - log pi_ref(chosen), in nats. Negative
                            means the preferred response became LESS likely, which the
                            margin alone would hide.
        rejected_logp_change  the same for the rejected response
    """
    run_dir = Path(run_dir)
    with (run_dir / "eval_pairs_final.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{run_dir}/eval_pairs_final.csv has no rows")
    margin = [float(r["reward_margin"]) for r in rows]
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "n_pairs": len(rows),
        "accuracy": _mean_ci([float(m > 0) for m in margin], n_resamples, seed),
        "reward_margin": _mean_ci(margin, n_resamples, seed),
        "chosen_logp_change": _mean_ci(
            [float(r["chosen_logp"]) - float(r["ref_chosen_logp"]) for r in rows], n_resamples, seed
        ),
        "rejected_logp_change": _mean_ci(
            [float(r["rejected_logp"]) - float(r["ref_rejected_logp"]) for r in rows],
            n_resamples,
            seed,
        ),
        "interval": "95% percentile bootstrap over held-out pairs",
        "n_resamples": n_resamples,
        "seed": seed,
    }
    info_path = run_dir / "run_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        for key in ("optimizer_steps", "n_train_pairs", "wall_time_s", "peak_gpu_memory_mb"):
            summary[key] = info.get(key)
    drift_path = run_dir / "drift.json"
    if drift_path.exists():
        summary["drift"] = json.loads(drift_path.read_text())
    return summary


def sweep_summary(
    sweep: Sweep, root: str | Path = ".", n_resamples: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """run_summary for every finished run of the sweep; unfinished runs are listed as missing."""
    root = Path(root)
    runs, missing = [], []
    for run in sweep.runs:
        run_dir = root / "results" / run.run_name
        if not (run_dir / "eval_pairs_final.csv").exists():
            missing.append(run.run_name)
            continue
        entry = run_summary(run_dir, n_resamples, seed)
        entry.update(label=run.label, value=run.value, series=run.series, run_name=run.run_name)
        entry["run_dir"] = f"results/{run.run_name}"
        runs.append(entry)
    return {
        "sweep": sweep.name,
        "field": sweep.field,
        "source": sweep.source,
        "runs": runs,
        "missing_runs": missing,
    }
