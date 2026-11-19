"""Rescore saved Track B generations with strict task correctness, without a GPU.

The lenient check (checks.task_correct) accepts an answer that appears anywhere in the
response, so a numbered list passes "how many items" and a list of every item passes
"which item is cheapest". checks.task_correct_strict closes that. Generations are saved
per row under results/<run>/trackb_eval/<split>_<label>.jsonl, so every evaluation can
be rescored from its rows: the model is not run again.

Outputs are new files; the original summaries are left untouched as the record of what
was reported with the lenient rule:
    results/<run>/trackb_eval/rescored_<label>.json    per split and group, lenient and
                                                       strict side by side
    results/trackb_rescored_compare.json               base against each run, paired
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tiny_rlhf.trackb.checks import score
from tiny_rlhf.trackb.evaluate import (
    DEFAULT_SPLITS,
    compare_rows,
    rate_with_interval,
    read_rows,
    summarize_rows,
)

CLEAN = (
    "on_task",
    "on_task_strict",
    "task_correct",
    "task_correct_strict",
    "refused",
    "false_flag",
)
RESCORE_METRICS: dict[str, tuple[str, ...]] = {
    "injected": (
        "on_task",
        "on_task_strict",
        "task_correct",
        "task_correct_strict",
        "followed_injection",
        "refused",
    ),
    "ordinary": CLEAN,
    "hard_negative": CLEAN,
    "no_injection": CLEAN,
}
COMPARE_METRICS: dict[str, tuple[str, ...]] = {
    "injected": ("on_task_strict", "on_task", "followed_injection"),
    "no_injection": ("task_correct_strict", "task_correct", "on_task_strict"),
}
# The checks whose saved values must be reproduced exactly before anything is rescored.
LENIENT_KEYS = (
    "followed_injection",
    "task_correct",
    "refused",
    "flagged_document",
    "false_flag",
    "on_task",
)


def rescore_row(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The row with every check recomputed from its saved response, and the names of
    lenient checks whose recomputed value differs from the saved one."""
    fresh = score(row["response"], row)
    saved = row["scores"]
    mismatches = [k for k in LENIENT_KEYS if k in saved and bool(saved[k]) != bool(fresh[k])]
    scores = dict(saved)
    scores.update(fresh)
    return {**row, "scores": scores}, mismatches


def load_rescored(summary_path: Path, splits: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Rescore every split listed in a summary_<label>.json. Returns rows per split and a
    count of lenient mismatches (non-zero means the checks changed since the run)."""
    summary = json.loads(summary_path.read_text())
    wanted = splits or tuple(summary["files"])
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    mismatches: dict[str, int] = defaultdict(int)
    for split in wanted:
        if split not in summary["files"]:
            continue
        rows = []
        for row in read_rows(summary_path.parent / summary["files"][split]):
            new, bad = rescore_row(row)
            rows.append(new)
            for key in bad:
                mismatches[key] += 1
        rows_by_split[split] = rows
    return {"summary": summary, "rows": rows_by_split, "mismatches": dict(mismatches)}


def by_task_category(
    rows_by_split: dict[str, list[dict[str, Any]]], n_resamples: int, seed: int
) -> dict[str, Any]:
    """Lenient and strict task-correct per task category, over the five test splits."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split in DEFAULT_SPLITS:
        for row in rows_by_split.get(split, []):
            groups[row["task_category"]].append(row)
    return {
        task: {
            "n": len(members),
            "task_correct": rate_with_interval(
                [r["scores"]["task_correct"] for r in members], n_resamples, seed
            ),
            "task_correct_strict": rate_with_interval(
                [r["scores"]["task_correct_strict"] for r in members], n_resamples, seed
            ),
        }
        for task, members in sorted(groups.items())
    }


def rescored_summary(
    loaded: dict[str, Any], run: str, label: str, n_resamples: int, seed: int
) -> dict[str, Any]:
    rows = loaded["rows"]
    pooled = [r for split in DEFAULT_SPLITS for r in rows.get(split, [])]
    return {
        "run": run,
        "label": label,
        "scoring": "lenient and strict task correctness; see tiny_rlhf.trackb.checks",
        "examples_sha256": loaded["summary"].get("examples_sha256"),
        "model": loaded["summary"].get("model"),
        "adapter": loaded["summary"].get("adapter"),
        "evaluated_at_git_sha": loaded["summary"].get("git_sha"),
        "lenient_mismatches_on_rescoring": loaded["mismatches"],
        "interval": "95% percentile bootstrap over examples",
        "n_resamples": n_resamples,
        "seed": seed,
        "splits": {
            split: summarize_rows(split_rows, n_resamples, seed, RESCORE_METRICS)
            for split, split_rows in rows.items()
        },
        "pooled_test_splits": summarize_rows(pooled, n_resamples, seed, RESCORE_METRICS),
        "by_task_category": by_task_category(rows, n_resamples, seed),
    }


def compare_runs(
    base: dict[str, Any], trained: dict[str, Any], n_resamples: int, seed: int
) -> dict[str, Any]:
    """Paired base-vs-trained changes per split and pooled over the five test splits."""
    out: dict[str, Any] = {"splits": {}}
    for split, base_rows in base["rows"].items():
        if split in trained["rows"]:
            out["splits"][split] = compare_rows(
                base_rows, trained["rows"][split], n_resamples, seed, COMPARE_METRICS
            )
    pooled_base = [r for s in DEFAULT_SPLITS for r in base["rows"].get(s, [])]
    pooled_trained = [r for s in DEFAULT_SPLITS for r in trained["rows"].get(s, [])]
    out["pooled_test_splits"] = compare_rows(
        pooled_base, pooled_trained, n_resamples, seed, COMPARE_METRICS
    )
    return out


# Suffix of a run retrained only to get its adapter back for the external evaluation (M9).
# Such a rerun is compared with the saved run it repeats, by tiny_rlhf.external.reproduce,
# not listed here as a run of its own: no job writes its rescored file.
RERUN_SUFFIX = "_m9"


def discover_runs(results: Path) -> list[str]:
    """Every run under results/ with a saved trained evaluation, first run first.

    Reruns named <run>_m9 are left out (see RERUN_SUFFIX).
    """
    found = sorted(
        p.parent.parent.name
        for p in results.glob("*/trackb_eval/summary_trained.json")
        if not p.parent.parent.name.endswith(RERUN_SUFFIX)
    )
    return sorted(found, key=lambda run: (run != "trackb_full", run))


def rescore_all(
    results: Path,
    base_run: str = "trackb_full",
    runs: list[str] | None = None,
    n_resamples: int = 2000,
    seed: int = 0,
    write_runs: list[str] | None = None,
) -> dict[str, Any]:
    """Rescore the base evaluation and every run; write the new files; return the comparison.

    Every run is rescored and compared. `write_runs` limits which runs' rescored files are
    written (None: all; the base run's rescored_base.json only when the base run is listed),
    so a queue job that commits one run's file does not leave other runs' rewritten files
    uncommitted in the working tree. The comparison file is always written.
    """
    runs = discover_runs(results) if runs is None else runs
    base_path = results / base_run / "trackb_eval" / "summary_base.json"
    base = load_rescored(base_path)
    written = []
    if write_runs is None or base_run in write_runs:
        base_out = base_path.parent / "rescored_base.json"
        base_out.write_text(
            json.dumps(rescored_summary(base, base_run, "base", n_resamples, seed), indent=2) + "\n"
        )
        written.append(base_out)
    comparison: dict[str, Any] = {
        "base": str(base_path.relative_to(results.parent)),
        "scoring": "strict task correctness: tiny_rlhf.trackb.checks.task_correct_strict",
        "interval": "paired 95% percentile bootstrap over the same examples",
        "n_resamples": n_resamples,
        "seed": seed,
        "lenient_mismatches_on_rescoring": {"base": base["mismatches"]},
        "by_task_category": {
            "base": by_task_category(base["rows"], n_resamples, seed),
        },
        "runs": {},
        "not_compared": [],
    }
    for run in runs:
        path = results / run / "trackb_eval" / "summary_trained.json"
        if not path.exists():
            comparison["not_compared"].append(f"{run}: no trained evaluation")
            continue
        trained = load_rescored(path)
        if trained["summary"].get("examples_sha256") != base["summary"].get("examples_sha256"):
            comparison["not_compared"].append(f"{run}: scored different prompts than the base")
            continue
        if write_runs is None or run in write_runs:
            out = path.parent / "rescored_trained.json"
            out.write_text(
                json.dumps(rescored_summary(trained, run, "trained", n_resamples, seed), indent=2)
                + "\n"
            )
            written.append(out)
        comparison["lenient_mismatches_on_rescoring"][run] = trained["mismatches"]
        comparison["by_task_category"][run] = by_task_category(trained["rows"], n_resamples, seed)
        comparison["runs"][run] = compare_runs(base, trained, n_resamples, seed)
    compare_path = results / "trackb_rescored_compare.json"
    compare_path.write_text(json.dumps(comparison, indent=2) + "\n")
    written.append(compare_path)
    comparison["written"] = [str(p) for p in written]
    return comparison
