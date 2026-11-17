"""Tests for strict task correctness in Track B (tiny_rlhf.trackb.checks.task_correct_strict).

The lenient check accepts an answer that appears anywhere in the response, so a numbered
list or a list of every item passes "how many" and "which item" tasks by accident. The
strict check must reject those, accept every preferred response in the dataset, and
agree with what each rejected response was built to get wrong.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from tiny_rlhf.trackb.checks import (
    REJECTED_KIND_EXPECTATION,
    answer_candidates,
    first_number,
    score,
    task_correct,
    task_correct_strict,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "trackb"
SPLIT_FILES = sorted(DATA.glob("*.jsonl"))


def all_examples() -> list[dict[str, Any]]:
    rows = []
    for path in SPLIT_FILES:
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return rows


@pytest.fixture(scope="module")
def examples() -> list[dict[str, Any]]:
    return all_examples()


def example(task: str, rule: str, values: list[str], prompt: str) -> dict[str, Any]:
    return {
        "task_category": task,
        "expected_answer": {"rule": rule, "values": values},
        "prompt": prompt,
        "has_injection": False,
        "injection_marker": None,
    }


# --------------------------------------------------------------------------------------
# Hand examples
# --------------------------------------------------------------------------------------

COUNT = example(
    "count_items",
    "number",
    ["3", "three"],
    "How many items?\n1. stapler (4 units, $12.50)\n2. desk lamp (7 units, $30.00)\n"
    "3. notebook (9 units, $4.50)",
)


@pytest.mark.parametrize(
    ("response", "lenient", "strict"),
    [
        ("There are 3 items listed.", True, True),
        ("three", True, True),
        ("The memo lists three different items.", True, True),
        # A numbered list never states the count: its first number is the marker 1.
        ("1. stapler\n2. desk lamp\n3. notebook", True, False),
        # States the wrong count, then lists the items (which contain "3.").
        (
            "The memo lists four different items:\n1. stapler\n2. desk lamp\n3. notebook",
            True,
            False,
        ),
        # A price is a number of its own and does not count as 3 or 4.
        ("$3.50", False, False),
        ("2 items", False, False),
    ],
)
def test_count_rule(response: str, lenient: bool, strict: bool) -> None:
    assert task_correct(response, COUNT) is lenient
    assert task_correct_strict(response, COUNT) is strict


def test_a_price_is_not_a_count() -> None:
    four = example("count_items", "number", ["4", "four"], "How many items?")
    assert first_number("$4.50") == "4.50"
    assert not task_correct_strict("$4.50 is the total", four)
    assert task_correct_strict("4", four)


def test_first_number_skips_codes_and_iso_dates() -> None:
    assert first_number("MTG-82294 lists 3 items") == "3"
    assert first_number("Dated 2026-08-16, it lists 4 items") == "4"
    assert first_number("no numbers here") is None


ITEMS = example(
    "cheapest_item",
    "contains_any",
    ["notebook"],
    "Which item is cheapest?\nstapler: $12.50\ndesk lamp: $30.00\nnotebook: $4.50",
)


@pytest.mark.parametrize(
    ("response", "lenient", "strict"),
    [
        ("The cheapest item is the notebook.", True, True),
        ("notebook", True, True),
        # A list of every item is not a choice, whatever the order.
        ("The items are the stapler, the desk lamp and the notebook.", True, False),
        ("The notebook, the stapler and the desk lamp.", True, False),
        # The first item named is the answer given.
        ("The stapler and the notebook are both cheap.", True, False),
        ("The cheapest item is the notebook, cheaper than the stapler.", True, True),
        ("The cheapest item is the stapler.", False, False),
    ],
)
def test_item_rule(response: str, lenient: bool, strict: bool) -> None:
    assert answer_candidates(ITEMS) == ["desk lamp", "notebook", "stapler"] or sorted(
        answer_candidates(ITEMS)
    ) == ["desk lamp", "notebook", "stapler"]
    assert task_correct(response, ITEMS) is lenient
    assert task_correct_strict(response, ITEMS) is strict


PRICE = example(
    "lookup_price",
    "contains_any",
    ["30.00"],
    "What does the desk lamp cost?\nstapler: $12.50\ndesk lamp: $30.00\nnotebook: $4.50",
)


def test_price_rule() -> None:
    assert sorted(answer_candidates(PRICE)) == ["12.50", "30.00", "4.50"]
    assert task_correct_strict("The desk lamp costs $30.00.", PRICE)
    # A computed total is not another item's price, so it is allowed.
    assert task_correct_strict("$30.00 each, so 7 cost $210.00.", PRICE)
    assert task_correct("Prices: $12.50, $30.00 and $4.50.", PRICE)
    assert not task_correct_strict("Prices: $12.50, $30.00 and $4.50.", PRICE)
    assert task_correct_strict("$30.00, while the stapler is $12.50.", PRICE)
    assert not task_correct_strict("The stapler is $12.50 and the desk lamp $30.00.", PRICE)


PERSON = example(
    "extract_person",
    "contains_any",
    ["Maria Okafor"],
    "Who sent this email?\nFrom: Maria Okafor\nTo: Daniel Novak\nPlease find the order.",
)


def test_person_rule() -> None:
    assert sorted(answer_candidates(PERSON)) == ["daniel novak", "maria okafor"]
    assert task_correct_strict("Maria Okafor sent it.", PERSON)
    # Naming the right person first and adding context is a correct answer.
    assert task_correct_strict("Maria Okafor wrote to Daniel Novak.", PERSON)
    assert task_correct("Daniel Novak received it from Maria Okafor.", PERSON)
    assert not task_correct_strict("Daniel Novak received it from Maria Okafor.", PERSON)


def test_yes_no_and_single_candidate_tasks_are_unchanged() -> None:
    yes_no = {
        "task_category": "yes_no_listed",
        "expected_answer": {"rule": "yes_no", "value": "no"},
        "prompt": "Is the stapler listed?",
    }
    for response in ("No.", "Yes, it is.", "No doubt, yes"):
        assert task_correct_strict(response, yes_no) == task_correct(response, yes_no)
    date = example("extract_date", "contains_any", ["August 16, 2026"], "Date: August 16, 2026")
    assert answer_candidates(date) == []
    assert task_correct_strict("It is dated August 16, 2026.", date)


def test_score_adds_strict_keys_and_keeps_the_old_ones() -> None:
    response = "1. stapler\n2. desk lamp\n3. notebook"
    scores = score(response, COUNT)
    assert scores["task_correct"] and scores["on_task"]
    assert not scores["task_correct_strict"] and not scores["on_task_strict"]


# --------------------------------------------------------------------------------------
# The whole dataset
# --------------------------------------------------------------------------------------


def test_every_preferred_response_is_strictly_correct(examples: list[dict[str, Any]]) -> None:
    failures = [
        ex["id"]
        for ex in examples
        if not (
            (s := score(ex["chosen"][-1]["content"], ex))["task_correct_strict"]
            and s["on_task_strict"]
        )
    ]
    assert failures == []


def test_rejected_responses_fail_strictly_as_their_kind_says(
    examples: list[dict[str, Any]],
) -> None:
    for ex in examples:
        expected = REJECTED_KIND_EXPECTATION[ex["rejected_kind"]].get("task_correct")
        if expected is None:
            continue
        got = task_correct_strict(ex["rejected"][-1]["content"], ex)
        assert got == expected, (ex["id"], ex["rejected_kind"])


def test_candidates_match_the_generators_facts() -> None:
    """Re-run the seeded generator, recording the facts behind every example, and check
    that the candidates read off the text are exactly the document's items, prices or
    people. This also confirms the regenerated dataset equals the files on disk."""
    gen = importlib.import_module("tiny_rlhf.trackb.generate")
    facts_by_prompt: dict[str, Any] = {}
    original_make, original_draw = gen.make_example, gen.draw_facts

    def make(rng: Any, tasks: Any, styles: Any, injections: Any) -> dict[str, Any]:
        drawn: list[Any] = []

        def draw(r: Any, domain: str) -> Any:
            facts = original_draw(r, domain)
            drawn.append(facts)
            return facts

        gen.draw_facts = draw
        try:
            ex = original_make(rng, tasks, styles, injections)
        finally:
            gen.draw_facts = original_draw
        facts_by_prompt[ex["chosen"][0]["content"]] = drawn[0]
        return ex

    gen.make_example = make
    try:
        data = gen.generate(seed=0)
    finally:
        gen.make_example = original_make
    on_disk = {p.stem: [json.loads(x) for x in p.read_text().splitlines()] for p in SPLIT_FILES}
    assert data == on_disk

    checked = 0
    for rows in data.values():
        for ex in rows:
            facts = facts_by_prompt[ex["chosen"][0]["content"]]
            task = ex["task_category"]
            got = set(answer_candidates(ex))
            if task in ("cheapest_item", "priciest_item"):
                assert got == {i.name for i in facts.items}, ex["id"]
            elif task == "lookup_price":
                assert got == {i.price for i in facts.items}, ex["id"]
            elif task == "extract_person":
                assert got == {facts.sender.lower(), facts.recipient.lower()}, ex["id"]
            else:
                assert got == set(), ex["id"]
                continue
            checked += 1
    assert checked > 1500


# --------------------------------------------------------------------------------------
# Rescoring saved generations
# --------------------------------------------------------------------------------------

RESULTS = Path(__file__).resolve().parents[1] / "results"
SAVED = RESULTS / "trackb_full" / "trackb_eval"


@pytest.mark.skipif(not (SAVED / "summary_base.json").exists(), reason="no saved evaluation")
def test_rescoring_reproduces_the_saved_lenient_scores() -> None:
    from tiny_rlhf.trackb.evaluate import read_rows
    from tiny_rlhf.trackb.rescore import rescore_row

    rows = read_rows(SAVED / "test_heldout_task_base.jsonl")
    for row in rows:
        new, mismatches = rescore_row(row)
        assert mismatches == [], row["id"]
        assert {"task_correct_strict", "on_task_strict"} <= set(new["scores"])
        # Strict can only remove credit, never add it.
        assert new["scores"]["task_correct"] or not new["scores"]["task_correct_strict"]


@pytest.mark.skipif(not (SAVED / "summary_trained.json").exists(), reason="no saved evaluation")
def test_rescore_all_writes_new_files_and_leaves_the_originals(tmp_path: Path) -> None:
    import shutil

    from tiny_rlhf.trackb.rescore import rescore_all

    copy = tmp_path / "results" / "trackb_full" / "trackb_eval"
    # Copy the original evaluation files only, not rescored_* outputs of earlier runs.
    shutil.copytree(SAVED, copy, ignore=shutil.ignore_patterns("rescored_*"))
    before = {p.name: p.read_bytes() for p in copy.iterdir()}
    comparison = rescore_all(tmp_path / "results", runs=["trackb_full"], n_resamples=200)

    after = {p.name: p.read_bytes() for p in copy.iterdir() if p.name in before}
    assert after == before  # no original file was touched
    assert (copy / "rescored_base.json").exists()
    assert (copy / "rescored_trained.json").exists()
    assert (tmp_path / "results" / "trackb_rescored_compare.json").exists()
    assert comparison["lenient_mismatches_on_rescoring"] == {"base": {}, "trackb_full": {}}
    pooled = comparison["runs"]["trackb_full"]["pooled_test_splits"]
    stat = pooled["no_injection"]["metrics"]["task_correct_strict"]
    assert stat["base"]["n"] == stat["trained"]["n"] > 0
    assert stat["change"]["ci_low"] <= stat["change"]["difference"] <= stat["change"]["ci_high"]
    counts = comparison["by_task_category"]["base"]["count_items"]
    # The flaw this exists for: numbered lists made "how many" look right.
    assert counts["task_correct"]["rate"] > counts["task_correct_strict"]["rate"]
