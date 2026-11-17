"""Side-by-side examples: the same prompts answered by the base and the trained model.

Usage:
    python scripts/make_examples.py \\
        --base results/trackb_full/trackb_eval/summary_base.json \\
        --trained results/trackb_full/trackb_eval/summary_trained.json

Writes results/<run>/examples.md.

Selection rule (fixed in advance, so the examples are not hand-picked): within each
split, every example falls into one of four outcomes by its on_task check,
    got better   base off task, trained on task
    got worse    base on task, trained off task
    both on task
    both off task
and --per-outcome examples (default 2) are drawn from each outcome with a seeded random
sample over the example ids in sorted order. The file states how many examples fell
into each outcome, so the sample can be read against the real proportions.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tiny_rlhf.trackb.evaluate import read_rows

OUTCOMES = (
    ("got better", False, True),
    ("got worse", True, False),
    ("both on task", True, True),
    ("both off task", False, False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="summary_base.json")
    parser.add_argument("--trained", required=True, help="summary_trained.json")
    parser.add_argument("--out", help="default: results/<run>/examples.md")
    parser.add_argument("--per-outcome", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def fenced(text: str) -> str:
    # A fence longer than any run of backticks in the text, so the text cannot close it.
    return "````text\n" + text.replace("````", "` ` ` `") + "\n````\n"


def flags(scores: dict[str, bool]) -> str:
    on = [name for name, value in scores.items() if value]
    return ", ".join(on) if on else "none"


def main() -> None:
    args = parse_args()
    base_path, trained_path = Path(args.base), Path(args.trained)
    base, trained = json.loads(base_path.read_text()), json.loads(trained_path.read_text())
    # Next to the trained run, which may differ from the base evaluation's run.
    out = Path(args.out) if args.out else trained_path.parent.parent / "examples.md"
    lines = [
        "# Track B: the same prompts, before and after training\n",
        f"Base: `{base.get('model')}`. Trained: the same model with the adapter "
        f"`{trained.get('adapter')}`. Greedy decoding.\n",
        "**How these were chosen.** Within each split, examples are grouped by what happened "
        "to the `on_task` check (got better, got worse, both on task, both off task). "
        f"{args.per_outcome} per group are drawn with a seeded random sample "
        f"(seed {args.seed}) over the sorted example ids. Nothing was picked by hand. The "
        "counts next to each heading show how common each outcome was.\n",
        "Checks listed under each answer are the ones that came out true "
        "(see `src/tiny_rlhf/trackb/evaluate.py` for definitions).\n",
    ]
    for split in [s for s in base["files"] if s in trained["files"]]:
        base_rows = {r["id"]: r for r in read_rows(base_path.parent / base["files"][split])}
        trained_rows = {
            r["id"]: r for r in read_rows(trained_path.parent / trained["files"][split])
        }
        ids = sorted(set(base_rows) & set(trained_rows))
        rng = random.Random(f"examples:{args.seed}:{split}")
        lines.append(f"\n## {split} ({len(ids)} examples)\n")
        for name, base_on, trained_on in OUTCOMES:
            members = [
                i
                for i in ids
                if base_rows[i]["scores"]["on_task"] == base_on
                and trained_rows[i]["scores"]["on_task"] == trained_on
            ]
            lines.append(f"\n### {name}: {len(members)} of {len(ids)}\n")
            for example_id in rng.sample(members, min(args.per_outcome, len(members))):
                b, t = base_rows[example_id], trained_rows[example_id]
                lines += [
                    f"\n**{example_id}** ({b['kind']}, task `{b['task_category']}`, "
                    f"style `{b['doc_style']}`, injection `{b['injection_type']}`)\n",
                    "Prompt:\n",
                    fenced(b["prompt"]),
                    f"Expected answer: `{json.dumps(b['expected_answer'])}`. "
                    f"Planted marker: `{b['injection_marker']}`.\n",
                    "Base model:\n",
                    fenced(b["response"]),
                    f"Checks true: {flags(b['scores'])}\n",
                    "Trained model:\n",
                    fenced(t["response"]),
                    f"Checks true: {flags(t['scores'])}\n",
                ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
