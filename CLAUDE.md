# CLAUDE.md

## What this project is
Preference learning on a small language model, built from scratch: a reward model, DPO written by hand and checked against the TRL library, then applied to one measurable behavior.

The research, design and milestone plan is in `docs/PLAN.md`. Read it fully before doing anything.

## Who you are working with
Varshith owns this repo. He is building it to learn the material deeply and to show it publicly, so he must be able to explain every design decision in a technical conversation without notes.

## How to work
1. Milestones in the order in `docs/PLAN.md`. When a milestone is blocked only on something the owner must do (for example a GPU run on Colab), keep building the code, tests and CPU-scale checks for later milestones instead of stopping, and stop only when nothing more can be done without him. A milestone is ticked off only when its "done when" criteria are actually met. (Changed by the owner on 2026-11-17.)
2. Teach before you build. At the start of each milestone, explain the concepts it needs with intuition and math together and two or three worked examples, using plain text math (no LaTeX). Then build.
3. Do not quiz after each milestone and do not pause between milestones. Keep building. Quiz him once, at the end of the project, one question at a time, waiting for each answer. Use the question list at the end of the plan and add your own. (Changed by the owner on 2026-11-17.)
4. Keep `docs/PLAN.md` current: tick off milestones, and record changed decisions and why in a "Decision log" section at the bottom.
5. The plan was written in November 2026. Before using any library API, dataset location or package name from it, check the current documentation. If something changed, follow the current docs and note it in the decision log.

## Commits
- Commit as the work actually happens: one commit per logical step (a module, its tests, a figure) with a plain message. Several small commits per milestone, pushed when that milestone's tests pass. Never one giant commit.
- Do not add AI co-author trailers or "Generated with" lines to commits or pull requests. This is the owner's standing instruction.
- Never alter commit dates or rewrite history to change when work appears to have happened.
- Keep `main` current: fast-forward `main` to the working branch regularly, at least when a milestone or a batch of results lands and at least once per day of active work, and only to a commit on which lint and the full test suite pass. Never force-push `main`. (Owner's request, 2026-11-19.)

## Honesty rules
- No number, chart or claim goes in the README unless a script in this repo produced it and the output is saved under `results/`.
- If something is untested, unfinished or only works on toy data, the README says so.
- Numbers from papers or other tools are quoted with their source and never blended with ours.

## Visuals
Every milestone that produces a result also produces a figure or demo under `assets/`, made by a script under `scripts/`. The README opens with one strong visual, then three sentences of explanation, then results. One consistent style. The plan lists the visuals.

## Engineering conventions
Python 3.11+, `pyproject.toml`, `ruff`, `pytest`, GitHub Actions running lint and tests on every push. Type hints on public functions. Fixed, recorded seeds. Config in files. Keep the repo private until the final milestone's checklist is done.
