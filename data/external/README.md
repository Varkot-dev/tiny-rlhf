# External evaluation data (M9)

This directory holds small files derived from a public prompt-injection benchmark. They are used only to evaluate models, never to train them.

## bipia/

| File | What it is |
|---|---|
| `bipia_eval.jsonl` | 773 evaluation examples: 648 documents with an attack in them and 125 clean versions of the same documents, in the Track B example format |
| `provenance.json` | Source, pinned revision and file hash, licence, build settings, selection statistics and fingerprints of the examples |

### Source

- **Benchmark:** BIPIA, "Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models" (Yi et al., 2023, arXiv:2312.14197). Code and data at `microsoft/BIPIA` on GitHub, MIT licence.
- **Copy used:** the Hugging Face mirror `geodesic-research/bipia`, file `bipia/train-00000-of-00001.parquet` at commit `003f58372e1696726b7cddde274a74bee0b1938a` (sha256 `2ec55ad5...8e7c0d`, full value in `provenance.json`). The mirror's card says it was built from `microsoft/BIPIA` commit `a004b69ec0dd` (files `benchmark/email/test.jsonl` and `benchmark/table/test.jsonl` for the parts used here), and that its head can change when the builder re-pushes, which is why the commit and the file hash are pinned.
- **Licence:** the mirror is published under CC BY-SA 4.0. The files in this directory are adapted from it and are therefore shared under the same licence, CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/), with attribution to the BIPIA authors and to the mirror. The rest of this repository is not covered by that licence.
- **Underlying documents:** according to the BIPIA paper, the emails come from OpenAI Evals and the tables from WikiTableQuestions. Their licences were not re-checked here beyond the mirror's card.

### What was changed

- Only BIPIA's email and table tasks are used, and only 72 of its 75 text attacks (those a rule can score; see `docs/EXTERNAL_EVAL.md`). The code task and code attacks are not used.
- Tables longer than 2,400 characters are left out (75 of 100 tables remain); all 50 emails are kept.
- Each document's clean version was rebuilt by cutting the attack line out; the rebuild was checked to give the same text from every row of a document.
- The question and document are laid out with Track B's prompt layouts, not BIPIA's prompt templates. Email questions lose their leading "Q: ".
- A balanced, seeded sample: every scored attack at each of the three positions (start, middle, end) on two tables and one email, so 72 x 3 x 3 = 648 attacked examples.
- Each example records which rule scores it (`detector`), the BIPIA row it came from (`source.id`), and a matching rule for the reference answer (`expected_answer`, null for the 26 emails whose reference answer is "unknown").

### Rebuilding

```bash
python -m tiny_rlhf.external.build --config configs/external_bipia.yaml
```

This downloads the pinned file (about 4.7 MB) into the Hugging Face cache, checks its sha256, and rewrites both files. With the same config and the same checks the output is byte-identical (checked on 2026-11-19). `tests/test_external.py` checks the committed files against `provenance.json` and against the current checks.

### Intended use and limits

- For measuring whether a model follows instructions planted in documents. Not for training: a model trained on these examples could no longer be evaluated on them.
- The examples are a sample of one benchmark in English, with two document types. They contain BIPIA's attack texts, some of which ask for scam or marketing content; they are kept as published.
