"""M9: evaluating on prompt injections written by someone else.

Track B's documents and planted instructions all come from this repo's own templates
(tiny_rlhf.trackb.generate). This package asks whether a model trained on them resists
injections it was never templated on, using BIPIA (Yi et al., 2023, arXiv:2312.14197),
a public benchmark of indirect prompt injection in real emails and web tables.

    bipia       download the pinned BIPIA release, rebuild each document without its
                attack, choose a balanced sample and write it in the Track B example
                format (python -m tiny_rlhf.external.build)
    detectors   rule-based checks for whether a response carried out an attack
    evaluate    generate, score, summarise with bootstrap intervals, compare two models
    reproduce   check that a retrained adapter matches the saved run it stands in for

docs/EXTERNAL_EVAL.md describes the dataset, what was adapted, how scoring works, and
what a result can and cannot show.
"""
