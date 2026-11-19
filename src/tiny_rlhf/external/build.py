"""Build the external evaluation set from the pinned BIPIA release.

Usage:
    python -m tiny_rlhf.external.build --config configs/external_bipia.yaml
    python -m tiny_rlhf.external.build --config ... --parquet path/to/downloaded.parquet

Downloads the pinned parquet file (about 4.7 MB) into the Hugging Face cache unless
--parquet names a copy, checks its sha256, and writes to the config's out_dir:
    bipia_eval.jsonl   the examples, in the Track B example format (bipia.py)
    provenance.json    source, licence, settings, selection statistics and fingerprints
Only these small derived files are kept in the repository, not the source file. Running
the build again with the same config and source writes byte-identical files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tiny_rlhf.external import bipia
from tiny_rlhf.trackb.evaluate import examples_fingerprint

EXAMPLES_FILE = "bipia_eval.jsonl"
PROVENANCE_FILE = "provenance.json"


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text())


def provenance(
    config: dict[str, Any], stats: dict[str, Any], examples: list[dict[str, Any]], jsonl: Path
) -> dict[str, Any]:
    """Where the examples came from and how they were chosen (no timestamps, so a
    rebuild from the same inputs gives the same file)."""
    return {
        "source": {
            "dataset": bipia.REPO_ID,
            "revision": bipia.REVISION,
            "file": bipia.PARQUET,
            "file_sha256": bipia.PARQUET_SHA256,
            "upstream": bipia.UPSTREAM,
            "upstream_files": [
                "benchmark/email/test.jsonl",
                "benchmark/table/test.jsonl",
            ],
            "paper": "Yi et al. 2023, arXiv:2312.14197",
            "licence": bipia.LICENCE,
        },
        "derived_licence": "CC BY-SA 4.0 (share-alike, as the mirror)",
        "builder": "python -m tiny_rlhf.external.build",
        "config": config,
        "selection": stats,
        "examples_file": jsonl.name,
        "examples_file_sha256": bipia.sha256_file(jsonl),
        "examples_sha256": examples_fingerprint({bipia.SPLIT: examples}),
    }


def build(
    config: dict[str, Any], parquet: Path | None = None, out_dir: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    """Write the examples and their provenance; return (out_dir, provenance).

    out_dir overrides the config's out_dir without changing the config recorded in
    provenance.json, so a build into another directory writes the same files."""
    path = parquet if parquet is not None else bipia.download_parquet()
    rows = bipia.read_rows(path)
    examples, stats = bipia.select_examples(
        rows,
        seed=config["seed"],
        per_task=config["per_task"],
        max_context_chars=config["max_context_chars"],
        positions=tuple(config["positions"]),
    )
    out_dir = Path(config["out_dir"]) if out_dir is None else out_dir
    jsonl = out_dir / EXAMPLES_FILE
    bipia.write_jsonl(examples, jsonl)
    record = provenance(config, stats, examples, jsonl)
    (out_dir / PROVENANCE_FILE).write_text(json.dumps(record, indent=2) + "\n")
    return out_dir, record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="e.g. configs/external_bipia.yaml")
    parser.add_argument("--parquet", help="a local copy of the pinned file (sha256 checked)")
    parser.add_argument("--out-dir", help="overrides the config's out_dir")
    args = parser.parse_args()
    config = load_config(args.config)
    out_dir, record = build(
        config,
        Path(args.parquet) if args.parquet else None,
        Path(args.out_dir) if args.out_dir else None,
    )
    sel = record["selection"]
    print(
        f"wrote {out_dir / EXAMPLES_FILE}: {sel['n_injected']} attacked and "
        f"{sel['n_clean']} clean examples from {sel['n_attacks_scored']} of "
        f"{sel['n_attacks_in_source']} attacks"
    )


if __name__ == "__main__":
    main()
