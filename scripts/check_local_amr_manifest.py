#!/usr/bin/env python3
"""Recompute the compact local AMR manifest without parsing every graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.amr_data import iter_amr_records  # noqa: E402
from automata_drs.penman import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify local AMR file sizes, hashes, and sentence/graph counts."
    )
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path("data/amr/manifest.json"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="directory containing the manifest file names instead of manifest.parent",
    )
    args = parser.parse_args()

    value = json.loads(args.manifest.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        parser.error("unsupported AMR manifest schema")
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        parser.error("manifest has no files")

    failures: list[str] = []
    for file_name, expected in files.items():
        path = (args.data_root or args.manifest.parent) / file_name
        if not path.is_file():
            failures.append(f"missing file: {path}")
            continue
        observed_size = path.stat().st_size
        observed_sha256 = sha256_file(path)
        with path.open(encoding="utf-8") as source:
            observed_records = sum(1 for _ in iter_amr_records(source))
        comparisons = {
            "size_bytes": observed_size,
            "sha256": observed_sha256,
            "examples": observed_records,
        }
        mismatches = {
            key: {"expected": expected.get(key), "observed": observed}
            for key, observed in comparisons.items()
            if expected.get(key) != observed
        }
        if mismatches:
            failures.append(f"{file_name}: {json.dumps(mismatches, sort_keys=True)}")
        else:
            print(
                f"{file_name}: records={observed_records} "
                f"bytes={observed_size} sha256={observed_sha256}"
            )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("manifest gate passed: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
