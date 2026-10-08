#!/usr/bin/env python3
"""Derive and audit a canonically equivalent NFC AMR corpus."""

from __future__ import annotations

import argparse
from itertools import zip_longest
import io
import json
from pathlib import Path
import sys
import unicodedata
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.amr_data import iter_amr_records  # noqa: E402
from automata_drs.penman import sha256_file  # noqa: E402


def _sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _record_changes(source: str, derived: str) -> dict[str, Any]:
    source_records = iter_amr_records(io.StringIO(source))
    derived_records = iter_amr_records(io.StringIO(derived))
    records = 0
    sentence_ordinals: list[int] = []
    graph_ordinals: list[int] = []
    for source_record, derived_record in zip_longest(
        source_records, derived_records
    ):
        if source_record is None or derived_record is None:
            raise ValueError("NFC derivation changed the AMR record count")
        records += 1
        if source_record.ordinal != derived_record.ordinal:
            raise ValueError("NFC derivation changed AMR record order")
        if source_record.sentence != derived_record.sentence:
            sentence_ordinals.append(source_record.ordinal)
        if source_record.graph != derived_record.graph:
            graph_ordinals.append(source_record.ordinal)
    return {
        "records": records,
        "changed_records": sorted(set(sentence_ordinals) | set(graph_ordinals)),
        "changed_sentence_records": sentence_ordinals,
        "changed_graph_records": graph_ordinals,
    }


def build_derivation(source_path: Path, output_path: Path) -> tuple[bytes, dict[str, Any]]:
    source_bytes = source_path.read_bytes()
    source = source_bytes.decode("utf-8")
    derived = unicodedata.normalize("NFC", source)
    derived_bytes = derived.encode("utf-8")
    changes = _record_changes(source, derived)
    canonical_equivalence = unicodedata.normalize(
        "NFD", source
    ) == unicodedata.normalize("NFD", derived)
    gate_passed = (
        unicodedata.is_normalized("NFC", derived)
        and canonical_equivalence
        and changes["records"] > 0
    )
    report = {
        "schema_version": 1,
        "audit": "amr-nfc-corpus-derivation",
        "normalization": {
            "form": "NFC",
            "scope": "complete paired corpus text",
            "canonical_equivalence": canonical_equivalence,
        },
        "source": {
            "file": source_path.name,
            "bytes": len(source_bytes),
            "sha256": _sha256_bytes(source_bytes),
            "records": changes["records"],
        },
        "derived": {
            "file": output_path.name,
            "bytes": len(derived_bytes),
            "sha256": _sha256_bytes(derived_bytes),
            "records": changes["records"],
            "is_nfc": unicodedata.is_normalized("NFC", derived),
        },
        "changes": {
            **changes,
            "source_codepoints": len(source),
            "derived_codepoints": len(derived),
        },
        "implementation": {
            "script": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "gate": {
            "passed": gate_passed,
            "source_preserved": True,
            "records_preserved": True,
            "canonical_equivalence": canonical_equivalence,
        },
    }
    return derived_bytes, report


def _write_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(value)


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    try:
        derived_bytes, report = build_derivation(args.source, args.output)
        if args.verify_existing:
            if args.output.read_bytes() != derived_bytes:
                raise ValueError("stored NFC corpus differs from recomputed derivation")
            stored = json.loads(args.audit.read_text(encoding="utf-8"))
            if stored != report:
                raise ValueError("stored NFC audit differs from recomputed evidence")
            print(f"verified NFC AMR derivation: {args.output}")
        else:
            if args.output.exists() or args.audit.exists():
                raise FileExistsError("refusing to overwrite NFC derivation evidence")
            _write_exclusive(args.output, derived_bytes)
            _write_json_exclusive(args.audit, report)
            print(f"NFC AMR derivation: {args.output}")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(f"changed records: {len(report['changes']['changed_records'])}")
    print(f"gate passed: {report['gate']['passed']}")
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
