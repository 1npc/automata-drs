#!/usr/bin/env python3
"""Replay a PENMAN corpus through a pinned tokenizer and generic constraint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import (  # noqa: E402
    ByteLevelTokenConstraintController,
    ByteLevelVocabulary,
    PENMANConstraintState,
    load_penman_records,
    sha256_file,
)


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2)
        destination.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", type=Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    parser.add_argument("--expect-corpus-sha256")
    parser.add_argument("--expect-records", type=int)
    parser.add_argument("--require-coverage", type=float, default=1.0)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--source-repository-url")
    parser.add_argument("--source-revision")
    parser.add_argument("--source-file")
    parser.add_argument("--source-license")
    args = parser.parse_args()

    if not 0.0 <= args.require_coverage <= 1.0:
        parser.error("--require-coverage must be between zero and one")
    if args.expect_records is not None and args.expect_records <= 0:
        parser.error("--expect-records must be positive")
    if args.examples < 0:
        parser.error("--examples must be non-negative")
    observed_hash = sha256_file(args.corpus)
    if (
        args.expect_corpus_sha256 is not None
        and observed_hash != args.expect_corpus_sha256
    ):
        parser.error(
            "corpus SHA-256 mismatch: "
            f"expected {args.expect_corpus_sha256}, observed {observed_hash}"
        )

    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        parser.error(f"transformers is required: {error}")

    load_kwargs: dict[str, Any] = {"revision": args.revision}
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = str(args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    try:
        vocabulary = ByteLevelVocabulary.from_tokenizer(tokenizer)
    except ValueError as error:
        parser.error(str(error))
    controller = ByteLevelTokenConstraintController(
        vocabulary,
        incomplete_utf8_is_extendable=(
            PENMANConstraintState.incomplete_utf8_is_extendable
        ),
    )
    records = load_penman_records(args.corpus)
    if args.expect_records is not None and len(records) != args.expect_records:
        parser.error(
            f"expected {args.expect_records} records, found {len(records)}"
        )

    accepted = 0
    exact_bytes = 0
    token_lengths: list[int] = []
    failures: list[dict[str, Any]] = []
    eos_token_id = min(vocabulary.eos_token_ids)
    for record in records:
        encoded = tokenizer(record.surface, add_special_tokens=False)
        values = encoded["input_ids"]
        if values and isinstance(values[0], list):
            if len(values) != 1:
                raise RuntimeError("tokenizer returned a non-singleton batch")
            values = values[0]
        token_ids = [int(token_id) for token_id in values]
        token_lengths.append(len(token_ids))
        fragments = [vocabulary.fragments.get(token_id) for token_id in token_ids]
        missing = [
            token_id
            for token_id, fragment in zip(token_ids, fragments, strict=True)
            if fragment is None
        ]
        reconstructed = (
            b""
            if missing
            else b"".join(fragment for fragment in fragments if fragment is not None)
        )
        byte_exact = not missing and reconstructed == record.surface.encode("utf-8")
        if byte_exact:
            exact_bytes += 1

        state = controller.initial(PENMANConstraintState.initial())
        rejected_index: int | None = None
        rejected_token: int | None = None
        if byte_exact:
            for token_index, token_id in enumerate(token_ids):
                successor = controller.transition(state, token_id)
                if successor is None:
                    rejected_index = token_index
                    rejected_token = token_id
                    break
                state = successor
        finished = (
            None
            if not byte_exact or rejected_index is not None
            else controller.transition(state, eos_token_id)
        )
        if finished is not None and finished.is_accepting:
            accepted += 1
            continue
        if len(failures) < args.examples:
            failures.append(
                {
                    "ordinal": record.ordinal,
                    "id": record.identifier,
                    "start_line": record.start_line,
                    "byte_exact": byte_exact,
                    "missing_token_ids": missing[:10],
                    "rejected_token_index": rejected_index,
                    "rejected_token_id": rejected_token,
                    "pending_utf8_hex": state.pending_utf8.hex(),
                    "representation_reason": state.representation.dead_reason,
                }
            )

    total = len(records)
    coverage = accepted / total if total else None
    config_bytes = (
        REPOSITORY_ROOT / "configs" / "lora_baselines.json"
    ).read_bytes()
    report = {
        "schema_version": 1,
        "audit": "penman-bytelevel-token-lifting",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "file_name": args.corpus.name,
            "sha256": observed_hash,
            "size_bytes": args.corpus.stat().st_size,
            "records": total,
            "repository_url": args.source_repository_url,
            "revision": args.source_revision,
            "repository_file": args.source_file,
            "license": args.source_license,
        },
        "tokenizer": {
            "model": args.model,
            "requested_revision": args.revision,
            "resolved_revision": getattr(tokenizer, "init_kwargs", {}).get(
                "_commit_hash"
            ),
            "class": type(tokenizer).__name__,
            "vocabulary_size": len(tokenizer),
            "eos_token_ids": sorted(vocabulary.eos_token_ids),
            "transformers": transformers.__version__,
        },
        "coverage": {
            "denominator": total,
            "exact_byte_roundtrip": exact_bytes,
            "accepted_with_eos": accepted,
            "acceptance_rate": coverage,
            "failure_examples": failures,
        },
        "token_lengths": {
            "median": _percentile(token_lengths, 0.5),
            "p95": _percentile(token_lengths, 0.95),
            "maximum": max(token_lengths, default=None),
        },
        "implementation": {
            "lifting_sha256": sha256_file(
                REPOSITORY_ROOT / "src" / "automata_drs" / "lifting.py"
            ),
            "penman_state_sha256": sha256_file(
                REPOSITORY_ROOT
                / "src"
                / "automata_drs"
                / "penman"
                / "state.py"
            ),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
    }
    report["gate"] = {
        "required_coverage": args.require_coverage,
        "passed": (
            coverage is not None
            and coverage >= args.require_coverage
            and exact_bytes == accepted
        ),
    }
    if args.verify_existing is not None:
        try:
            stored = json.loads(args.verify_existing.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"cannot read stored audit: {error}")
        if not isinstance(stored, dict):
            parser.error("stored audit is not a JSON object")
        observed = {
            key: value for key, value in stored.items() if key != "created_at_utc"
        }
        expected = {
            key: value for key, value in report.items() if key != "created_at_utc"
        }
        if observed != expected:
            differing = sorted(
                key
                for key in set(observed) | set(expected)
                if observed.get(key) != expected.get(key)
            )
            parser.error(
                "stored token-lifting audit differs from current evidence: "
                + ", ".join(differing)
            )
        print(f"verified stored PENMAN token-lifting audit: {args.verify_existing}")
    else:
        assert args.output is not None
        try:
            _write_exclusive(args.output, report)
        except FileExistsError:
            parser.error(f"refusing to overwrite existing audit: {args.output}")
    print(
        f"PENMAN token lifting: {accepted}/{total}; "
        f"exact bytes: {exact_bytes}/{total}; gate={report['gate']['passed']}"
    )
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
