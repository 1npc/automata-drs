#!/usr/bin/env python3
"""Measure gold AMR target lengths under the frozen generation tokenizer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.amr_data import iter_amr_records  # noqa: E402
from automata_drs.generation_manifest import sha256_file  # noqa: E402


def percentile(values: list[int], probability: float) -> int:
    if not values:
        raise ValueError("percentile requires data")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def horizon_summary(lengths: list[int], horizons: list[int]) -> dict[str, Any]:
    if not lengths:
        raise ValueError("lengths must not be empty")
    count = len(lengths)
    return {
        str(horizon): {
            "fit": sum(length <= horizon for length in lengths),
            "over": sum(length > horizon for length in lengths),
            "over_rate": sum(length > horizon for length in lengths) / count,
        }
        for horizon in horizons
    }


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=(192, 256, 288, 384, 512),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.corpus.is_file():
        parser.error(f"corpus is missing: {args.corpus}")
    if len(set(args.horizons)) != len(args.horizons) or any(
        horizon <= 0 for horizon in args.horizons
    ):
        parser.error("--horizons must be unique positive integers")
    if args.output.exists():
        parser.error(f"refusing to overwrite report: {args.output}")

    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        parser.error(f"transformers is unavailable: {error}")

    load_kwargs: dict[str, Any] = {"local_files_only": True}
    if args.revision:
        load_kwargs["revision"] = args.revision
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = str(args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    if tokenizer.eos_token_id is None:
        parser.error("tokenizer has no EOS token")

    lengths: list[int] = []
    roundtrip_failures: list[dict[str, Any]] = []
    roundtrip_failure_count = 0
    with args.corpus.open(encoding="utf-8") as source:
        for record in iter_amr_records(source):
            token_ids = tokenizer(
                record.graph, add_special_tokens=False
            )["input_ids"]
            if token_ids and isinstance(token_ids[0], list):
                if len(token_ids) != 1:
                    parser.error("tokenizer returned a non-singleton batch")
                token_ids = token_ids[0]
            decoded = tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if decoded != record.graph:
                roundtrip_failure_count += 1
                if len(roundtrip_failures) < 20:
                    roundtrip_failures.append(
                        {
                            "ordinal": record.ordinal,
                            "expected": record.graph,
                            "decoded": decoded,
                        }
                    )
            lengths.append(len(token_ids) + 1)

    if not lengths:
        parser.error("corpus contains no AMR records")
    count_matches = args.expected_records is None or len(lengths) == args.expected_records
    report = {
        "schema_version": 1,
        "kind": "amr-gold-target-token-lengths",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "path": str(args.corpus),
            "sha256": sha256_file(args.corpus),
            "records": len(lengths),
            "expected_records": args.expected_records,
        },
        "tokenizer": {
            "model": args.model,
            "revision": args.revision,
            "class": type(tokenizer).__name__,
            "transformers": transformers.__version__,
            "eos_token_id": int(tokenizer.eos_token_id),
        },
        "target_tokens_including_eos": {
            "minimum": min(lengths),
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "maximum": max(lengths),
            "by_horizon": horizon_summary(lengths, sorted(args.horizons)),
        },
        "roundtrip": {
            "failure_count": roundtrip_failure_count,
            "failure_examples": roundtrip_failures,
        },
        "gate": {
            "passed": count_matches and roundtrip_failure_count == 0,
            "record_count_matches": count_matches,
            "exact_roundtrip": roundtrip_failure_count == 0,
        },
    }
    try:
        _write_exclusive(args.output, report)
    except FileExistsError:
        parser.error(f"refusing to overwrite report: {args.output}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
