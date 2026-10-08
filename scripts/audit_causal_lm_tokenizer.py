#!/usr/bin/env python3
"""Audit SBN sequence lengths and target round trips for one tokenizer."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.causal_lm import (  # noqa: E402
    build_sbn_prompt,
    iter_sbn_records,
)


def percentile(values: list[int], probability: float) -> int:
    if not values:
        raise ValueError("percentile requires data")
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        parser.error("install the LoRA runtime with: pip install -e '.[lora]'")
        raise AssertionError("unreachable") from error

    load_kwargs = {}
    if args.revision:
        load_kwargs["revision"] = args.revision
    if args.cache_dir:
        load_kwargs["cache_dir"] = str(args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    if tokenizer.eos_token_id is None:
        parser.error("tokenizer has no EOS token")

    prompt_lengths: list[int] = []
    target_lengths: list[int] = []
    total_lengths: list[int] = []
    roundtrip_failures: list[dict[str, object]] = []
    records = list(iter_sbn_records(args.path, limit=args.limit))
    if not records:
        parser.error(f"no records found in {args.path}")

    for record in records:
        prompt_ids = tokenizer(
            build_sbn_prompt(record.source), add_special_tokens=True
        )["input_ids"]
        target_ids = tokenizer(record.target, add_special_tokens=False)["input_ids"]
        decoded = tokenizer.decode(
            target_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != record.target and len(roundtrip_failures) < 20:
            roundtrip_failures.append(
                {
                    "id": record.identifier,
                    "expected": record.target,
                    "decoded": decoded,
                }
            )
        prompt_lengths.append(len(prompt_ids))
        target_lengths.append(len(target_ids) + 1)
        total_lengths.append(len(prompt_ids) + len(target_ids) + 1)

    report = {
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": (
            getattr(tokenizer, "_commit_hash", None)
            or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
            or args.revision
        ),
        "transformers": transformers.__version__,
        "records": len(records),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "lengths": {
            "prompt_max": max(prompt_lengths),
            "target_p50": percentile(target_lengths, 0.50),
            "target_p95": percentile(target_lengths, 0.95),
            "target_p99": percentile(target_lengths, 0.99),
            "target_max": max(target_lengths),
            "total_p95": percentile(total_lengths, 0.95),
            "total_p99": percentile(total_lengths, 0.99),
            "total_max": max(total_lengths),
        },
        "target_roundtrip_failure_count": sum(
            tokenizer.decode(
                tokenizer(record.target, add_special_tokens=False)["input_ids"],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            != record.target
            for record in records
        ),
        "target_roundtrip_failure_examples": roundtrip_failures,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            parser.error(f"output already exists: {args.output}")
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
