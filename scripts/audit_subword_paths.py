#!/usr/bin/env python3
"""Replay gold SBNs through an exact ByteLevel-BPE constraint processor."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import SBNPolicy, validate_sbn  # noqa: E402
from automata_drs.causal_lm import iter_sbn_records  # noqa: E402
from automata_drs.subword import (  # noqa: E402
    ByteLevelVocabulary,
    CausalPrefixStateCache,
    SubwordSBNLogitsProcessor,
)


def policy_for(configuration: str) -> SBNPolicy:
    base = {"BF": "B", "WF": "W"}.get(configuration, configuration)
    return SBNPolicy.for_configuration(base)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=["B", "W", "BF", "WF"],
        default=["W", "WF"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--require-zero-false-rejections",
        action="store_true",
        help="write the audit, then exit nonzero unless every eligible path passes",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.max_new_tokens <= 0:
        parser.error("max-new-tokens must be positive")
    if args.expected_records is not None and args.expected_records <= 0:
        parser.error("expected-records must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        parser.error("install the LoRA runtime with: pip install -e '.[lora]'")
        raise AssertionError("unreachable") from error

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocabulary = ByteLevelVocabulary.from_tokenizer(tokenizer)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        parser.error("tokenizer has no EOS token")
    records = list(iter_sbn_records(args.path, limit=args.limit))
    if not records:
        parser.error(f"no records found in {args.path}")
    if args.expected_records is not None and len(records) != args.expected_records:
        parser.error(
            f"found {len(records)} records, expected {args.expected_records}"
        )

    reports = []
    for configuration in args.configurations:
        policy = policy_for(configuration)
        processor = SubwordSBNLogitsProcessor(
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            prompt_token_ids=(),
            policy=policy,
            max_new_tokens=args.max_new_tokens,
            bounded_completion=configuration in {"BF", "WF"},
        )
        accepted = 0
        skipped_policy_invalid = 0
        skipped_over_budget = 0
        rejected_count = 0
        false_rejections: list[dict[str, object]] = []
        peak_prefix_cache_entries = 0
        started = time.perf_counter()
        for record_index, record in enumerate(records, start=1):
            # Generation creates one processor per example. Mirror that memory
            # boundary here instead of retaining an audit-wide trie of every
            # gold token prefix. Transition and completion caches remain LRU-
            # bounded and reusable, while live prefix memory is O(one record).
            processor.prefix_states = CausalPrefixStateCache(
                processor.controller,
                prompt_token_ids=(),
                policy=policy,
            )
            if not validate_sbn(record.target, policy=policy).valid:
                skipped_policy_invalid += 1
                continue
            target_ids = [
                int(token_id)
                for token_id in tokenizer(
                    record.target, add_special_tokens=False
                )["input_ids"]
            ]
            path = target_ids + [int(eos_token_id)]
            if len(path) > args.max_new_tokens:
                skipped_over_budget += 1
                continue
            prefix: list[int] = []
            rejected = None
            for token_index, token_id in enumerate(path):
                if not processor.token_is_allowed_for_prefix(prefix, token_id):
                    rejected = {
                        "id": record.identifier,
                        "token_index": token_index,
                        "token_id": token_id,
                        "path_tokens": len(path),
                        "decoded_prefix": tokenizer.decode(
                            prefix,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        ),
                    }
                    break
                prefix.append(token_id)
            if rejected is None:
                final_state = processor.prefix_states.state_for(prefix)
                if final_state is None or not final_state.is_accepting:
                    rejected = {
                        "id": record.identifier,
                        "token_index": len(path),
                        "token_id": None,
                        "path_tokens": len(path),
                        "decoded_prefix": record.target,
                        "reason": "replayed path did not end accepting",
                    }
            if rejected is None:
                accepted += 1
            else:
                rejected_count += 1
                if len(false_rejections) < 50:
                    false_rejections.append(rejected)
            peak_prefix_cache_entries = max(
                peak_prefix_cache_entries, processor.prefix_states.size
            )
            if record_index % args.progress_every == 0:
                print(
                    f"{configuration}: {record_index}/{len(records)} "
                    f"accepted={accepted} false={rejected_count}",
                    flush=True,
                )
        eligible = (
            len(records) - skipped_policy_invalid - skipped_over_budget
        )
        reports.append(
            {
                "configuration": configuration,
                "records": len(records),
                "eligible": eligible,
                "accepted": accepted,
                "false_rejection_count": rejected_count,
                "skipped_policy_invalid": skipped_policy_invalid,
                "skipped_over_budget": skipped_over_budget,
                "false_rejection_examples": false_rejections,
                "seconds": time.perf_counter() - started,
                "prefix_cache_entries": processor.prefix_states.size,
                "peak_prefix_cache_entries": peak_prefix_cache_entries,
                "transition_cache": asdict(processor.controller.cache_info()),
                "completion_cache": asdict(processor.completion_cache_info()),
            }
        )

    gate_passed = all(
        report["eligible"] > 0
        and report["accepted"] == report["eligible"]
        and report["false_rejection_count"] == 0
        for report in reports
    )
    result = {
        "path": str(args.path),
        "input_sha256": hashlib.sha256(args.path.read_bytes()).hexdigest(),
        "tokenizer": args.tokenizer,
        "transformers": transformers.__version__,
        "max_new_tokens": args.max_new_tokens,
        "expected_records": args.expected_records,
        "vocabulary_fragments": len(vocabulary.fragments),
        "raw_byte_fallbacks": len(vocabulary.byte_token_ids),
        "reports": reports,
        "gate": {
            "require_zero_false_rejections": args.require_zero_false_rejections,
            "passed": gate_passed,
        },
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if gate_passed or not args.require_zero_false_rejections else 1


if __name__ == "__main__":
    raise SystemExit(main())
