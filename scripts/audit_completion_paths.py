#!/usr/bin/env python3
"""Replay known valid ByT5 paths through finite-budget certification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import (  # noqa: E402
    ByT5SBNLogitsProcessor,
    ByT5Vocabulary,
    SBNPolicy,
    check_sbn_state_wellformedness,
    validate_sbn,
)


def strategy_key(row: dict[str, Any]) -> str:
    decode = row["decode"]
    if decode["strategy"] == "gold":
        return "gold"
    if decode["strategy"] == "greedy":
        return "greedy"
    return f"beam{decode['num_beams']}"


def token_label(vocabulary: ByT5Vocabulary, token_id: int) -> str:
    if token_id == vocabulary.eos_token_id:
        return "<eos>"
    byte = vocabulary.byte_for_token(token_id)
    if byte is None:
        return f"<id:{token_id}>"
    if 0x20 <= byte < 0x7F:
        return repr(chr(byte))
    return f"<byte:0x{byte:02x}>"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="generation experiment JSONL")
    parser.add_argument(
        "--input-format",
        choices=["experiment-jsonl", "sbn-tsv"],
        default="experiment-jsonl",
        help="audit generated JSONL rows or gold source<TAB>SBN records",
    )
    parser.add_argument("--source-configuration", default="U")
    parser.add_argument("--strategy")
    parser.add_argument("--policy", choices=["B", "W"], default="B")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--completion-budget-factor", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        help="optionally persist the JSON audit report",
    )
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="write one stderr progress line every N selected paths (0 disables)",
    )
    parser.add_argument(
        "--fail-on-rejection",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if not 0 < args.completion_budget_factor <= 1:
        parser.error("--completion-budget-factor must be in (0, 1]")
    if args.limit == 0 or args.limit < -1:
        parser.error("--limit must be -1 or positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")

    rows: list[dict[str, Any]] = []
    with args.path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            if args.input_format == "experiment-jsonl":
                row = json.loads(line)
                if row.get("configuration") != args.source_configuration:
                    continue
                if args.strategy and strategy_key(row) != args.strategy:
                    continue
            else:
                raw = line.rstrip("\n\r")
                if "\t" not in raw:
                    raise ValueError(
                        f"expected source<TAB>SBN at {args.path}:{line_number}"
                    )
                _, prediction = raw.split("\t", 1)
                structural = validate_sbn(prediction)
                semantic_valid: bool | None = None
                if structural.valid:
                    semantic_valid = check_sbn_state_wellformedness(
                        structural.state
                    ).valid
                row = {
                    "id": f"{args.path}:{line_number}",
                    "prediction": prediction,
                    "configuration": "gold",
                    "decode": {"strategy": "gold", "num_beams": 1},
                    "validity": {
                        "sbn": structural.valid,
                        "clf_referee": semantic_valid,
                    },
                }
            rows.append(row)
            if args.limit >= 0 and len(rows) >= args.limit:
                break

    vocabulary = ByT5Vocabulary()
    eligible = 0
    certified = 0
    skipped_invalid = 0
    skipped_over_budget = 0
    rejections: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows, start=1):
        if args.progress_every and row_index % args.progress_every == 0:
            print(
                f"completion audit: {row_index}/{len(rows)} selected paths",
                file=sys.stderr,
                flush=True,
            )
        validity = row.get("validity", {})
        expected_valid = validity.get("raw_sbn", validity.get("sbn")) is True
        if args.policy == "W":
            expected_valid = expected_valid and validity.get(
                "raw_clf_referee", validity.get("clf_referee")
            ) is True
        if not expected_valid:
            skipped_invalid += 1
            continue

        prediction = str(row["prediction"])
        path = [
            vocabulary.token_for_byte(byte)
            for byte in prediction.encode("utf-8")
        ] + [vocabulary.eos_token_id]
        if len(path) > args.max_new_tokens:
            skipped_over_budget += 1
            continue
        eligible += 1

        processor = ByT5SBNLogitsProcessor(
            policy=SBNPolicy.for_configuration(args.policy),
            max_new_tokens=args.max_new_tokens,
            bounded_completion_window=args.max_new_tokens,
            completion_budget_factor=args.completion_budget_factor,
        )
        prefix = [0]
        first_rejection: dict[str, Any] | None = None
        for zero_based_step, token_id in enumerate(path):
            if processor.token_is_allowed_for_prefix(prefix, token_id):
                prefix.append(token_id)
                continue

            state = processor.prefix_states.state_for(prefix)
            successor = (
                None
                if state is None
                else processor.controller._compute_transition(state, token_id)
            )
            actual = successor
            for later in path[zero_based_step + 1 :]:
                actual = (
                    None
                    if actual is None
                    else processor.controller._compute_transition(actual, later)
                )
            first_rejection = {
                "id": row["id"],
                "strategy": strategy_key(row),
                "step": zero_based_step + 1,
                "token_id": token_id,
                "token": token_label(vocabulary, token_id),
                "generated_before": zero_based_step,
                "remaining_after": args.max_new_tokens - zero_based_step - 1,
                "family": (
                    "structural-transition"
                    if successor is None
                    else "completion-certificate"
                ),
                "mode": None if state is None else state.sbn.mode.value,
                "lexeme": None if state is None else state.sbn.current_lexeme,
                "pending_label": None if state is None else state.sbn.pending_label,
                "actual_suffix_accepts": (
                    actual is not None and actual.is_accepting
                ),
                "actual_path_tokens": len(path),
            }
            break

        if first_rejection is None:
            certified += 1
        else:
            rejections.append(first_rejection)

    report = {
        "path": str(args.path),
        "input_format": args.input_format,
        "source_configuration": (
            args.source_configuration
            if args.input_format == "experiment-jsonl"
            else None
        ),
        "strategy": (
            args.strategy
            if args.input_format == "experiment-jsonl"
            else "gold"
        ),
        "policy": args.policy,
        "max_new_tokens": args.max_new_tokens,
        "completion_budget_factor": args.completion_budget_factor,
        "selected": len(rows),
        "eligible_valid_paths": eligible,
        "certified_complete_paths": certified,
        "false_rejections": len(rejections),
        "skipped_invalid": skipped_invalid,
        "skipped_over_budget": skipped_over_budget,
        "rejections": rejections,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if args.fail_on_rejection and rejections else 0


if __name__ == "__main__":
    raise SystemExit(main())
