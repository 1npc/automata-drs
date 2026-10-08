#!/usr/bin/env python3
"""Stream AMR train/dev through the selected tokenizer and PENMAN controller."""

from __future__ import annotations

import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import (  # noqa: E402
    ByteLevelTokenConstraintController,
    ByteLevelVocabulary,
    PENMANConstraintState,
    sha256_file,
)
from automata_drs.amr_causal_lm import build_amr_prompt  # noqa: E402
from automata_drs.amr_data import iter_amr_records  # noqa: E402


def _token_ids(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    values = encoded["input_ids"]
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("tokenizer returned a non-singleton batch")
        values = values[0]
    return [int(value) for value in values]


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _length_summary(values: list[int]) -> dict[str, int | None]:
    return {
        "minimum": min(values, default=None),
        "median": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "maximum": max(values, default=None),
    }


def _resolved_huggingface_revision(
    tokenizer: Any, cached_tokenizer_config: Path
) -> str | None:
    """Return the concrete Hub commit without trusting the requested revision."""

    candidates = [
        getattr(tokenizer, "_commit_hash", None),
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
    ]
    # Hugging Face snapshot entries are normally symlinks into ``blobs``.
    # Inspect the path returned by ``cached_file`` before resolving that
    # symlink, while also accepting a concrete non-symlink snapshot path.
    for candidate_path in (cached_tokenizer_config, cached_tokenizer_config.resolve()):
        parts = candidate_path.parts
        for index, part in enumerate(parts[:-1]):
            if part == "snapshots":
                candidates.append(parts[index + 1])
    revisions = {
        value.lower()
        for value in candidates
        if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}", value)
    }
    if len(revisions) > 1:
        raise ValueError(
            "conflicting concrete tokenizer revisions: " + ", ".join(sorted(revisions))
        )
    return next(iter(revisions), None)


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/amr_transfer_experiment.json"),
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "development"), default=("train", "development")
    )
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--max-cache-entries", type=int, default=20_000)
    args = parser.parse_args()
    if args.examples < 0:
        parser.error("--examples must be non-negative")
    if args.max_cache_entries < 0:
        parser.error("--max-cache-entries must be non-negative")

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    host = config["host"]
    try:
        import transformers
        from transformers import AutoTokenizer
        from transformers.utils.hub import cached_file
    except ImportError as error:
        parser.error(f"transformers is required: {error}")

    load_kwargs: dict[str, Any] = {"revision": host["revision"]}
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = str(args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(host["repo_id"], **load_kwargs)
    cached_tokenizer_config = cached_file(
        host["repo_id"],
        "tokenizer_config.json",
        revision=host["revision"],
        cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
    )
    if cached_tokenizer_config is None:
        parser.error("could not resolve the cached tokenizer_config.json")
    try:
        resolved_revision = _resolved_huggingface_revision(
            tokenizer, Path(cached_tokenizer_config)
        )
    except ValueError as error:
        parser.error(str(error))
    if resolved_revision is None:
        parser.error(
            "could not prove the concrete tokenizer revision from the tokenizer "
            "metadata or Hub snapshot path"
        )
    vocabulary = ByteLevelVocabulary.from_tokenizer(tokenizer)
    eos_token_id = int(tokenizer.eos_token_id)
    if eos_token_id not in vocabulary.eos_token_ids:
        parser.error("tokenizer EOS is absent from the byte-level vocabulary")
    controller = ByteLevelTokenConstraintController(
        vocabulary,
        incomplete_utf8_is_extendable=PENMANConstraintState.incomplete_utf8_is_extendable,
        max_cache_entries=args.max_cache_entries,
    )

    split_reports: dict[str, dict[str, Any]] = {}
    total_sequence_tokens = 0
    total_records = 0
    all_passed = True
    global_maximum = 0

    for split in args.splits:
        expected = config["data"][split]
        configured_path = Path(expected["path"])
        path = (
            args.data_root / configured_path.name
            if args.data_root is not None
            else REPOSITORY_ROOT / configured_path
        )
        observed_hash = sha256_file(path)
        hash_matches = observed_hash == expected["sha256"]
        prompt_lengths: list[int] = []
        target_lengths: list[int] = []
        sequence_lengths: list[int] = []
        exact_bytes = 0
        accepted_with_eos = 0
        failures: list[dict[str, Any]] = []
        longest: list[tuple[int, int, dict[str, int]]] = []
        records = 0

        with path.open(encoding="utf-8") as source:
            for record in iter_amr_records(source):
                records += 1
                prompt_ids = _token_ids(
                    tokenizer, build_amr_prompt(record.sentence), add_special_tokens=True
                )
                target_ids = _token_ids(
                    tokenizer, record.graph, add_special_tokens=False
                )
                prompt_length = len(prompt_ids)
                target_length = len(target_ids) + 1
                sequence_length = prompt_length + target_length
                prompt_lengths.append(prompt_length)
                target_lengths.append(target_length)
                sequence_lengths.append(sequence_length)
                length_record = {
                    "ordinal": record.ordinal,
                    "sentence_start_line": record.sentence_start_line,
                    "graph_start_line": record.graph_start_line,
                    "prompt_tokens": prompt_length,
                    "target_tokens_with_eos": target_length,
                    "sequence_tokens": sequence_length,
                }
                item = (sequence_length, record.ordinal, length_record)
                if len(longest) < 16:
                    heapq.heappush(longest, item)
                elif item[:2] > longest[0][:2]:
                    heapq.heapreplace(longest, item)

                fragments = [vocabulary.fragments.get(token_id) for token_id in target_ids]
                missing = [
                    token_id
                    for token_id, fragment in zip(target_ids, fragments, strict=True)
                    if fragment is None
                ]
                reconstructed = (
                    b""
                    if missing
                    else b"".join(
                        fragment for fragment in fragments if fragment is not None
                    )
                )
                byte_exact = not missing and reconstructed == record.graph.encode("utf-8")
                if byte_exact:
                    exact_bytes += 1

                state = controller.initial(PENMANConstraintState.initial())
                rejected_index: int | None = None
                if byte_exact:
                    for token_index, token_id in enumerate(target_ids):
                        successor = controller.transition(state, token_id)
                        if successor is None:
                            rejected_index = token_index
                            break
                        state = successor
                finished = (
                    None
                    if not byte_exact or rejected_index is not None
                    else controller.transition(state, eos_token_id)
                )
                if finished is not None and finished.is_accepting:
                    accepted_with_eos += 1
                elif len(failures) < args.examples:
                    failures.append(
                        {
                            "ordinal": record.ordinal,
                            "graph_start_line": record.graph_start_line,
                            "byte_exact": byte_exact,
                            "missing_token_ids": missing[:10],
                            "rejected_token_index": rejected_index,
                            "pending_utf8_hex": state.pending_utf8.hex(),
                            "representation_reason": state.representation.dead_reason,
                        }
                    )

        count_matches = records == expected["records"]
        coverage = accepted_with_eos / records if records else None
        threshold_counts = {
            str(threshold): sum(length > threshold for length in sequence_lengths)
            for threshold in (1024, 1536, 2048, 3072, 4096)
        }
        passed = (
            hash_matches
            and count_matches
            and exact_bytes == records
            and accepted_with_eos == records
        )
        all_passed = all_passed and passed
        total_sequence_tokens += sum(sequence_lengths)
        total_records += records
        global_maximum = max(global_maximum, max(sequence_lengths, default=0))
        split_reports[split] = {
            "path": str(path),
            "sha256": observed_hash,
            "records": records,
            "checks": {
                "hash_matches": hash_matches,
                "record_count_matches": count_matches,
                "exact_target_byte_roundtrip": exact_bytes == records,
                "accepted_with_eos": accepted_with_eos == records,
            },
            "coverage": {
                "exact_target_byte_roundtrip": exact_bytes,
                "accepted_with_eos": accepted_with_eos,
                "acceptance_rate": coverage,
                "failure_examples": failures,
            },
            "lengths": {
                "prompt": _length_summary(prompt_lengths),
                "target_with_eos": _length_summary(target_lengths),
                "sequence": _length_summary(sequence_lengths),
                "sequence_over_threshold": threshold_counts,
                "sequence_token_total": sum(sequence_lengths),
                "longest_records": [
                    value for _, _, value in sorted(longest, reverse=True)
                ],
            },
            "gate_passed": passed,
        }

    rounded_maximum = (
        math.ceil(global_maximum / 128) * 128 if global_maximum else None
    )
    empty_array_overhead = sys.getsizeof(array("I"))
    empty_tuple_overhead = sys.getsizeof((array("I"), 0))
    list_pointer_bytes = 8
    approximate_compact_bytes = (
        total_sequence_tokens * 4
        + total_records
        * (empty_array_overhead + empty_tuple_overhead + list_pointer_bytes)
    )
    report = {
        "schema_version": 1,
        "audit": "amr-selected-tokenizer-length-and-lifting",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "path": str(args.config),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "tokenizer": {
            "model": host["repo_id"],
            "requested_revision": host["revision"],
            "resolved_revision": resolved_revision,
            "class": type(tokenizer).__name__,
            "vocabulary_size": len(tokenizer),
            "eos_token_id": eos_token_id,
            "transformers": transformers.__version__,
            "model_max_length": tokenizer.model_max_length,
        },
        "controller": {
            "max_cache_entries": args.max_cache_entries,
            "final_cache": {
                "hits": controller.cache_info().hits,
                "misses": controller.cache_info().misses,
                "entries": controller.cache_info().size,
                "evictions": controller.cache_info().evictions,
            },
        },
        "splits": split_reports,
        "training_length_decision": {
            "minimum_observed_no_truncation": global_maximum,
            "rounded_up_multiple_of_128": rounded_maximum,
            "status": "evidence-only; freeze max_length in config after resource review",
        },
        "host_memory_estimate": {
            "method": "compact uint32 token payload plus measured empty array/tuple and one list pointer per record; excludes transient batches and Python allocator fragmentation",
            "records": total_records,
            "sequence_tokens": total_sequence_tokens,
            "approximate_compact_bytes": approximate_compact_bytes,
        },
        "implementation": {
            "reader_sha256": sha256_file(
                REPOSITORY_ROOT / "src" / "automata_drs" / "amr_data.py"
            ),
            "prompt_sha256": sha256_file(
                REPOSITORY_ROOT / "src" / "automata_drs" / "amr_causal_lm.py"
            ),
            "lifting_sha256": sha256_file(
                REPOSITORY_ROOT / "src" / "automata_drs" / "lifting.py"
            ),
            "penman_state_sha256": sha256_file(
                REPOSITORY_ROOT / "src" / "automata_drs" / "penman" / "state.py"
            ),
            "trainer_sha256": sha256_file(
                REPOSITORY_ROOT / "scripts" / "train_causal_lm_lora.py"
            ),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "gate": {"passed": all_passed},
    }
    try:
        _write_exclusive(args.output, report)
    except FileExistsError:
        parser.error(f"refusing to overwrite existing audit: {args.output}")
    print(f"report: {args.output}")
    for split, value in split_reports.items():
        print(
            f"{split}: {value['coverage']['accepted_with_eos']}/"
            f"{value['records']} accepted; max sequence="
            f"{value['lengths']['sequence']['maximum']}"
        )
    print(f"gate passed: {all_passed}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
