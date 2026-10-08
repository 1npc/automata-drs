#!/usr/bin/env python3
"""Audit positive finite-horizon PENMAN certificates on gold token paths."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import (  # noqa: E402
    ByteLevelFiniteHorizonPlanner,
    ByteLevelTokenConstraintController,
    ByteLevelVocabulary,
    PENMANConstraintState,
    construct_penman_completion_suffix,
    load_amr_records,
    load_penman_records,
    sha256_file,
    validate_penman,
)


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("tokenizer returned a non-singleton batch")
        encoded = encoded[0]
    return tuple(int(value) for value in encoded)


def _records(path: Path, input_format: str) -> list[dict[str, Any]]:
    if input_format == "penman":
        return [
            {
                "ordinal": record.ordinal,
                "identifier": record.identifier,
                "start_line": record.start_line,
                "surface": record.surface,
            }
            for record in load_penman_records(path)
        ]
    return [
        {
            "ordinal": record.ordinal,
            "identifier": None,
            "start_line": record.graph_start_line,
            "surface": record.graph,
        }
        for record in load_amr_records(path)
    ]


def _byte_vocabulary_sha256(vocabulary: ByteLevelVocabulary) -> str:
    """Fingerprint the exact token-to-byte transducer used by the audit."""

    digest = hashlib.sha256()
    for token_id, fragment in sorted(vocabulary.fragments.items()):
        digest.update(b"F")
        digest.update(struct.pack(">QI", int(token_id), len(fragment)))
        digest.update(fragment)
    for token_id in sorted(vocabulary.eos_token_ids):
        digest.update(b"E")
        digest.update(struct.pack(">Q", int(token_id)))
    for byte, token_id in sorted(vocabulary.byte_token_ids.items()):
        digest.update(b"B")
        digest.update(struct.pack(">BQ", int(byte), int(token_id)))
    return digest.hexdigest()


def _update_token_path_digest(
    digest: Any, ordinal: int, token_ids: Iterable[int]
) -> None:
    values = tuple(int(token_id) for token_id in token_ids)
    digest.update(struct.pack(">QQ", int(ordinal), len(values)))
    for token_id in values:
        if token_id < 0:
            raise ValueError("token IDs must be non-negative")
        digest.update(struct.pack(">Q", token_id))


def _local_tokenizer_assets(path: Path) -> dict[str, Any] | None:
    if not path.is_dir():
        return None
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    )
    files = {
        name: sha256_file(path / name)
        for name in names
        if (path / name).is_file()
    }
    if not files:
        raise ValueError(f"local tokenizer source has no recognized assets: {path}")
    digest = hashlib.sha256()
    for name, file_sha256 in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256.encode("ascii"))
        digest.update(b"\n")
    return {"files": files, "aggregate_sha256": digest.hexdigest()}


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2)
        destination.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument(
        "--input-format", choices=("penman", "amr-paired"), required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--tokenizer-source",
        type=Path,
        help="optional local saved-tokenizer directory; model/revision remain the identity",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-cache-entries", type=int, default=20_000)
    parser.add_argument("--expect-records", type=int)
    parser.add_argument("--expect-corpus-sha256")
    parser.add_argument("--expect-config-sha256")
    parser.add_argument(
        "--contract-file",
        type=Path,
        help="optional immutable resource/experiment plan to bind into the audit",
    )
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--require-zero-false-rejections", action="store_true")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.max_cache_entries < 0:
        parser.error("--max-cache-entries must be non-negative")
    if args.expect_records is not None and args.expect_records <= 0:
        parser.error("--expect-records must be positive")
    if args.limit == 0 or args.limit < -1:
        parser.error("--limit must be -1 or positive")
    if args.examples < 0:
        parser.error("--examples must be non-negative")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")

    corpus_sha256 = sha256_file(args.corpus)
    if (
        args.expect_corpus_sha256 is not None
        and corpus_sha256 != args.expect_corpus_sha256
    ):
        parser.error(
            "corpus SHA-256 mismatch: "
            f"expected {args.expect_corpus_sha256}, observed {corpus_sha256}"
        )
    config_path = REPOSITORY_ROOT / "configs/amr_transfer_experiment.json"
    config_sha256 = sha256_file(config_path)
    if (
        args.expect_config_sha256 is not None
        and config_sha256 != args.expect_config_sha256
    ):
        parser.error(
            "AMR config SHA-256 mismatch: "
            f"expected {args.expect_config_sha256}, observed {config_sha256}"
        )
    contract_binding = None
    if args.contract_file is not None:
        contract_binding = {
            "path": str(args.contract_file),
            "sha256": sha256_file(args.contract_file),
        }

    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        parser.error(f"transformers is required: {error}")

    tokenizer_source = args.tokenizer_source or args.model
    load_kwargs: dict[str, Any] = {}
    if args.tokenizer_source is None:
        load_kwargs["revision"] = args.revision
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = str(args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), **load_kwargs)
    try:
        local_tokenizer_assets = (
            _local_tokenizer_assets(args.tokenizer_source)
            if args.tokenizer_source is not None
            else None
        )
    except ValueError as error:
        parser.error(str(error))
    try:
        vocabulary = ByteLevelVocabulary.from_tokenizer(tokenizer)
    except ValueError as error:
        parser.error(str(error))
    controller = ByteLevelTokenConstraintController(
        vocabulary,
        incomplete_utf8_is_extendable=(
            PENMANConstraintState.incomplete_utf8_is_extendable
        ),
        max_cache_entries=args.max_cache_entries,
    )
    planner = ByteLevelFiniteHorizonPlanner(
        controller,
        construct_surface_suffix=construct_penman_completion_suffix,
        encode_surface=lambda text: _token_ids(tokenizer, text),
    )

    records = _records(args.corpus, args.input_format)
    if args.limit > 0:
        records = records[: args.limit]
    if args.expect_records is not None and len(records) != args.expect_records:
        parser.error(
            f"expected {args.expect_records} records, found {len(records)}"
        )

    exact_byte_records = 0
    locally_valid_records = 0
    over_budget_records = 0
    certified_records = 0
    successor_tests = 0
    certified_successors = 0
    witness_distances: list[int] = []
    witness_slacks: list[int] = []
    failures: list[dict[str, Any]] = []
    token_lengths: list[int] = []
    eos_token_id = min(vocabulary.eos_token_ids)
    token_path_digest = hashlib.sha256()

    def report_progress(record_index: int) -> None:
        if args.progress_every and record_index % args.progress_every == 0:
            print(
                "PENMAN PF progress: "
                f"{record_index}/{len(records)} records, "
                f"{certified_successors}/{successor_tests} successors certified",
                file=sys.stderr,
                flush=True,
            )

    for record_index, record in enumerate(records, 1):
        surface = str(record["surface"])
        token_ids = _token_ids(tokenizer, surface)
        token_lengths.append(len(token_ids))
        fragments = [vocabulary.fragments.get(token_id) for token_id in token_ids]
        missing_token_ids = [
            token_id
            for token_id, fragment in zip(token_ids, fragments, strict=True)
            if fragment is None
        ]
        reconstructed = (
            b""
            if missing_token_ids
            else b"".join(
                fragment for fragment in fragments if fragment is not None
            )
        )
        byte_exact = (
            not missing_token_ids and reconstructed == surface.encode("utf-8")
        )
        local_valid = validate_penman(surface).valid
        exact_byte_records += int(byte_exact)
        locally_valid_records += int(local_valid)
        path = token_ids + (eos_token_id,)
        _update_token_path_digest(token_path_digest, int(record["ordinal"]), path)
        if len(path) > args.max_new_tokens:
            over_budget_records += 1
            if len(failures) < args.examples:
                failures.append(
                    {
                        **{key: record[key] for key in ("ordinal", "identifier", "start_line")},
                        "reason": "gold-path-exceeds-model-token-horizon",
                        "path_tokens": len(path),
                    }
                )
            report_progress(record_index)
            continue
        if not byte_exact or not local_valid:
            if len(failures) < args.examples:
                failures.append(
                    {
                        **{key: record[key] for key in ("ordinal", "identifier", "start_line")},
                        "reason": (
                            "tokenizer-byte-roundtrip-failed"
                            if not byte_exact
                            else "local-penman-validation-failed"
                        ),
                        "missing_token_ids": missing_token_ids[:10],
                        "path_tokens": len(path),
                    }
                )
            report_progress(record_index)
            continue

        state = controller.initial(PENMANConstraintState.initial())
        failure: dict[str, Any] | None = None
        for token_index, token_id in enumerate(path):
            successor = controller.transition(state, token_id)
            if successor is None:
                failure = {
                    "reason": "gold-token-transition-rejected",
                    "token_index": token_index,
                    "token_id": token_id,
                }
                break
            remaining = args.max_new_tokens - token_index - 1
            successor_tests += 1
            witness = planner.completion_within(successor, remaining)
            if witness is None:
                failure = {
                    "reason": "positive-completion-certificate-not-found",
                    "token_index": token_index,
                    "token_id": token_id,
                    "remaining_after_token": remaining,
                    "parser_mode": successor.representation.mode.value,
                    "pending_utf8_hex": successor.pending_utf8.hex(),
                }
                break
            certified_successors += 1
            witness_distances.append(witness.distance)
            witness_slacks.append(remaining - witness.distance)
            state = successor

        if failure is None and state.is_accepting:
            certified_records += 1
            report_progress(record_index)
            continue
        if failure is None:
            failure = {"reason": "gold-path-did-not-end-accepting"}
        if len(failures) < args.examples:
            failures.append(
                {
                    **{key: record[key] for key in ("ordinal", "identifier", "start_line")},
                    "path_tokens": len(path),
                    **failure,
                }
            )
        report_progress(record_index)

    total = len(records)
    gate_passed = bool(
        total
        and exact_byte_records == total
        and locally_valid_records == total
        and over_budget_records == 0
        and certified_records == total
        and successor_tests > 0
        and certified_successors == successor_tests
    )
    reported_revision = (
        getattr(tokenizer, "_commit_hash", None)
        or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    )
    effective_revision = reported_revision or args.revision
    cache_info = controller.cache_info()
    report = {
        "schema_version": 2,
        "audit": "penman-positive-finite-horizon-gold-paths",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(args.corpus),
            "input_format": args.input_format,
            "sha256": corpus_sha256,
            "size_bytes": args.corpus.stat().st_size,
            "records": total,
            "gold_token_paths_sha256": token_path_digest.hexdigest(),
        },
        "tokenizer": {
            "model": args.model,
            "requested_revision": args.revision,
            "load_source": str(tokenizer_source),
            "local_assets": local_tokenizer_assets,
            "reported_resolved_revision": reported_revision,
            "effective_revision": effective_revision,
            "revision_bound": effective_revision == args.revision,
            "class": type(tokenizer).__name__,
            "vocabulary_size": len(tokenizer),
            "eos_token_ids": sorted(vocabulary.eos_token_ids),
            "byte_vocabulary_sha256": _byte_vocabulary_sha256(vocabulary),
            "transformers": transformers.__version__,
        },
        "contract": {
            "configuration_name": "PF-feasibility",
            "max_new_tokens": args.max_new_tokens,
            "completion_budget_factor": 1.0,
            "positive_certificate_only": True,
            "failed_construction_proves_unreachability": False,
            "performance_threshold_used": False,
            "changes_frozen_amr_u_p_suite": False,
            "bound_contract_file": contract_binding,
        },
        "controller": {
            "max_cache_entries": args.max_cache_entries,
            "final_cache": {
                "hits": cache_info.hits,
                "misses": cache_info.misses,
                "entries": cache_info.size,
                "evictions": cache_info.evictions,
            },
        },
        "coverage": {
            "records": total,
            "exact_byte_records": exact_byte_records,
            "locally_valid_records": locally_valid_records,
            "over_budget_records": over_budget_records,
            "certified_records": certified_records,
            "gold_token_successors_tested": successor_tests,
            "gold_token_successors_certified": certified_successors,
            "failure_examples": failures,
        },
        "gold_token_lengths": {
            "median": _percentile(token_lengths, 0.5),
            "p95": _percentile(token_lengths, 0.95),
            "maximum": max(token_lengths, default=None),
        },
        "witnesses": {
            "count": len(witness_distances),
            "tokens_minimum": min(witness_distances, default=None),
            "tokens_median": _percentile(witness_distances, 0.5),
            "tokens_p95": _percentile(witness_distances, 0.95),
            "tokens_maximum": max(witness_distances, default=None),
            "slack_minimum": min(witness_slacks, default=None),
            "slack_median": _percentile(witness_slacks, 0.5),
        },
        "implementation": {
            "lifting_sha256": sha256_file(
                REPOSITORY_ROOT / "src/automata_drs/lifting.py"
            ),
            "penman_state_sha256": sha256_file(
                REPOSITORY_ROOT / "src/automata_drs/penman/state.py"
            ),
            "penman_completion_sha256": sha256_file(
                REPOSITORY_ROOT / "src/automata_drs/penman/completion.py"
            ),
            "audit_sha256": sha256_file(Path(__file__)),
            "config_sha256": config_sha256,
        },
        "gate": {
            "passed": gate_passed and effective_revision == args.revision,
            "require_zero_false_rejections": args.require_zero_false_rejections,
            "performance_threshold_used": False,
        },
    }

    destination = args.verify_existing or args.output
    assert destination is not None
    if args.verify_existing is not None:
        try:
            stored = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"cannot read stored PF audit: {error}")
        observed = {key: value for key, value in stored.items() if key != "created_at_utc"}
        expected = {key: value for key, value in report.items() if key != "created_at_utc"}
        if observed != expected:
            parser.error("stored PF audit differs from current evidence")
        print(f"verified PENMAN PF gold-path audit: {destination}")
    else:
        try:
            _write_exclusive(destination, report)
        except FileExistsError:
            parser.error(f"refusing to overwrite PF audit: {destination}")
    print(
        "PENMAN PF gold paths: "
        f"{certified_records}/{total} records, "
        f"{certified_successors}/{successor_tests} successors; "
        f"gate={report['gate']['passed']}"
    )
    return (
        0
        if report["gate"]["passed"] or not args.require_zero_false_rejections
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
