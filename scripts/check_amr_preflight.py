#!/usr/bin/env python3
"""Recompute the AMR preflight gate before any GPU model is loaded."""

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

from automata_drs.penman import sha256_file  # noqa: E402


TRAIN_AUDIT_DEFAULT = Path("runs/amr/server-train-corpus-audit.json")
TOKEN_AUDIT_DEFAULT = Path("runs/amr/qwen35-4b-train-dev-tokenization.json")
MAX_COMPACT_DATASET_BYTES = 4 * 1024 * 1024 * 1024


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _without_timestamp(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "created_at_utc"}


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")


def build_report(
    config_path: Path, train_audit_path: Path, token_audit_path: Path
) -> dict[str, Any]:
    config = _read_object(config_path)
    train_audit = _read_object(train_audit_path)
    token_audit = _read_object(token_audit_path)
    host = config["host"]
    data = config["data"]

    train_expected = data["train"]
    train_normalization = train_expected.get("normalization", {})
    development_expected = data["development"]
    train_observed = train_audit.get("corpus", {})
    token_splits = token_audit.get("splits", {})
    token_train = token_splits.get("train", {})
    token_development = token_splits.get("development", {})
    implementation = token_audit.get("implementation", {})
    train_implementation = train_audit.get("implementation", {})
    tokenizer = token_audit.get("tokenizer", {})
    length_decision = token_audit.get("training_length_decision", {})
    memory = token_audit.get("host_memory_estimate", {})
    normalization_audit_path = REPOSITORY_ROOT / str(
        train_normalization.get("derivation_audit", "")
    )
    normalization_audit = (
        _read_object(normalization_audit_path)
        if normalization_audit_path.is_file()
        else {}
    )

    minimum_length = length_decision.get("minimum_observed_no_truncation")
    rounded_length = length_decision.get("rounded_up_multiple_of_128")
    model_max_length = tokenizer.get("model_max_length")
    compact_bytes = memory.get("approximate_compact_bytes")
    longest = token_train.get("lengths", {}).get("longest_records", [])
    train_prompt_max = token_train.get("lengths", {}).get("prompt", {}).get("maximum")
    development_prompt_max = (
        token_development.get("lengths", {}).get("prompt", {}).get("maximum")
    )
    development_target_max = (
        token_development.get("lengths", {})
        .get("target_with_eos", {})
        .get("maximum")
    )

    current_hashes = {
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
        "script_sha256": sha256_file(
            REPOSITORY_ROOT / "scripts" / "audit_amr_tokenization.py"
        ),
    }
    checks = {
        "train_audit_gate": train_audit.get("gate", {}).get("passed") is True,
        "train_nfc_derivation": (
            train_normalization.get("form") == "NFC"
            and train_normalization.get("changed_records") == [50632]
            and normalization_audit.get("audit")
            == "amr-nfc-corpus-derivation"
            and normalization_audit.get("gate", {}).get("passed") is True
            and normalization_audit.get("source", {}).get("sha256")
            == train_normalization.get("source_sha256")
            and normalization_audit.get("derived", {}).get("sha256")
            == train_expected["sha256"]
            and normalization_audit.get("derived", {}).get("records")
            == train_expected["records"]
            and normalization_audit.get("changes", {}).get("changed_records")
            == train_normalization.get("changed_records")
            and normalization_audit.get("implementation", {}).get(
                "script_sha256"
            )
            == sha256_file(
                REPOSITORY_ROOT / "scripts" / "derive_amr_nfc_corpus.py"
            )
            and sha256_file(normalization_audit_path)
            == train_normalization.get("derivation_audit_sha256")
        ),
        "train_hash": train_observed.get("sha256") == train_expected["sha256"],
        "train_records": train_observed.get("records") == train_expected["records"],
        "train_official_coverage": (
            train_audit.get("official_decode", {}).get("acceptance_rate") == 1.0
        ),
        "train_local_coverage": (
            train_audit.get("local_raw", {}).get("acceptance_rate") == 1.0
        ),
        "train_reader_implementation": (
            train_implementation.get("reader_sha256")
            == current_hashes["reader_sha256"]
        ),
        "train_state_implementation": (
            train_implementation.get("state_sha256")
            == current_hashes["penman_state_sha256"]
        ),
        "train_audit_implementation": (
            train_implementation.get("script_sha256")
            == sha256_file(REPOSITORY_ROOT / "scripts" / "audit_local_amr_corpus.py")
        ),
        "token_audit_gate": token_audit.get("gate", {}).get("passed") is True,
        "token_config_hash": (
            token_audit.get("config", {}).get("sha256") == sha256_file(config_path)
        ),
        "token_splits_are_train_development_only": (
            set(token_splits) == {"train", "development"}
        ),
        "token_train_hash_records": (
            token_train.get("sha256") == train_expected["sha256"]
            and token_train.get("records") == train_expected["records"]
        ),
        "token_development_hash_records": (
            token_development.get("sha256") == development_expected["sha256"]
            and token_development.get("records") == development_expected["records"]
        ),
        "token_train_checks": all(token_train.get("checks", {}).values()),
        "token_train_coverage": (
            token_train.get("coverage", {}).get("exact_target_byte_roundtrip")
            == train_expected["records"]
            and token_train.get("coverage", {}).get("accepted_with_eos")
            == train_expected["records"]
            and token_train.get("coverage", {}).get("acceptance_rate") == 1.0
        ),
        "token_development_checks": all(
            token_development.get("checks", {}).values()
        ),
        "token_development_coverage": (
            token_development.get("coverage", {}).get(
                "exact_target_byte_roundtrip"
            )
            == development_expected["records"]
            and token_development.get("coverage", {}).get("accepted_with_eos")
            == development_expected["records"]
            and token_development.get("coverage", {}).get("acceptance_rate")
            == 1.0
        ),
        "tokenizer_model": tokenizer.get("model") == host["repo_id"],
        "tokenizer_requested_revision": (
            tokenizer.get("requested_revision") == host["revision"]
        ),
        "tokenizer_resolved_revision": (
            tokenizer.get("resolved_revision") == host["revision"]
        ),
        "bounded_transition_cache": (
            token_audit.get("controller", {}).get("max_cache_entries") == 20_000
            and isinstance(
                token_audit.get("controller", {}).get("final_cache", {}).get("entries"),
                int,
            )
            and token_audit["controller"]["final_cache"]["entries"] <= 20_000
        ),
        "token_implementation_hashes": all(
            implementation.get(name) == digest
            for name, digest in current_hashes.items()
        ),
        "integer_no_truncation_length": (
            isinstance(minimum_length, int)
            and not isinstance(minimum_length, bool)
            and minimum_length > 0
            and isinstance(rounded_length, int)
            and not isinstance(rounded_length, bool)
            and rounded_length >= minimum_length
            and rounded_length % 128 == 0
        ),
        "within_model_context": (
            isinstance(rounded_length, int)
            and isinstance(model_max_length, int)
            and rounded_length <= model_max_length
        ),
        "compact_dataset_memory_ceiling": (
            isinstance(compact_bytes, int)
            and compact_bytes > 0
            and compact_bytes <= MAX_COMPACT_DATASET_BYTES
        ),
        "worst_case_smoke_selection_available": (
            isinstance(longest, list)
            and len(longest) == 16
            and all(
                isinstance(row, dict)
                and isinstance(row.get("ordinal"), int)
                and isinstance(row.get("sequence_tokens"), int)
                for row in longest
            )
        ),
        "generation_length_evidence": (
            isinstance(train_prompt_max, int)
            and train_prompt_max > 0
            and isinstance(development_prompt_max, int)
            and development_prompt_max > 0
            and isinstance(development_target_max, int)
            and development_target_max > 0
        ),
    }
    ordinals = [int(row["ordinal"]) for row in longest] if checks[
        "worst_case_smoke_selection_available"
    ] else []
    maximum_prompt = (
        max(train_prompt_max, development_prompt_max)
        if checks["generation_length_evidence"]
        else None
    )
    recommended_input = (
        math.ceil(maximum_prompt / 128) * 128
        if isinstance(maximum_prompt, int)
        else None
    )
    recommended_new = (
        math.ceil(development_target_max / 128) * 128
        if isinstance(development_target_max, int)
        else None
    )
    report = {
        "schema_version": 1,
        "audit": "amr-preflight-gate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "train_audit": str(train_audit_path),
            "train_audit_sha256": sha256_file(train_audit_path),
            "token_audit": str(token_audit_path),
            "token_audit_sha256": sha256_file(token_audit_path),
            "train_normalization_audit": str(normalization_audit_path),
            "train_normalization_audit_sha256": (
                sha256_file(normalization_audit_path)
                if normalization_audit_path.is_file()
                else None
            ),
        },
        "recommendation": {
            "max_length": rounded_length,
            "max_input_tokens": recommended_input,
            "max_new_tokens": recommended_new,
            "maximum_observed_development_target_tokens_with_eos": development_target_max,
            "maximum_observed_sequence_tokens": minimum_length,
            "compact_dataset_bytes_estimate": compact_bytes,
            "compact_dataset_bytes_ceiling": MAX_COMPACT_DATASET_BYTES,
            "smoke_train_record_ordinals": ordinals,
            "smoke_per_device_batch_size": 1,
            "smoke_gradient_accumulation_steps": 16,
            "smoke_max_steps": 8,
        },
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "performance_threshold_used": False,
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/amr_transfer_experiment.json"),
    )
    parser.add_argument("--train-audit", type=Path, default=TRAIN_AUDIT_DEFAULT)
    parser.add_argument("--token-audit", type=Path, default=TOKEN_AUDIT_DEFAULT)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()

    try:
        report = build_report(args.config, args.train_audit, args.token_audit)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    if args.verify_existing is not None:
        try:
            stored = _read_object(args.verify_existing)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            parser.error(str(error))
        if _without_timestamp(stored) != _without_timestamp(report):
            parser.error("stored AMR preflight gate differs from current evidence")
        print(f"verified AMR preflight gate: {args.verify_existing}")
    else:
        assert args.output is not None
        try:
            _write_exclusive(args.output, report)
        except FileExistsError:
            parser.error(f"refusing to overwrite AMR preflight gate: {args.output}")
    print(f"recommended max_length: {report['recommendation']['max_length']}")
    print(f"compact dataset estimate: {report['recommendation']['compact_dataset_bytes_estimate']}")
    print(f"gate passed: {report['gate']['passed']}")
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
