#!/usr/bin/env python3
"""Replay frozen B/W outputs under fixed counterfactual token horizons.

This is a CPU-only, no-model diagnostic.  It preserves every source row, binds
the complete generation artifact and manifest, and reports where the stored
selected path first loses a constructive BF/WF certificate.  It never claims
what a fresh bounded generation would have selected instead.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import SBNPolicy, validate_sbn  # noqa: E402
from automata_drs.subword import (  # noqa: E402
    ByteLevelVocabulary,
    CausalPrefixStateCache,
    SubwordSBNLogitsProcessor,
)


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/qwen_drs_horizon_stress.json"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repository_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"registered path must be repository-relative: {value}")
    resolved = (REPOSITORY_ROOT / path).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"registered path escapes repository: {value}") from error
    return resolved


def load_contract(config_path: Path) -> dict[str, Any]:
    contract = json.loads(config_path.read_text(encoding="utf-8"))
    schema_version = contract.get("schema_version")
    if schema_version not in {1, 2}:
        raise ValueError("unsupported horizon-stress schema")
    expected_kind = {
        1: "post-primary-qwen-drs-horizon-stress",
        2: "post-closure-qwen-drs-horizon-stress",
    }[schema_version]
    if contract.get("kind") != expected_kind:
        raise ValueError("unexpected horizon-stress kind")
    source = contract.get("source") or {}
    if source.get("configurations") != ["B", "W"]:
        raise ValueError("source configurations must remain exactly B/W")
    audit = contract.get("audit") or {}
    horizons = audit.get("counterfactual_horizons")
    if horizons != [64, 128, 192, 256]:
        raise ValueError("counterfactual horizons changed from the registered grid")
    if audit.get("mapping") != {"B": "BF", "W": "WF"}:
        raise ValueError("B/W to BF/WF mapping changed")
    if audit.get("performance_threshold_used") is not False:
        raise ValueError("horizon stress must not have a performance threshold")
    resources = contract.get("resource_contract") or {}
    if resources.get("model_weights_loaded") is not False:
        raise ValueError("horizon stress must remain model-free")
    if resources.get("gpu_used") is not False:
        raise ValueError("horizon stress must remain CPU-only")
    if resources.get("cpu_math_threads") != 1:
        raise ValueError("horizon stress must remain one-threaded")
    raw_selection = source.get("selection")
    selection = raw_selection if isinstance(raw_selection, dict) else None
    if schema_version == 2 and selection is None:
        raise ValueError("schema version 2 requires structured source selection")
    if selection is not None:
        if schema_version != 2:
            raise ValueError("source selection requires schema version 2")
        if selection.get("rule") != "first-n-paired-source-order":
            raise ValueError("unsupported horizon-stress selection rule")
        selected_records = selection.get("records")
        if (
            not isinstance(selected_records, int)
            or selected_records <= 0
            or selected_records > int(source.get("records", 0))
        ):
            raise ValueError("invalid selected source record count")
        if selection.get("performance_conditioned") is not False:
            raise ValueError("horizon-stress selection must not be performance-conditioned")
    return contract


def load_source_rows(
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], Path, Path]:
    source = contract["source"]
    predictions = resolve_repository_path(source["predictions"])
    manifest_path = resolve_repository_path(source["manifest"])
    if not predictions.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"source B/W artifact is incomplete: {predictions}, {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_records = int(source["records"])
    expected_rows = expected_records * len(source["configurations"])
    expected_manifest = {
        "status": "complete",
        "start": 0,
        "limit": expected_records,
        "max_new_tokens": int(source["source_max_new_tokens"]),
        "num_beams": 1,
        "batch_size": 1,
        "configurations": source["configurations"],
        "expected_examples": expected_records,
        "expected_rows": expected_rows,
        "examples": expected_records,
        "rows": expected_rows,
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected_manifest.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"source manifest violates the registered contract: {mismatches}")
    actual_sha = sha256_path(predictions)
    if manifest.get("output_sha256") != actual_sha:
        raise ValueError("source prediction hash does not match its manifest")

    rows = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_rows:
        raise ValueError(f"source has {len(rows)} rows, expected {expected_rows}")
    by_configuration: dict[str, list[dict[str, Any]]] = {
        configuration: [] for configuration in source["configurations"]
    }
    for row in rows:
        configuration = row.get("configuration")
        if configuration not in by_configuration:
            raise ValueError(f"unexpected source configuration: {configuration!r}")
        by_configuration[configuration].append(row)
    for configuration, selected in by_configuration.items():
        if len(selected) != expected_records:
            raise ValueError(
                f"source {configuration} has {len(selected)} rows, expected {expected_records}"
            )
        identifiers = [row.get("id") for row in selected]
        if len(set(identifiers)) != expected_records:
            raise ValueError(f"source {configuration} IDs are not unique")
    id_sets = [
        {row["id"] for row in by_configuration[configuration]}
        for configuration in source["configurations"]
    ]
    if any(identifier_set != id_sets[0] for identifier_set in id_sets[1:]):
        raise ValueError("B/W source ID sets differ")

    raw_selection = source.get("selection")
    selection = raw_selection if isinstance(raw_selection, dict) else None
    if selection is not None:
        ordered_identifiers = [
            [row["id"] for row in by_configuration[configuration]]
            for configuration in source["configurations"]
        ]
        if any(
            identifiers != ordered_identifiers[0]
            for identifiers in ordered_identifiers[1:]
        ):
            raise ValueError("B/W source orders differ; paired prefix selection is unsafe")
        selected_count = int(selection["records"])
        selected_identifiers = set(ordered_identifiers[0][:selected_count])
        rows = [row for row in rows if row["id"] in selected_identifiers]
        expected_selected_rows = selected_count * len(source["configurations"])
        if len(rows) != expected_selected_rows:
            raise ValueError(
                f"selected source has {len(rows)} rows, expected {expected_selected_rows}"
            )
    return rows, manifest, predictions, manifest_path


def exact_surface_ids(
    tokenizer: Any,
    vocabulary: ByteLevelVocabulary,
    surface: str,
) -> tuple[int, ...]:
    encoded = tuple(
        int(token_id)
        for token_id in tokenizer(surface, add_special_tokens=False)["input_ids"]
    )
    encoded_bytes = b"".join(
        vocabulary.fragments.get(token_id, b"") for token_id in encoded
    )
    if encoded and encoded_bytes == surface.encode("utf-8"):
        return encoded
    return tuple(
        vocabulary.byte_token_ids[byte] for byte in surface.encode("utf-8")
    )


def _token_fragment(
    vocabulary: ByteLevelVocabulary, token_id: int
) -> dict[str, Any]:
    fragment = vocabulary.fragments.get(token_id)
    if fragment is None:
        return {"hex": None, "utf8": None}
    try:
        text = fragment.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        text = None
    return {"hex": fragment.hex(), "utf8": text}


def audit_selected_path(
    *,
    row: dict[str, Any],
    tokenizer: Any,
    vocabulary: ByteLevelVocabulary,
    processor: SubwordSBNLogitsProcessor,
    horizon: int,
) -> dict[str, Any]:
    initial_state = processor.prefix_states.state_for(())
    if initial_state is None:
        raise ValueError("processor has no initial state")
    policy = initial_state.sbn.policy
    surface = str(row.get("prediction", ""))
    policy_validation = validate_sbn(surface, policy=policy)
    generation = row.get("generation") or {}
    source_execution_complete = (
        generation.get("ended_with_eos") is True
        and generation.get("hit_max_new_tokens") is False
    )
    target_ids = exact_surface_ids(tokenizer, vocabulary, surface)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer has no EOS token")
    path = target_ids + (int(eos_token_id),)
    result: dict[str, Any] = {
        "id": row.get("id"),
        "path_tokens_including_eos": len(path),
        "path_within_horizon": len(path) <= horizon,
        "source_policy_valid": policy_validation.valid,
        "source_policy_reason": policy_validation.reason,
        "source_execution_complete": source_execution_complete,
        "accepted": False,
        "first_rejection": None,
    }
    if not policy_validation.valid or not source_execution_complete:
        return result

    processor.prefix_states = CausalPrefixStateCache(
        processor.controller,
        prompt_token_ids=(),
        policy=policy,
    )
    prefix: list[int] = []
    for token_index, token_id in enumerate(path):
        if processor.token_is_allowed_for_prefix(prefix, token_id):
            prefix.append(token_id)
            continue
        state = processor.prefix_states.state_for(prefix)
        successor = (
            None
            if state is None
            else processor.controller._compute_transition(state, token_id)
        )
        remaining_after_candidate = horizon - token_index - 1
        distance = (
            None
            if successor is None or remaining_after_candidate < 0
            else processor._successor_completion_distance(
                successor, remaining_after_candidate
            )
        )
        result["first_rejection"] = {
            "token_index_zero_based": token_index,
            "step_one_based": token_index + 1,
            "token_id": token_id,
            "token_fragment": _token_fragment(vocabulary, token_id),
            "remaining_after_candidate": remaining_after_candidate,
            "unbounded_transition_live": successor is not None,
            "bounded_witness_tokens": distance,
            "decoded_prefix": tokenizer.decode(
                prefix,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        }
        return result

    final_state = processor.prefix_states.state_for(prefix)
    if final_state is None or not final_state.is_accepting:
        result["first_rejection"] = {
            "token_index_zero_based": len(path),
            "step_one_based": len(path) + 1,
            "token_id": None,
            "token_fragment": {"hex": None, "utf8": None},
            "remaining_after_candidate": horizon - len(path),
            "unbounded_transition_live": False,
            "bounded_witness_tokens": None,
            "decoded_prefix": surface,
            "reason": "replayed path did not end accepting",
        }
        return result
    result["accepted"] = True
    return result


def summarize_cell(
    outcomes: Iterable[dict[str, Any]],
    *,
    horizon: int,
    source_configuration: str,
    bounded_configuration: str,
    max_examples: int,
) -> dict[str, Any]:
    rows = list(outcomes)
    counts: Counter[str] = Counter()
    first_steps: list[int] = []
    examples: list[dict[str, Any]] = []
    for outcome in rows:
        counts["attempted"] += 1
        if not outcome["source_policy_valid"]:
            counts["source_policy_invalid"] += 1
            category = "source-policy-invalid"
        elif not outcome["source_execution_complete"]:
            counts["source_execution_incomplete"] += 1
            category = "source-execution-incomplete"
        else:
            counts["auditable"] += 1
            if outcome["path_within_horizon"]:
                counts["path_within_horizon"] += 1
            else:
                counts["path_exceeds_horizon"] += 1
            if outcome["accepted"]:
                counts["accepted"] += 1
                category = "accepted"
            else:
                counts["rejected"] += 1
                rejection = outcome["first_rejection"] or {}
                step = rejection.get("step_one_based")
                if isinstance(step, int):
                    first_steps.append(step)
                if rejection.get("unbounded_transition_live"):
                    counts["locally_live_rejection"] += 1
                    if rejection.get("bounded_witness_tokens") is None:
                        counts["locally_live_without_certificate"] += 1
                    category = "locally-live-without-fitting-certificate"
                else:
                    counts["locally_dead_rejection"] += 1
                    category = "locally-dead-rejection"
                if outcome["path_within_horizon"]:
                    counts["planner_false_rejection_in_horizon"] += 1
        if category != "accepted" and len(examples) < max_examples:
            examples.append({"category": category, **outcome})

    return {
        "horizon": horizon,
        "source_configuration": source_configuration,
        "bounded_configuration": bounded_configuration,
        "attempted": counts["attempted"],
        "source_policy_invalid": counts["source_policy_invalid"],
        "source_execution_incomplete": counts["source_execution_incomplete"],
        "auditable": counts["auditable"],
        "path_within_horizon": counts["path_within_horizon"],
        "path_exceeds_horizon": counts["path_exceeds_horizon"],
        "accepted": counts["accepted"],
        "rejected": counts["rejected"],
        "locally_live_rejection": counts["locally_live_rejection"],
        "locally_live_without_certificate": counts[
            "locally_live_without_certificate"
        ],
        "locally_dead_rejection": counts["locally_dead_rejection"],
        "planner_false_rejection_in_horizon": counts[
            "planner_false_rejection_in_horizon"
        ],
        "first_rejection_step_min": min(first_steps) if first_steps else None,
        "first_rejection_step_max": max(first_steps) if first_steps else None,
        "examples": examples,
    }


def build_report(
    *,
    contract: dict[str, Any],
    config_path: Path,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    predictions_path: Path,
    manifest_path: Path,
    tokenizer: Any,
    tokenizer_path: Path,
    transformers_version: str,
) -> dict[str, Any]:
    vocabulary = ByteLevelVocabulary.from_tokenizer(tokenizer)
    mapping = contract["audit"]["mapping"]
    max_examples = int(contract["audit"]["max_examples_per_cell"])
    max_cache_entries = int(contract["audit"]["max_completion_cache_entries"])
    cells: list[dict[str, Any]] = []
    for horizon in contract["audit"]["counterfactual_horizons"]:
        for source_configuration in contract["source"]["configurations"]:
            bounded_configuration = mapping[source_configuration]
            policy = SBNPolicy.for_configuration(source_configuration)
            processor = SubwordSBNLogitsProcessor(
                tokenizer=tokenizer,
                vocabulary=vocabulary,
                prompt_token_ids=(),
                policy=policy,
                max_new_tokens=horizon,
                bounded_completion=True,
                completion_budget_factor=1.0,
                max_completion_cache_entries=max_cache_entries,
            )
            outcomes = [
                audit_selected_path(
                    row=row,
                    tokenizer=tokenizer,
                    vocabulary=vocabulary,
                    processor=processor,
                    horizon=horizon,
                )
                for row in rows
                if row["configuration"] == source_configuration
            ]
            cells.append(
                summarize_cell(
                    outcomes,
                    horizon=horizon,
                    source_configuration=source_configuration,
                    bounded_configuration=bounded_configuration,
                    max_examples=max_examples,
                )
            )

    tokenizer_files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = tokenizer_path / name
        if path.is_file():
            tokenizer_files[name] = sha256_path(path)
    selected_records = len(rows) // len(contract["source"]["configurations"])
    raw_selection = contract["source"].get("selection")
    selection = raw_selection if isinstance(raw_selection, dict) else None
    report_kind = (
        "post-closure-qwen-drs-horizon-stress-report"
        if contract["schema_version"] == 2
        else "post-primary-qwen-drs-horizon-stress-report"
    )
    return {
        "schema_version": 1,
        "kind": report_kind,
        "contract": {
            "path": str(config_path.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_path(config_path),
            "implementation": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
            "implementation_sha256": sha256_path(Path(__file__).resolve()),
            "registration_boundary": contract["registration_boundary"],
            "performance_threshold_used": False,
        },
        "prerequisites": contract.get("prerequisites"),
        "source": {
            "predictions": str(predictions_path.relative_to(REPOSITORY_ROOT)),
            "predictions_sha256": sha256_path(predictions_path),
            "manifest": str(manifest_path.relative_to(REPOSITORY_ROOT)),
            "manifest_sha256": sha256_path(manifest_path),
            "manifest_output_sha256": manifest["output_sha256"],
            "population_records": contract["source"]["records"],
            "records": selected_records,
            "rows": len(rows),
            "configurations": contract["source"]["configurations"],
            "selection": selection,
        },
        "tokenizer": {
            "path": str(tokenizer_path.relative_to(REPOSITORY_ROOT)),
            "files": tokenizer_files,
            "transformers": transformers_version,
            "vocabulary_fragments": len(vocabulary.fragments),
            "raw_byte_fallbacks": len(vocabulary.byte_token_ids),
        },
        "audit": {
            "counterfactual_horizons": contract["audit"][
                "counterfactual_horizons"
            ],
            "cells": cells,
        },
        "integrity": {
            "complete_source_contract": True,
            "identical_B_W_id_sets": True,
            "all_selected_rows_retained": sum(cell["attempted"] for cell in cells)
            == len(rows) * len(contract["audit"]["counterfactual_horizons"]),
            "all_source_rows_retained": selection is None,
            "model_weights_loaded": False,
            "gpu_used": False,
            "claim_boundary": contract["claim_boundary"],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config_path = args.config.resolve()
    contract = load_contract(config_path)
    rows, manifest, predictions_path, manifest_path = load_source_rows(contract)
    tokenizer_path = resolve_repository_path(contract["model"]["adapter"])
    if not tokenizer_path.is_dir():
        parser.error(f"selected tokenizer directory is missing: {tokenizer_path}")
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        parser.error("install the LoRA runtime with: pip install -e '.[lora]'")
        raise AssertionError("unreachable") from error
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    report = build_report(
        contract=contract,
        config_path=config_path,
        rows=rows,
        manifest=manifest,
        predictions_path=predictions_path,
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        transformers_version=transformers.__version__,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    maximum = int(contract["resource_contract"]["maximum_output_bytes"])
    if len(rendered.encode("utf-8")) > maximum:
        raise ValueError("compact horizon-stress report exceeds its output ceiling")
    if args.verify_existing is not None:
        expected = args.verify_existing.read_text(encoding="utf-8")
        if expected != rendered:
            raise ValueError(f"stored horizon-stress report differs: {args.verify_existing}")
        print(f"verified horizon-stress report: {args.verify_existing}")
        return 0
    assert args.output is not None
    if args.output.exists():
        parser.error(f"refusing to overwrite existing report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote horizon-stress report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
