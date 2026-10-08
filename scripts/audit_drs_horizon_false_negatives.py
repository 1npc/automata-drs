#!/usr/bin/env python3
"""Audit frozen W paths for constructive-planner false negatives.

The audit is deliberately post-hoc and model-free.  It replays every W path
from a hash-bound generation matrix through the corresponding WF admission
rule.  A rejection is called a false negative only when the independently
verified remainder of that same native-token path reaches accepting EOS
inside the remaining budget.  Planner failure alone remains UNKNOWN.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import SBNPolicy  # noqa: E402
from automata_drs.certificates import verify_accepting_certificate  # noqa: E402
from automata_drs.subword import (  # noqa: E402
    ByteLevelVocabulary,
    CausalPrefixStateCache,
    SubwordSBNLogitsProcessor,
)


DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/wcd_drs_horizon_test128_false_negative_audit_v1.json"
)


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
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported false-negative audit schema")
    if contract.get("kind") != "wcd-drs-heldout-horizon-false-negative-audit":
        raise ValueError("unexpected false-negative audit kind")
    if contract.get("status") != "registered-posthoc-diagnostic":
        raise ValueError("audit must remain explicitly post-hoc")
    source = contract.get("source") or {}
    if source.get("records") != 128:
        raise ValueError("audit must retain all 128 registered test rows")
    if source.get("configurations") != ["W", "WF", "WWCD"]:
        raise ValueError("source configuration order changed")
    if source.get("max_new_tokens") != 48:
        raise ValueError("source native-token horizon changed")
    audit = contract.get("audit") or {}
    if audit.get("population") != "all-128-W-rows-in-source-order":
        raise ValueError("audit population changed")
    if audit.get("performance_conditioned_selection") is not False:
        raise ValueError("false-negative audit must retain the full W population")
    resources = contract.get("resource_contract") or {}
    if resources.get("model_weights_loaded") is not False:
        raise ValueError("false-negative audit must remain model-free")
    if resources.get("gpu_used") is not False:
        raise ValueError("false-negative audit must remain CPU-only")
    if resources.get("cpu_math_threads") != 1:
        raise ValueError("false-negative audit must remain one-threaded")
    return contract


def verify_registered_evidence(contract: dict[str, Any]) -> dict[str, Path]:
    source = contract["source"]
    paths: dict[str, Path] = {}
    for name in (
        "experiment_contract",
        "predictions",
        "prediction_manifest",
        "scores",
        "score_manifest",
        "gate",
    ):
        evidence = source[name]
        path = resolve_repository_path(evidence["path"])
        if not path.is_file():
            raise FileNotFoundError(f"missing registered evidence: {path}")
        observed = sha256_path(path)
        if observed != evidence["sha256"]:
            raise ValueError(f"registered evidence hash changed: {path}")
        paths[name] = path
    adapter = resolve_repository_path(source["adapter"])
    adapter_config = adapter / "adapter_config.json"
    if not adapter_config.is_file():
        raise FileNotFoundError(f"missing registered adapter config: {adapter_config}")
    if sha256_path(adapter_config) != source["adapter_config_sha256"]:
        raise ValueError("registered adapter config hash changed")
    paths["adapter"] = adapter
    return paths


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_source(
    contract: dict[str, Any], paths: dict[str, Path]
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float]]:
    source = contract["source"]
    records = int(source["records"])
    configurations = list(source["configurations"])
    expected_rows = records * len(configurations)
    manifest = json.loads(paths["prediction_manifest"].read_text(encoding="utf-8"))
    expected_manifest = {
        "status": "complete",
        "start": 0,
        "limit": records,
        "max_new_tokens": int(source["max_new_tokens"]),
        "num_beams": 1,
        "batch_size": 1,
        "configurations": configurations,
        "expected_examples": records,
        "expected_rows": expected_rows,
        "examples": records,
        "rows": expected_rows,
        "output_sha256": source["predictions"]["sha256"],
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected_manifest.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"prediction manifest mismatch: {mismatches}")
    rows = load_jsonl(paths["predictions"])
    if len(rows) != expected_rows:
        raise ValueError(f"prediction rows={len(rows)}, expected {expected_rows}")
    counts = Counter(str(row.get("configuration")) for row in rows)
    if counts != Counter({configuration: records for configuration in configurations}):
        raise ValueError(f"prediction configuration counts changed: {dict(counts)}")
    identities = [(str(row.get("id")), str(row.get("configuration"))) for row in rows]
    if len(set(identities)) != expected_rows:
        raise ValueError("prediction identities are not unique")
    id_sets = [
        {str(row["id"]) for row in rows if row["configuration"] == configuration}
        for configuration in configurations
    ]
    if any(identifier_set != id_sets[0] for identifier_set in id_sets[1:]):
        raise ValueError("paired configuration ID sets differ")

    scores = load_jsonl(paths["scores"])
    if len(scores) != expected_rows:
        raise ValueError(f"score rows={len(scores)}, expected {expected_rows}")
    score_map: dict[tuple[str, str], float] = {}
    for row in scores:
        key = (str(row.get("id")), str(row.get("configuration")))
        value = (row.get("semantic_scores") or {}).get("sbn_smatch")
        if key in score_map or not isinstance(value, (int, float)):
            raise ValueError(f"invalid or duplicate semantic score: {key}")
        score_map[key] = float(value)
    if set(score_map) != set(identities):
        raise ValueError("prediction and score identities differ")
    score_manifest = json.loads(paths["score_manifest"].read_text(encoding="utf-8"))
    if (
        score_manifest.get("input_sha256") != source["predictions"]["sha256"]
        or score_manifest.get("output_sha256") != source["scores"]["sha256"]
        or score_manifest.get("records") != expected_rows
        or score_manifest.get("identical_pair_score_consistency") is not True
    ):
        raise ValueError("score manifest violates the registered source")
    gate = json.loads(paths["gate"].read_text(encoding="utf-8"))
    if gate.get("gate", {}).get("passed") is not True:
        raise ValueError("registered generation gate did not pass")
    return rows, score_map


def exact_surface_ids(
    tokenizer: Any, vocabulary: ByteLevelVocabulary, surface: str
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


def token_fragment(vocabulary: ByteLevelVocabulary, token_id: int) -> dict[str, Any]:
    fragment = vocabulary.fragments.get(token_id)
    if fragment is None:
        return {"hex": None, "utf8": None}
    try:
        text = fragment.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        text = None
    return {"hex": fragment.hex(), "utf8": text}


def audit_w_path(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    vocabulary: ByteLevelVocabulary,
    horizon: int,
) -> dict[str, Any]:
    policy = SBNPolicy.for_configuration("W")
    processor = SubwordSBNLogitsProcessor(
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        prompt_token_ids=(),
        policy=policy,
        max_new_tokens=horizon,
        bounded_completion=True,
        completion_budget_factor=1.0,
        max_completion_cache_entries=20_000,
    )
    processor.prefix_states = CausalPrefixStateCache(
        processor.controller, prompt_token_ids=(), policy=policy
    )
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer has no EOS token")
    path = exact_surface_ids(tokenizer, vocabulary, str(row.get("prediction", ""))) + (
        int(eos_token_id),
    )
    generation = row.get("generation") or {}
    recorded_tokens = generation.get("output_tokens_including_eos")
    canonical = generation.get("canonical_tokenization") is True
    source_complete = (
        generation.get("ended_with_eos") is True
        and generation.get("hit_max_new_tokens") is False
    )
    auditable = (
        canonical
        and source_complete
        and row.get("validity", {}).get("raw_sbn") is True
        and row.get("validity", {}).get("raw_clf_referee") is True
        and recorded_tokens == len(path)
        and len(path) <= horizon
    )
    result: dict[str, Any] = {
        "id": str(row.get("id")),
        "path_tokens_including_eos": len(path),
        "recorded_tokens_including_eos": recorded_tokens,
        "path_within_horizon": len(path) <= horizon,
        "canonical_tokenization": canonical,
        "source_complete": source_complete,
        "source_raw_sbn_valid": row.get("validity", {}).get("raw_sbn") is True,
        "source_raw_referee_valid": row.get("validity", {}).get("raw_clf_referee") is True,
        "auditable": auditable,
        "planner_accepts_entire_path": False,
        "false_negative_events": [],
        "other_rejection_events": [],
    }
    if not auditable:
        return result

    prefix: list[int] = []
    rejected = False
    for token_index, token_id in enumerate(path):
        state = processor.prefix_states.state_for(prefix)
        if state is None:
            result["other_rejection_events"].append(
                {"step_one_based": token_index + 1, "reason": "dead-source-prefix"}
            )
            rejected = True
            break
        remaining_after_token = horizon - token_index - 1
        successor = processor.controller._compute_transition(state, token_id)
        planner = (
            None
            if successor is None or remaining_after_token < 0
            else processor.certificate_within(successor, remaining_after_token)
        )
        planner_accepts = planner is not None and planner.certificate is not None
        if successor is None or not planner_accepts:
            rejected = True
            residual = path[token_index + 1 :]
            verified = (
                None
                if successor is None
                else verify_accepting_certificate(
                    successor,
                    residual,
                    transition=processor.controller.transition,
                    is_accepting=lambda current: current.is_accepting,
                    max_steps=remaining_after_token,
                )
            )
            event = {
                "step_one_based": token_index + 1,
                "token_id": int(token_id),
                "token_fragment": token_fragment(vocabulary, int(token_id)),
                "remaining_budget_after_token": remaining_after_token,
                "observed_residual_tokens": len(residual),
                "planner_status": None if planner is None else planner.status.value,
                "planner_reason": None if planner is None else planner.reason,
                "observed_residual_verifier_accepted": (
                    verified is not None and verified.accepted
                ),
                "observed_residual_failure_reason": (
                    None if verified is None or verified.accepted else verified.failure_reason
                ),
                "decoded_prefix_before_token": tokenizer.decode(
                    prefix,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "decoded_observed_residual": tokenizer.decode(
                    residual,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
            }
            if (
                successor is not None
                and planner is not None
                and planner.certificate is None
                and verified is not None
                and verified.accepted
            ):
                result["false_negative_events"].append(event)
            else:
                result["other_rejection_events"].append(event)
        prefix.append(int(token_id))
    final_state = processor.prefix_states.state_for(prefix)
    result["source_path_replay_accepting"] = bool(
        final_state is not None and final_state.is_accepting
    )
    result["planner_accepts_entire_path"] = not rejected
    return result


def build_report(
    *,
    contract: dict[str, Any],
    config_path: Path,
    paths: dict[str, Path],
    rows: list[dict[str, Any]],
    score_map: dict[tuple[str, str], float],
    tokenizer: Any,
    transformers_version: str,
) -> dict[str, Any]:
    vocabulary = ByteLevelVocabulary.from_tokenizer(tokenizer)
    configurations = contract["source"]["configurations"]
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    id_order: list[str] = []
    for row in rows:
        identifier = str(row["id"])
        if identifier not in by_id:
            by_id[identifier] = {}
            id_order.append(identifier)
        by_id[identifier][str(row["configuration"])] = row
    if any(set(by_id[identifier]) != set(configurations) for identifier in id_order):
        raise ValueError("source matrix is not paired")

    horizon = int(contract["source"]["max_new_tokens"])
    outcomes = [
        audit_w_path(
            by_id[identifier]["W"],
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            horizon=horizon,
        )
        for identifier in id_order
    ]
    maximum_examples = int(contract["audit"]["maximum_examples"])
    examples: list[dict[str, Any]] = []
    rows_with_false_negative = 0
    false_negative_events = 0
    other_rejection_events = 0
    changed_rows = 0
    changed_false_negative_rows = 0
    score_deltas: list[float] = []
    for outcome in outcomes:
        identifier = outcome["id"]
        false_count = len(outcome["false_negative_events"])
        other_count = len(outcome["other_rejection_events"])
        rows_with_false_negative += int(false_count > 0)
        false_negative_events += false_count
        other_rejection_events += other_count
        changed = (
            by_id[identifier]["W"]["prediction"]
            != by_id[identifier]["WF"]["prediction"]
        )
        changed_rows += int(changed)
        changed_false_negative_rows += int(changed and false_count > 0)
        delta = score_map[(identifier, "WF")] - score_map[(identifier, "W")]
        score_deltas.append(delta)
        if (false_count or other_count) and len(examples) < maximum_examples:
            examples.append(
                {
                    **outcome,
                    "input": by_id[identifier]["W"].get("input"),
                    "prediction_changed_W_to_WF": changed,
                    "W_prediction": by_id[identifier]["W"].get("prediction"),
                    "WF_prediction": by_id[identifier]["WF"].get("prediction"),
                    "WWCD_prediction": by_id[identifier]["WWCD"].get("prediction"),
                    "W_smatch": score_map[(identifier, "W")],
                    "WF_smatch": score_map[(identifier, "WF")],
                    "WWCD_smatch": score_map[(identifier, "WWCD")],
                    "WF_minus_W_smatch": delta,
                }
            )

    adapter = paths["adapter"]
    tokenizer_files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = adapter / name
        if path.is_file():
            tokenizer_files[name] = sha256_path(path)
    summary = {
        "rows": len(outcomes),
        "auditable_rows": sum(int(outcome["auditable"]) for outcome in outcomes),
        "source_path_replay_accepting_rows": sum(
            int(outcome.get("source_path_replay_accepting") is True)
            for outcome in outcomes
        ),
        "planner_accepts_entire_path_rows": sum(
            int(outcome["planner_accepts_entire_path"]) for outcome in outcomes
        ),
        "rows_with_constructive_planner_false_negative": rows_with_false_negative,
        "constructive_planner_false_negative_events": false_negative_events,
        "other_rejection_events": other_rejection_events,
        "W_to_WF_prediction_changed_rows": changed_rows,
        "changed_rows_with_constructive_planner_false_negative": (
            changed_false_negative_rows
        ),
        "mean_WF_minus_W_smatch": sum(score_deltas) / len(score_deltas),
    }
    return {
        "schema_version": 1,
        "kind": "wcd-drs-heldout-horizon-false-negative-audit-report",
        "status": "passed" if other_rejection_events == 0 else "diagnostic-failure",
        "contract": {
            "path": str(config_path.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_path(config_path),
            "implementation": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
            "implementation_sha256": sha256_path(Path(__file__).resolve()),
            "registration_boundary": contract["registration_boundary"],
        },
        "source": {
            name: {
                "path": str(path.relative_to(REPOSITORY_ROOT)),
                "sha256": sha256_path(path),
            }
            for name, path in paths.items()
            if name != "adapter"
        },
        "tokenizer": {
            "adapter": str(adapter.relative_to(REPOSITORY_ROOT)),
            "files": tokenizer_files,
            "transformers": transformers_version,
            "vocabulary_fragments": len(vocabulary.fragments),
            "raw_byte_fallbacks": len(vocabulary.byte_token_ids),
        },
        "summary": summary,
        "examples": examples,
        "integrity": {
            "all_W_rows_retained": len(outcomes) == int(contract["source"]["records"]),
            "performance_conditioned_selection": False,
            "model_weights_loaded": False,
            "gpu_used": False,
            "posthoc_outcomes_known_before_registration": True,
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
    paths = verify_registered_evidence(contract)
    rows, score_map = validate_source(contract, paths)
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        parser.error("install the LoRA runtime with: pip install -e '.[lora]'")
        raise AssertionError("unreachable") from error
    tokenizer = AutoTokenizer.from_pretrained(paths["adapter"], local_files_only=True)
    report = build_report(
        contract=contract,
        config_path=config_path,
        paths=paths,
        rows=rows,
        score_map=score_map,
        tokenizer=tokenizer,
        transformers_version=transformers.__version__,
    )
    if report["status"] != "passed":
        raise ValueError("source path replay produced non-false-negative rejection events")
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    maximum = int(contract["resource_contract"]["maximum_output_bytes"])
    if len(rendered.encode("utf-8")) > maximum:
        raise ValueError("false-negative audit report exceeds its output ceiling")
    if args.verify_existing is not None:
        if args.verify_existing.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"stored audit differs: {args.verify_existing}")
        print(f"verified false-negative audit: {args.verify_existing}")
        return 0
    assert args.output is not None
    if args.output.exists():
        parser.error(f"refusing to overwrite existing report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    print(f"wrote false-negative audit: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
