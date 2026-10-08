#!/usr/bin/env python3
"""Replay frozen U/P first divergences at the exact AMR BPE boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import (  # noqa: E402
    ByteLevelTokenConstraintController,
    ByteLevelVocabulary,
    PENMANConstraintState,
    PENMANMode,
)


_RECORD_WHITESPACE_BYTES = b" \n\t\r\f\v"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_fragment_is_allowed(state: Any, fragment: bytes) -> bool:
    """Replay the pre-fix rule: any char after COMPLETE is rejected."""

    if state.pending_utf8:
        return False
    try:
        text = fragment.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    representation = state.representation
    for character in text:
        if representation.mode is PENMANMode.COMPLETE:
            return False
        representation = representation._advance_char(character)
        if representation.is_dead:
            return False
    return representation.is_extendable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_records <= 0:
        parser.error("expected-records must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        parser.error(f"Transformers is required for exact-token replay: {error}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=str(args.cache_dir),
        local_files_only=True,
    )
    vocabulary = ByteLevelVocabulary.from_tokenizer(tokenizer)
    rows = [
        json.loads(line)
        for line in args.baseline.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    paired: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        configuration = row.get("configuration")
        if configuration not in {"U", "P"}:
            continue
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            parser.error("baseline row has no stable id")
        if configuration in paired.setdefault(identifier, {}):
            parser.error(f"duplicate {configuration} row for {identifier}")
        paired[identifier][configuration] = row

    evidence: list[dict[str, Any]] = []
    for identifier in sorted(paired):
        pair = paired[identifier]
        if set(pair) != {"U", "P"}:
            parser.error(f"incomplete U/P pair: {identifier}")
        u_surface = pair["U"].get("transport_envelope", {}).get("model_suffix")
        p_surface = pair["P"].get("transport_envelope", {}).get("model_suffix")
        if not isinstance(u_surface, str) or not isinstance(p_surface, str):
            parser.error(f"missing exact model suffix: {identifier}")
        u_ids = [
            int(value)
            for value in tokenizer(u_surface, add_special_tokens=False)["input_ids"]
        ]
        p_ids = [
            int(value)
            for value in tokenizer(p_surface, add_special_tokens=False)["input_ids"]
        ]
        common = 0
        while (
            common < len(u_ids)
            and common < len(p_ids)
            and u_ids[common] == p_ids[common]
        ):
            common += 1
        if common >= len(u_ids) or common >= len(p_ids):
            parser.error(f"U/P pair has no two-sided token divergence: {identifier}")

        controller = ByteLevelTokenConstraintController(
            vocabulary,
            incomplete_utf8_is_extendable=(
                PENMANConstraintState.incomplete_utf8_is_extendable
            ),
        )
        state = controller.initial(PENMANConstraintState.initial())
        for position, token_id in enumerate(u_ids[:common]):
            successor = controller.transition(state, token_id)
            if successor is None:
                parser.error(
                    f"common prefix dies at {identifier} token {position}"
                )
            state = successor

        u_token_id, p_token_id = u_ids[common], p_ids[common]
        u_fragment = vocabulary.fragments[u_token_id]
        p_fragment = vocabulary.fragments[p_token_id]
        stripped = u_fragment.rstrip(_RECORD_WHITESPACE_BYTES)
        trailing = u_fragment[len(stripped) :]
        core = state.representation.advance_text(stripped.decode("utf-8"))
        evidence.append(
            {
                "id": identifier,
                "common_tokens": common,
                "u_next_token_id": u_token_id,
                "u_next_fragment": u_fragment.decode("utf-8"),
                "p_next_token_id": p_token_id,
                "p_next_fragment": p_fragment.decode("utf-8"),
                "legacy_u_next_allowed": _legacy_fragment_is_allowed(
                    state, u_fragment
                ),
                "corrected_u_next_allowed": (
                    controller.transition(state, u_token_id) is not None
                ),
                "p_next_allowed": controller.transition(state, p_token_id) is not None,
                "u_core_reaches_complete": (
                    not core.is_dead and core.mode is PENMANMode.COMPLETE
                ),
                "u_trailing_record_whitespace": trailing.decode("utf-8"),
                "u_token_crosses_transport_tag": b"</amr>" in u_fragment,
            }
        )

    row_gate = len(evidence) == args.expected_records and all(
        not row["legacy_u_next_allowed"]
        and row["corrected_u_next_allowed"]
        and row["p_next_allowed"]
        and row["u_core_reaches_complete"]
        and bool(row["u_trailing_record_whitespace"])
        and not row["u_token_crosses_transport_tag"]
        for row in evidence
    )
    report = {
        "schema_version": 1,
        "audit": "amr-exact-token-first-divergence",
        "baseline": {
            "path": str(args.baseline),
            "sha256": _sha256(args.baseline),
        },
        "model": args.model,
        "revision": args.revision,
        "expected_records": args.expected_records,
        "rows": evidence,
        "gate": {"passed": row_gate},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(f"gate passed: {row_gate}")
    return 0 if row_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
