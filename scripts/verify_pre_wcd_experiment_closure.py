#!/usr/bin/env python3
"""Verify the immutable pre-WCD closure and every hash-bound dependency."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"closure dependency must be repository-relative: {value}")
    resolved = (REPOSITORY_ROOT / path).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"closure dependency escapes repository: {value}") from error
    return resolved


def dependency_pairs(value: Any) -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        path = value.get("path")
        digest = value.get("sha256")
        if isinstance(path, str) and isinstance(digest, str):
            yield path, digest
        legacy_path = value.get("v2_summary")
        legacy_digest = value.get("v2_summary_sha256")
        if isinstance(legacy_path, str) and isinstance(legacy_digest, str):
            yield legacy_path, legacy_digest
        for child in value.values():
            yield from dependency_pairs(child)
    elif isinstance(value, list):
        for child in value:
            yield from dependency_pairs(child)


def verify_closure(path: Path, expected_sha256: str | None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"closure is missing or unsafe: {path}")
    observed_sha256 = sha256_path(path)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError("pre-WCD closure hash changed")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1:
        raise ValueError("unsupported pre-WCD closure schema")
    if report.get("audit") != "pre-wcd-experiment-closure":
        raise ValueError("unexpected pre-WCD closure kind")
    if report.get("status") != "closed":
        raise ValueError("pre-WCD closure is not closed")
    if report.get("primary_gates_passed") is not True:
        raise ValueError("pre-WCD primary gates did not all pass")
    gate = report.get("gate") or {}
    required_gate_fields = (
        "passed",
        "all_primary_drs_gates_passed",
        "all_primary_amr_gates_passed",
        "all_amr_alignments_optimal",
    )
    if any(gate.get(name) is not True for name in required_gate_fields):
        raise ValueError("pre-WCD closure gate is incomplete")

    checked: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for relative, expected in dependency_pairs(report):
        pair = (relative, expected)
        if pair in seen:
            continue
        seen.add(pair)
        dependency = repository_path(relative)
        if not dependency.is_file() or dependency.is_symlink():
            raise ValueError(f"closure dependency is missing or unsafe: {relative}")
        observed = sha256_path(dependency)
        if observed != expected:
            raise ValueError(f"closure dependency hash changed: {relative}")
        checked.append({"path": relative, "sha256": observed})
    if not checked:
        raise ValueError("pre-WCD closure contains no hash-bound dependencies")
    return {
        "closure": str(path),
        "closure_sha256": observed_sha256,
        "dependencies_verified": len(checked),
        "gate_passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--closure",
        type=Path,
        default=Path("runs/closure/pre-wcd-experiments-v1.json"),
    )
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    try:
        result = verify_closure(args.closure, args.expected_sha256)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
