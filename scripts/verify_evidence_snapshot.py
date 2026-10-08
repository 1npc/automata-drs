#!/usr/bin/env python3
"""Build or verify a compact, reviewer-facing evidence snapshot manifest.

The compact snapshot deliberately excludes prediction JSONL, checkpoints, and
corpora.  This verifier therefore does not pretend to recompute the experiments.
It checks the stronger claim that is available locally: both canonical suite
collectors passed the manuscript result lock before transfer, their rendered
Markdown is reproducible, and every compact JSON/Markdown input referenced by
the suites is present with the recorded SHA-256 digest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.lora_suite import lora_suite_to_markdown  # noqa: E402


FAMILIES = ("lora", "penman", "amr", "efficiency", "paper")
DRS_SUITE = PurePosixPath("runs/lora/recent-model-and-method-results-v3.json")
DRS_MARKDOWN = PurePosixPath("runs/lora/recent-model-and-method-results-v3.md")
AMR_SUITE = PurePosixPath("runs/amr/qwen35-4b-base-transfer-suite-v1.json")
AMR_MARKDOWN = PurePosixPath("runs/amr/qwen35-4b-base-transfer-suite-v1.md")
RESULT_PACKAGE = PurePosixPath("runs/paper/manuscript-result-package-v1.json")
RESULT_PACKAGE_MARKDOWN = PurePosixPath(
    "runs/paper/manuscript-result-package-v1.md"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load verifier module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_lock_builder(drs: Path, amr: Path, package: Path) -> dict[str, Any]:
    module = _load_script(
        "snapshot_manuscript_result_lock",
        REPOSITORY_ROOT / "scripts" / "check_manuscript_result_lock.py",
    )
    return module.build_report(drs, amr, package)


def _default_amr_renderer(report: dict[str, Any]) -> str:
    module = _load_script(
        "snapshot_amr_suite_renderer",
        REPOSITORY_ROOT / "scripts" / "summarize_amr_suite.py",
    )
    return module.to_markdown(report)


def _default_result_package_renderer(report: dict[str, Any]) -> str:
    module = _load_script(
        "snapshot_result_package_renderer",
        REPOSITORY_ROOT / "scripts" / "build_manuscript_result_package.py",
    )
    return module.to_markdown(report)


def _inventory(snapshot_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if not snapshot_root.is_dir() or snapshot_root.is_symlink():
        raise ValueError("snapshot root must be a real directory")
    rows: list[dict[str, Any]] = []
    digests: dict[str, str] = {}
    for family in FAMILIES:
        family_root = snapshot_root / "runs" / family
        if not family_root.exists():
            continue
        if not family_root.is_dir() or family_root.is_symlink():
            raise ValueError(f"snapshot family is not a real directory: {family_root}")
        for path in sorted(family_root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"snapshot contains a symlink: {path}")
            if not path.is_file():
                continue
            if path.suffix not in {".json", ".md"}:
                raise ValueError(f"snapshot contains an unregistered file: {path}")
            relative = path.relative_to(snapshot_root).as_posix()
            digest = _sha256(path)
            size = path.stat().st_size
            rows.append({"path": relative, "bytes": size, "sha256": digest})
            digests[relative] = digest
    if not rows:
        raise ValueError("compact evidence snapshot is empty")
    return rows, digests


def _portable_compact_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 3:
        return None
    if path.parts[0] != "runs" or path.parts[1] not in FAMILIES:
        return None
    if path.suffix not in {".json", ".md"}:
        return None
    return path.as_posix()


def _check_references(
    label: str,
    evidence_hashes: Any,
    snapshot_digests: dict[str, str],
) -> dict[str, Any]:
    if not isinstance(evidence_hashes, dict) or not evidence_hashes:
        raise ValueError(f"{label} suite has no evidence hash table")
    required: list[str] = []
    for source_path, expected_digest in evidence_hashes.items():
        compact_path = _portable_compact_path(source_path)
        if compact_path is None:
            continue
        if not _is_sha256(expected_digest):
            raise ValueError(f"{label} suite has an invalid digest: {source_path}")
        observed_digest = snapshot_digests.get(compact_path)
        if observed_digest is None:
            raise ValueError(f"{label} compact evidence is missing: {compact_path}")
        if observed_digest != expected_digest:
            raise ValueError(f"{label} compact evidence hash drifted: {compact_path}")
        required.append(compact_path)
    if not required:
        raise ValueError(f"{label} suite references no portable compact evidence")
    return {
        "required_compact_files": len(required),
        "all_present_and_hash_matched": True,
    }


def build_manifest(
    snapshot_root: Path,
    *,
    lock_builder: Callable[[Path, Path, Path], dict[str, Any]] = _default_lock_builder,
    drs_renderer: Callable[[dict[str, Any]], str] = lora_suite_to_markdown,
    amr_renderer: Callable[[dict[str, Any]], str] = _default_amr_renderer,
    result_package_renderer: Callable[[dict[str, Any]], str] = (
        _default_result_package_renderer
    ),
) -> dict[str, Any]:
    files, snapshot_digests = _inventory(snapshot_root)
    paths = {
        "drs_json": snapshot_root / str(DRS_SUITE),
        "drs_markdown": snapshot_root / str(DRS_MARKDOWN),
        "amr_json": snapshot_root / str(AMR_SUITE),
        "amr_markdown": snapshot_root / str(AMR_MARKDOWN),
        "result_package_json": snapshot_root / str(RESULT_PACKAGE),
        "result_package_markdown": snapshot_root / str(RESULT_PACKAGE_MARKDOWN),
    }
    for label, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing canonical {label}: {path}")
    drs = _read_object(paths["drs_json"])
    amr = _read_object(paths["amr_json"])
    result_package = _read_object(paths["result_package_json"])
    if paths["drs_markdown"].read_text(encoding="utf-8") != drs_renderer(drs):
        raise ValueError("DRS suite Markdown differs from the canonical renderer")
    if paths["amr_markdown"].read_text(encoding="utf-8") != amr_renderer(amr):
        raise ValueError("AMR suite Markdown differs from the canonical renderer")
    if paths["result_package_markdown"].read_text(
        encoding="utf-8"
    ) != result_package_renderer(result_package):
        raise ValueError(
            "manuscript result Markdown differs from the canonical renderer"
        )
    lock = lock_builder(
        paths["drs_json"], paths["amr_json"], paths["result_package_json"]
    )
    if (
        lock.get("state")
        != "all-suite-evidence-and-result-package-ready-for-integration"
        or lock.get("drs", {}).get("ready") is not True
        or lock.get("amr", {}).get("ready") is not True
        or lock.get("result_package", {}).get("ready") is not True
    ):
        raise ValueError("canonical DRS and AMR suites do not pass the result lock")
    drs_references = _check_references(
        "DRS", drs.get("evidence", {}).get("input_sha256"), snapshot_digests
    )
    amr_references = _check_references(
        "AMR", amr.get("evidence"), snapshot_digests
    )
    return {
        "schema_version": 1,
        "audit": "compact-evidence-snapshot",
        "scope": {
            "included": "size-capped JSON/Markdown reviewer evidence",
            "excluded": [
                "prediction JSONL",
                "model checkpoints and adapters",
                "raw corpora",
                "logs and caches",
            ],
            "independent_raw_recomputation_claimed": False,
        },
        "inventory": {
            "files": len(files),
            "bytes": sum(row["bytes"] for row in files),
            "entries": files,
        },
        "canonical_suites": {
            "drs": {
                "path": str(DRS_SUITE),
                "sha256": snapshot_digests[str(DRS_SUITE)],
                "markdown_reproduced": True,
                "references": drs_references,
            },
            "amr": {
                "path": str(AMR_SUITE),
                "sha256": snapshot_digests[str(AMR_SUITE)],
                "markdown_reproduced": True,
                "references": amr_references,
            },
            "manuscript_result_package": {
                "path": str(RESULT_PACKAGE),
                "sha256": snapshot_digests[str(RESULT_PACKAGE)],
                "markdown_reproduced": True,
            },
        },
        "result_lock": {
            "state": lock["state"],
            "drs_ready": True,
            "amr_ready": True,
            "result_package_ready": True,
        },
        "gate": {"passed": True, "performance_threshold_used": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_root", type=Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()
    try:
        report = build_manifest(args.snapshot_root)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    if args.verify_existing is not None:
        try:
            stored = _read_object(args.verify_existing)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            parser.error(str(error))
        if stored != report:
            parser.error("stored compact evidence manifest differs from current files")
        print(f"verified compact evidence snapshot: {args.snapshot_root}")
        return 0
    assert args.output is not None
    if args.output.exists():
        parser.error(f"refusing to overwrite snapshot manifest: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as destination_file:
            json.dump(report, destination_file, ensure_ascii=False, indent=2)
            destination_file.write("\n")
    except FileExistsError:
        parser.error(f"refusing to overwrite snapshot manifest: {args.output}")
    print(f"compact evidence snapshot verified: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
