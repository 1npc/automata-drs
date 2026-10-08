#!/usr/bin/env python3
"""Audit process-unique Hugging Face cache partials without deleting them."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


OWNED_LORA_CACHE = re.compile(r"^automata-drs-lora\.[A-Za-z0-9]+$")
PROCESS_TEMP = re.compile(r"^(?P<blob>.+)\.(?P<nonce>[0-9a-fA-F]{8})\.incomplete$")


def resolve_owned_cache(cache: Path, scratch_root: Path) -> tuple[Path, Path]:
    resolved_root = scratch_root.resolve(strict=True)
    resolved_cache = cache.resolve(strict=True)
    if not resolved_root.is_dir() or resolved_root == Path("/"):
        raise ValueError("scratch root must be an existing non-root directory")
    if not resolved_cache.is_dir():
        raise ValueError("LoRA cache must be an existing directory")
    if resolved_cache.parent != resolved_root:
        raise ValueError("LoRA cache is outside the exact scratch root")
    if not OWNED_LORA_CACHE.fullmatch(resolved_cache.name):
        raise ValueError("LoRA cache name is not an owned automata-drs cache")
    hub_root = resolved_cache / "huggingface" / "hub"
    if not hub_root.is_dir():
        raise ValueError("LoRA cache has no Hugging Face hub directory")
    return resolved_cache, hub_root


def audit_cache(cache: Path, scratch_root: Path) -> dict[str, object]:
    resolved_cache, hub_root = resolve_owned_cache(cache, scratch_root)
    rows: list[dict[str, object]] = []
    for partial in sorted(hub_root.rglob("*.incomplete")):
        if not partial.is_file():
            continue
        stat = partial.stat()
        match = PROCESS_TEMP.fullmatch(partial.name)
        complete_blob = partial.with_name(match.group("blob")) if match else None
        shadowed = bool(complete_blob is not None and complete_blob.is_file())
        rows.append(
            {
                "path": str(partial.relative_to(resolved_cache)),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "kind": (
                    "process-unique-partial"
                    if match
                    else "unrecognized-incomplete-file"
                ),
                "complete_blob": (
                    str(complete_blob.relative_to(resolved_cache))
                    if shadowed and complete_blob is not None
                    else None
                ),
                "safe_to_remove_after_process_exit": shadowed,
            }
        )
    total_bytes = sum(int(row["bytes"]) for row in rows)
    reclaimable_bytes = sum(
        int(row["bytes"])
        for row in rows
        if row["safe_to_remove_after_process_exit"] is True
    )
    return {
        "schema_version": 1,
        "kind": "huggingface-process-partial-audit",
        "cache": str(resolved_cache),
        "hub": str(hub_root),
        "deletes_files": False,
        "partial_files": len(rows),
        "partial_bytes": total_bytes,
        "proven_reclaimable_files_after_process_exit": sum(
            row["safe_to_remove_after_process_exit"] is True for row in rows
        ),
        "proven_reclaimable_bytes_after_process_exit": reclaimable_bytes,
        "partials": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        report = audit_cache(args.cache, args.scratch_root)
    except ValueError as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    if args.output.exists():
        parser.error(f"refusing to overwrite existing audit: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
