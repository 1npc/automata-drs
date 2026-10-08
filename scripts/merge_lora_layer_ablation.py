#!/usr/bin/env python3
"""Merge selected-host pilot U/BF/WF with generated B/W cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.lora_layer_ablation import (  # noqa: E402
    combine_layer_ablation_rows,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/lora_baselines.json")
    )
    args = parser.parse_args()
    output_manifest = Path(str(args.output) + ".manifest.json")
    if args.output.exists() or output_manifest.exists():
        parser.error(f"refusing to overwrite derived layer matrix: {args.output}")
    for source_path in (args.pilot, args.supplement):
        if not source_path.is_file() or not Path(
            str(source_path) + ".manifest.json"
        ).is_file():
            parser.error(f"source matrix or manifest is missing: {source_path}")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        ablation = config["method_experiment"]["layer_ablation"]
        if ablation["generated_configurations"] != ["B", "W"]:
            raise ValueError("frozen supplement is not B/W")
        if ablation["derived_full_ladder"] != ["U", "B", "W", "BF", "WF"]:
            raise ValueError("frozen derived ladder is not U/B/W/BF/WF")
        rows, evidence = combine_layer_ablation_rows(
            read_jsonl(args.pilot),
            read_jsonl(args.supplement),
            expected_records=int(ablation["records"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    sources = []
    for path in (args.pilot, args.supplement):
        manifest_path = Path(str(path) + ".manifest.json")
        sources.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
            }
        )
    output_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "kind": "derived-lora-layer-ablation",
                "output": str(args.output),
                "output_sha256": sha256(args.output),
                "sources": sources,
                **evidence,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"merged {len(rows)} immutable cells into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
