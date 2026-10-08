#!/usr/bin/env python3
"""Build one comparison artifact from the recent-model LoRA run summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.lora_suite import (  # noqa: E402
    collect_lora_suite,
    lora_suite_to_markdown,
    verify_lora_suite_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/lora_baselines.json")
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs/lora"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument(
        "--scope",
        choices=("baseline", "full"),
        default=None,
        help=(
            "artifact dependency scope; omitted only for backward-compatible "
            "inference from the canonical baseline output names"
        ),
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="recompute and verify the two existing outputs instead of writing",
    )
    args = parser.parse_args()

    scope = args.scope
    if scope is None:
        canonical_baseline_names = (
            args.output_json.name == "recent-model-baselines.json"
            and args.output_markdown.name == "recent-model-baselines.md"
        )
        scope = "baseline" if canonical_baseline_names else "full"

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    try:
        suite = collect_lora_suite(
            config,
            args.runs_root,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            config_bytes=config_bytes,
            repository_root=REPOSITORY_ROOT,
            scope=scope,
            require_complete_methods=(scope == "full"),
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.verify_existing:
        if not args.output_json.is_file() or not args.output_markdown.is_file():
            parser.error("both stored suite outputs are required for verification")
        try:
            verify_lora_suite_outputs(
                suite, args.output_json, args.output_markdown
            )
        except ValueError as error:
            parser.error(str(error))
        print("verified LoRA suite outputs against current evidence")
        return 0
    if args.output_json.exists() or args.output_markdown.exists():
        parser.error("refusing to overwrite suite output")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(suite, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(
        lora_suite_to_markdown(suite), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
