#!/usr/bin/env python3
"""Check whether a recent-model LoRA smoke run may advance to full training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.lora_gate import evaluate_lora_smoke_gate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_key")
    parser.add_argument("--config", type=Path, default=Path("configs/lora_baselines.json"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs/lora"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    matches = [row for row in config["models"] if row["key"] == args.model_key]
    if len(matches) != 1:
        parser.error(f"unknown or duplicate model key: {args.model_key}")
    run_root = args.runs_root / f"{args.model_key}-smoke"
    report = evaluate_lora_smoke_gate(
        matches[0],
        run_root,
        max_length=int(config["shared_recipe"]["max_length"]),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            parser.error(f"output already exists: {args.output}")
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
