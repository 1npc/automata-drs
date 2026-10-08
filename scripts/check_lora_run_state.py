#!/usr/bin/env python3
"""Inspect one recent-model LoRA baseline before a safe staged resume."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.lora_run_state import inspect_lora_baseline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_key")
    parser.add_argument("--config", type=Path, default=Path("configs/lora_baselines.json"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs/lora"))
    parser.add_argument(
        "--field",
        choices=["next_action", "smoke.state", "full.state"],
        help="print one shell-friendly field instead of the JSON report",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        report = inspect_lora_baseline(
            config, args.model_key, args.runs_root, REPOSITORY_ROOT
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.field == "next_action":
        print(report["next_action"])
    elif args.field == "smoke.state":
        print(report["smoke"]["state"])
    elif args.field == "full.state":
        print(report["full"]["state"])
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["resumable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
