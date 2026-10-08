#!/usr/bin/env python3
"""Verify all recent-model dev/test/tax baseline outputs and core summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.lora_run_state import (  # noqa: E402
    verify_complete_lora_baseline_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/lora_baselines.json")
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs/lora"))
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        report = verify_complete_lora_baseline_outputs(
            config,
            args.runs_root,
            REPOSITORY_ROOT,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(
        "verified complete baseline outputs: "
        + ", ".join(report["models"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
