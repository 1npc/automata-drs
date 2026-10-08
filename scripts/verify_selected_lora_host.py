#!/usr/bin/env python3
"""Verify and print the frozen recent-model constrained-decoding host."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.host_selection import (  # noqa: E402
    verify_frozen_lora_host_selection,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/lora_baselines.json")
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs/lora"))
    parser.add_argument(
        "--selection", type=Path, default=Path("runs/lora/selected-host.json")
    )
    args = parser.parse_args()
    try:
        config_bytes = args.config.read_bytes()
        config = json.loads(config_bytes.decode("utf-8"))
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(selection, dict):
            raise ValueError("config and selection must be JSON objects")
        selected = verify_frozen_lora_host_selection(
            config,
            config_bytes,
            selection,
            args.runs_root,
            repository_root=REPOSITORY_ROOT,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
