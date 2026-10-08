#!/usr/bin/env python3
"""Freeze the development-only LoRA host selection after all baselines exist."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.host_selection import select_lora_host  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/lora_baselines.json")
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs/lora"))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/lora/selected-host.json")
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite frozen selection: {args.output}")
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    try:
        selection = select_lora_host(
            config, args.runs_root, repository_root=REPOSITORY_ROOT
        )
    except ValueError as error:
        parser.error(str(error))
    selection["selected_at_utc"] = datetime.now(timezone.utc).isoformat()
    selection["config_path"] = str(args.config)
    selection["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(selection["selected_key"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
