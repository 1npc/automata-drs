#!/usr/bin/env python3
"""Write or verify the selected-host method-suite resource contract."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.method_resource import build_method_resource_plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/lora_baselines.json")
    )
    parser.add_argument(
        "--selection", type=Path, default=Path("runs/lora/selected-host.json")
    )
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()

    try:
        config_bytes = args.config.read_bytes()
        selection_bytes = args.selection.read_bytes()
        config = json.loads(config_bytes.decode("utf-8"))
        selection = json.loads(selection_bytes.decode("utf-8"))
        plan = build_method_resource_plan(
            config,
            selection,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            selection_sha256=hashlib.sha256(selection_bytes).hexdigest(),
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))

    if args.verify_existing is not None:
        try:
            stored = json.loads(args.verify_existing.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"cannot read stored resource plan: {error}")
        observed = {key: value for key, value in stored.items() if key != "created_at_utc"}
        if observed != plan:
            parser.error("stored method resource plan differs from current evidence")
        print("verified selected-host method resource plan")
        return 0

    assert args.output is not None
    plan["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as destination_file:
            json.dump(plan, destination_file, ensure_ascii=False, indent=2)
            destination_file.write("\n")
    except FileExistsError:
        parser.error(f"refusing to overwrite resource plan: {args.output}")
    print(
        "method resource ceiling passed: "
        f"{plan['upper_bounds']['generated_surfaces']} surfaces, "
        f"{plan['upper_bounds']['theoretical_output_tokens']} tokens"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
