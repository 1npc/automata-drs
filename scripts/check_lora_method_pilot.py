#!/usr/bin/env python3
"""Check the selected-host method pilot before publication-scale matrices."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.host_selection import (  # noqa: E402
    verify_frozen_lora_host_selection,
)
from automata_drs.lora_method_gate import (  # noqa: E402
    evaluate_lora_method_pilot,
    verify_frozen_lora_method_pilot,
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.verify_existing is not None:
        parser.error("--output and --verify-existing are mutually exclusive")
    try:
        config_bytes = args.config.read_bytes()
        selection_bytes = args.selection.read_bytes()
        config = json.loads(config_bytes.decode("utf-8"))
        selection = json.loads(selection_bytes.decode("utf-8"))
        selected_key = verify_frozen_lora_host_selection(
            config, config_bytes, selection, args.runs_root
        )
        model_rows = [row for row in config["models"] if row["key"] == selected_key]
        if len(model_rows) != 1:
            raise ValueError("frozen host is absent or duplicated in model config")
        report = evaluate_lora_method_pilot(
            config, model_rows[0], args.runs_root
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    selection_sha = hashlib.sha256(selection_bytes).hexdigest()
    if args.verify_existing is not None:
        try:
            stored = json.loads(args.verify_existing.read_text(encoding="utf-8"))
            if not isinstance(stored, dict):
                raise ValueError("stored method pilot gate must be a JSON object")
            verify_frozen_lora_method_pilot(
                report,
                stored,
                config_sha256=config_sha,
                selection_sha256=selection_sha,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        print(f"verified method pilot gate: {selected_key}")
        return 0
    report["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["config_sha256"] = config_sha
    report["selection_sha256"] = selection_sha
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            parser.error(f"refusing to overwrite pilot gate: {args.output}")
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
