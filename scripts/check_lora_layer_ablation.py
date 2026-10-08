#!/usr/bin/env python3
"""Check and freeze the selected-host U/B/W/BF/WF layer ablation."""

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
from automata_drs.lora_layer_gate import (  # noqa: E402
    evaluate_lora_layer_ablation,
    verify_frozen_lora_layer_ablation,
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
    parser.add_argument("--pilot-gate", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite layer-ablation gate: {args.output}")
    try:
        config_bytes = args.config.read_bytes()
        selection_bytes = args.selection.read_bytes()
        config = json.loads(config_bytes.decode("utf-8"))
        selection = json.loads(selection_bytes.decode("utf-8"))
        selected = verify_frozen_lora_host_selection(
            config, config_bytes, selection, args.runs_root
        )
        models = [row for row in config["models"] if row["key"] == selected]
        if len(models) != 1:
            raise ValueError("selected model is missing or duplicated")
        config_sha = hashlib.sha256(config_bytes).hexdigest()
        selection_sha = hashlib.sha256(selection_bytes).hexdigest()
        stored_pilot = json.loads(args.pilot_gate.read_text(encoding="utf-8"))
        if not isinstance(stored_pilot, dict):
            raise ValueError("stored pilot gate is not a JSON object")
        verify_frozen_lora_method_pilot(
            evaluate_lora_method_pilot(config, models[0], args.runs_root),
            stored_pilot,
            config_sha256=config_sha,
            selection_sha256=selection_sha,
        )
        report = evaluate_lora_layer_ablation(
            config, models[0], args.runs_root
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    pilot_gate_sha = hashlib.sha256(args.pilot_gate.read_bytes()).hexdigest()
    if args.verify_existing is not None:
        try:
            stored = json.loads(args.verify_existing.read_text(encoding="utf-8"))
            if not isinstance(stored, dict):
                raise ValueError("stored layer gate is not a JSON object")
            verify_frozen_lora_layer_ablation(
                report,
                stored,
                config_sha256=config_sha,
                selection_sha256=selection_sha,
                pilot_gate_sha256=pilot_gate_sha,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        print(f"verified layer-ablation gate: {selected}")
        return 0
    report.update(
        {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_sha,
            "selection_sha256": selection_sha,
            "pilot_gate_sha256": pilot_gate_sha,
        }
    )
    assert args.output is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
