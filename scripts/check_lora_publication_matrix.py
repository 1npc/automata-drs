#!/usr/bin/env python3
"""Check and freeze a selected-host publication-scale method matrix."""

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
    evaluate_lora_publication_matrix,
    verify_frozen_lora_method_pilot,
    verify_frozen_lora_publication_matrix,
)
from automata_drs.publication_control import (  # noqa: E402
    sha256_file,
    verify_publication_control_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["dev", "test", "tax"])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/lora_baselines.json")
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs/lora"))
    parser.add_argument(
        "--selection", type=Path, default=Path("runs/lora/selected-host.json")
    )
    parser.add_argument("--pilot-gate", type=Path, required=True)
    parser.add_argument("--u-control-plan", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite publication gate: {args.output}")
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
        model_row = model_rows[0]
        pilot = evaluate_lora_method_pilot(config, model_row, args.runs_root)
        stored_pilot = json.loads(args.pilot_gate.read_text(encoding="utf-8"))
        if not isinstance(stored_pilot, dict):
            raise ValueError("stored method pilot gate must be a JSON object")
        config_sha = hashlib.sha256(config_bytes).hexdigest()
        selection_sha = hashlib.sha256(selection_bytes).hexdigest()
        verify_frozen_lora_method_pilot(
            pilot,
            stored_pilot,
            config_sha256=config_sha,
            selection_sha256=selection_sha,
        )
        verify_publication_control_plan(
            args.u_control_plan,
            config=config,
            config_sha256=config_sha,
            selection_sha256=selection_sha,
            model_key=selected_key,
            runs_root=args.runs_root,
            pilot_gate_path=args.pilot_gate,
        )
        report = evaluate_lora_publication_matrix(
            config, model_row, args.runs_root, args.split
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    pilot_gate_sha = hashlib.sha256(args.pilot_gate.read_bytes()).hexdigest()
    if args.verify_existing is not None:
        try:
            stored = json.loads(args.verify_existing.read_text(encoding="utf-8"))
            if not isinstance(stored, dict):
                raise ValueError("stored publication gate is not a JSON object")
            verify_frozen_lora_publication_matrix(
                report,
                stored,
                split=args.split,
                config_sha256=config_sha,
                selection_sha256=selection_sha,
                pilot_gate_sha256=pilot_gate_sha,
                u_control_plan_sha256=sha256_file(args.u_control_plan),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        print(f"verified publication gate: {selected_key}/{args.split}")
        return 0
    report["split"] = args.split
    report["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["config_sha256"] = config_sha
    report["selection_sha256"] = selection_sha
    report["pilot_gate_sha256"] = pilot_gate_sha
    report["u_control_plan_sha256"] = sha256_file(args.u_control_plan)
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
