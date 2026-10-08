#!/usr/bin/env python3
"""Freeze within-matrix U control before any publication method output."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.host_selection import (  # noqa: E402
    verify_frozen_lora_host_selection,
)
from automata_drs.lora_method_gate import (  # noqa: E402
    evaluate_lora_method_pilot,
    verify_frozen_lora_method_pilot,
)
from automata_drs.publication_control import (  # noqa: E402
    KIND,
    SCHEMA_VERSION,
    planned_outputs,
    sha256_file,
    verify_publication_control_plan,
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
        parser.error(f"refusing to overwrite publication U-control plan: {args.output}")
    try:
        config_bytes = args.config.read_bytes()
        selection_bytes = args.selection.read_bytes()
        config = json.loads(config_bytes)
        selection = json.loads(selection_bytes)
        selected = verify_frozen_lora_host_selection(
            config, config_bytes, selection, args.runs_root
        )
        models = [row for row in config["models"] if row["key"] == selected]
        if len(models) != 1:
            raise ValueError("selected model is absent or duplicated")
        pilot_gate = json.loads(args.pilot_gate.read_text(encoding="utf-8"))
        if not isinstance(pilot_gate, dict):
            raise ValueError("pilot gate must be a JSON object")
        verify_frozen_lora_method_pilot(
            evaluate_lora_method_pilot(config, models[0], args.runs_root),
            pilot_gate,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            selection_sha256=hashlib.sha256(selection_bytes).hexdigest(),
        )
        outputs = planned_outputs(config, selected, args.runs_root)
        if args.verify_existing is not None:
            verify_publication_control_plan(
                args.verify_existing,
                config=config,
                config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                selection_sha256=hashlib.sha256(selection_bytes).hexdigest(),
                model_key=selected,
                runs_root=args.runs_root,
                pilot_gate_path=args.pilot_gate,
            )
        else:
            conflicting = []
            for output_value in outputs:
                output = Path(output_value)
                prefix = output.with_suffix("")
                candidates = (
                    output,
                    Path(str(output) + ".manifest.json"),
                    Path(str(prefix) + ".smatch.jsonl"),
                    Path(str(prefix) + ".paired-statistics.json"),
                )
                conflicting.extend(
                    str(path) for path in candidates if path.exists()
                )
            if conflicting:
                raise ValueError(
                    "publication method output exists before plan freeze: "
                    + ", ".join(conflicting)
                )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    if args.verify_existing is not None:
        print(f"verified publication U-control plan: {args.verify_existing}")
        return 0

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_key": selected,
        "reason": (
            "The legacy batch-8 host screen is quality-selection evidence; "
            "the registered batch-1 U cell in each method matrix is its paired control."
        ),
        "control_contract": {
            "host_selection_u_role": "quality-only legacy batch-8 screen",
            "method_u_role": "within-matrix batch-1 greedy control",
            "method_u_logits_processor": None,
            "cross_contract_surface_identity_required": False,
            "pilot_batch1_exact_replay_required": True,
            "paired_population_required": True,
            "task_metric_threshold_used": False,
        },
        "planned_outputs": outputs,
        "outputs_absent_when_frozen": True,
        "evidence": {
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
            "pilot_gate_path": str(args.pilot_gate),
            "pilot_gate_sha256": sha256_file(args.pilot_gate),
        },
    }
    assert args.output is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"frozen publication U-control plan: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
