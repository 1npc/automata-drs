#!/usr/bin/env python3
"""Check, freeze, or re-verify the selected-host beam-selection study."""

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
from automata_drs.lora_beam_gate import (  # noqa: E402
    evaluate_lora_beam_selection,
    verify_frozen_lora_beam_selection,
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
    parser.add_argument(
        "--config", type=Path, default=Path("configs/lora_baselines.json")
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs/lora"))
    parser.add_argument(
        "--selection", type=Path, default=Path("runs/lora/selected-host.json")
    )
    parser.add_argument("--pilot-gate", type=Path)
    parser.add_argument("--publication-gate", type=Path)
    parser.add_argument("--u-control-plan", type=Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite beam-selection gate: {args.output}")

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
        model_row = models[0]
        records = int(config["method_experiment"]["pilot"]["records"])
        run_root = args.runs_root / f"{selected}-r16-seed42"
        pilot_gate_path = args.pilot_gate or (
            run_root / f"dev{records}-method-pilot-gate.json"
        )
        publication_gate_path = args.publication_gate or (
            run_root / "dev1132-publication-matrix-gate.json"
        )
        u_control_plan_path = args.u_control_plan or (
            run_root / "publication-u-control-plan.json"
        )
        stored_pilot = json.loads(pilot_gate_path.read_text(encoding="utf-8"))
        stored_publication = json.loads(
            publication_gate_path.read_text(encoding="utf-8")
        )
        if not isinstance(stored_pilot, dict) or not isinstance(
            stored_publication, dict
        ):
            raise ValueError("pilot/publication gate must be JSON objects")
        config_sha = hashlib.sha256(config_bytes).hexdigest()
        selection_sha = hashlib.sha256(selection_bytes).hexdigest()
        pilot = evaluate_lora_method_pilot(config, model_row, args.runs_root)
        verify_frozen_lora_method_pilot(
            pilot,
            stored_pilot,
            config_sha256=config_sha,
            selection_sha256=selection_sha,
        )
        verify_publication_control_plan(
            u_control_plan_path,
            config=config,
            config_sha256=config_sha,
            selection_sha256=selection_sha,
            model_key=selected,
            runs_root=args.runs_root,
            pilot_gate_path=pilot_gate_path,
        )
        publication = evaluate_lora_publication_matrix(
            config, model_row, args.runs_root, "dev"
        )
        verify_frozen_lora_publication_matrix(
            publication,
            stored_publication,
            split="dev",
            config_sha256=config_sha,
            selection_sha256=selection_sha,
            pilot_gate_sha256=hashlib.sha256(
                pilot_gate_path.read_bytes()
            ).hexdigest(),
            u_control_plan_sha256=sha256_file(u_control_plan_path),
        )
        report = evaluate_lora_beam_selection(
            config,
            model_row,
            args.runs_root,
            config_sha256=config_sha,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    publication_sha = hashlib.sha256(publication_gate_path.read_bytes()).hexdigest()
    if args.verify_existing is not None:
        try:
            stored = json.loads(args.verify_existing.read_text(encoding="utf-8"))
            if not isinstance(stored, dict):
                raise ValueError("stored beam-selection gate is not an object")
            verify_frozen_lora_beam_selection(
                report,
                stored,
                config_sha256=config_sha,
                selection_sha256=selection_sha,
                publication_gate_sha256=publication_sha,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        print(f"verified beam-selection gate: {selected}")
        return 0

    report.update(
        {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_sha,
            "selection_sha256": selection_sha,
            "publication_gate_sha256": publication_sha,
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
