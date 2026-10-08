#!/usr/bin/env python3
"""Check and freeze one selected-host conditioned intervention split."""

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
from automata_drs.intervention_gate import (  # noqa: E402
    evaluate_lora_intervention,
    verify_frozen_lora_intervention,
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
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    parser.add_argument("--model-key")
    parser.add_argument(
        "--artifact-suffix",
        choices=["", "-v2"],
        default="",
        help="versioned corrected intervention artifact suffix",
    )
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite intervention gate: {args.output}")
    try:
        config_bytes = args.config.read_bytes()
        selection_bytes = args.selection.read_bytes()
        config = json.loads(config_bytes.decode("utf-8"))
        selection = json.loads(selection_bytes.decode("utf-8"))
        selected = verify_frozen_lora_host_selection(
            config, config_bytes, selection, args.runs_root
        )
        if args.model_key is not None and args.model_key != selected:
            raise ValueError(
                f"requested intervention host {args.model_key} is not frozen "
                f"selected host {selected}"
            )
        models = [row for row in config["models"] if row["key"] == selected]
        if len(models) != 1:
            raise ValueError("selected model is missing or duplicated")
        report = evaluate_lora_intervention(
            config,
            models[0],
            args.runs_root,
            args.split,
            artifact_suffix=args.artifact_suffix,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    selection_sha = hashlib.sha256(selection_bytes).hexdigest()
    if args.verify_existing is not None:
        try:
            stored = json.loads(args.verify_existing.read_text(encoding="utf-8"))
            if not isinstance(stored, dict):
                raise ValueError("stored intervention gate is not a JSON object")
            verify_frozen_lora_intervention(
                report,
                stored,
                config_sha256=config_sha,
                selection_sha256=selection_sha,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        print(f"verified intervention gate: {selected}/{args.split}")
        return 0
    report.update(
        {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_sha,
            "selection_sha256": selection_sha,
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
