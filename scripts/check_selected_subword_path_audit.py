#!/usr/bin/env python3
"""Create or verify a selected-host gold-path audit gate."""

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
    SPLIT_PATHS,
    verify_frozen_lora_host_selection,
)
from automata_drs.subword_path_gate import (  # noqa: E402
    evaluate_selected_subword_path_audit,
    verify_frozen_selected_subword_path_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lora_baselines.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/lora/selected-host.json"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs/lora"))
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--audit", type=Path)
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
        if args.verify_existing is None:
            # Gate creation is allowed only where the full selected-host
            # adapter and baseline evidence can be recomputed (the server).
            selected_key = verify_frozen_lora_host_selection(
                config, config_bytes, selection, args.runs_root
            )
        else:
            # Verification can run on a lightweight artifact mirror that does
            # not copy multi-gigabyte adapters. Gate creation and the pilot
            # runner separately revalidate full host evidence on the server,
            # and the frozen gate below is bound to both exact file hashes.
            config_sha = hashlib.sha256(config_bytes).hexdigest()
            if (
                selection.get("schema_version") != 1
                or selection.get("status") != "selected"
                or selection.get("config_sha256") != config_sha
                or not isinstance(selection.get("selected_key"), str)
            ):
                raise ValueError("selected-host pointer is not bound to this config")
            selected_key = selection["selected_key"]
        rows = [row for row in config["models"] if row.get("key") == selected_key]
        if len(rows) != 1:
            raise ValueError("selected host is absent or duplicated in config")
        audit_name = (
            "train-gold-w-wf-path-audit-v2.json"
            if args.split == "train"
            else f"{args.split}-gold-w-wf-path-audit.json"
        )
        audit_path = args.audit or (
            args.runs_root / f"{selected_key}-r16-seed42" / audit_name
        )
        corpus_path = (
            Path("data/en/train/gold.sbn")
            if args.split == "train"
            else SPLIT_PATHS[args.split]
        )
        report = evaluate_selected_subword_path_audit(
            config,
            rows[0],
            split=args.split,
            corpus_path=corpus_path,
            audit_path=audit_path,
            runs_root=args.runs_root,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    selection_sha = hashlib.sha256(selection_bytes).hexdigest()
    if args.verify_existing is not None:
        try:
            stored = json.loads(args.verify_existing.read_text(encoding="utf-8"))
            if not isinstance(stored, dict):
                raise ValueError("stored path-audit gate must be a JSON object")
            verify_frozen_selected_subword_path_audit(
                report,
                stored,
                config_sha256=config_sha,
                selection_sha256=selection_sha,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        print(f"verified selected-host {args.split} gold-path gate: {selected_key}")
        return 0
    report["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["config_sha256"] = config_sha
    report["selection_sha256"] = selection_sha
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            parser.error(f"refusing to overwrite path-audit gate: {args.output}")
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
