#!/usr/bin/env python3
"""Run a resumable same-logit frozen/stable/adaptive ranking audit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import (  # noqa: E402
    ByteLevelVocabulary,
    SBNPolicy,
    check_sbn_state_wellformedness,
    validate_sbn,
)
from automata_drs.adaptive_subword import (  # noqa: E402
    ThreeWaySubwordRankingAuditProcessor,
)
from automata_drs.causal_lm import (  # noqa: E402
    build_sbn_prompt,
    extract_one_line_sbn,
    iter_sbn_records,
    scheduled_generation_identities,
    select_sbn_records_by_line,
    validate_generation_resume_prefix,
)
from automata_drs.generation_manifest import sha256_file  # noqa: E402
from automata_drs.model_loading import (  # noqa: E402
    MODEL_LOADERS,
    load_generation_model,
)


SCHEMA_VERSION = 1
KIND = "adaptive-ranking-real-logit-integration-audit"
CONFIGURATIONS = ("BF", "WF")


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank audit row at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object audit row at {path}:{line_number}")
            rows.append(value)
    return rows


def adapter_hashes(adapter: Path) -> dict[str, str]:
    names = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    missing = [name for name in names if not (adapter / name).is_file()]
    if missing:
        raise ValueError("adapter is missing required files: " + ", ".join(missing))
    return {name: sha256_file(adapter / name) for name in names}


def validate_selection(plan: dict[str, Any], selection_path: Path) -> tuple[Path, list[int]]:
    if plan.get("schema_version") != 1 or plan.get("kind") != (
        "adaptive-ranking-real-logit-audit-selection"
    ):
        raise ValueError("unsupported adaptive-ranking audit selection")
    registered = plan.get("registered_audit")
    expected_registered = {
        "configurations": ["BF", "WF"],
        "decode_cells": [
            {"strategy": "greedy", "num_beams": 1, "retained": 1},
            {"strategy": "beam", "num_beams": 4, "retained": 8},
        ],
        "max_new_tokens": 256,
        "completion_budget_factor": 1.0,
        "adaptive_initial_k": 32,
        "same_logits_three_way_shadow": True,
        "returned_generation_path": "frozen_unstable_full_sort",
        "stable_adaptive_mismatches_required": 0,
        "frozen_stable_mismatches_required_for_drop_in_claim": 0,
        "performance_threshold_used": False,
    }
    if registered != expected_registered:
        raise ValueError("audit selection has a different registered experiment")
    data_value = plan.get("data")
    if not isinstance(data_value, str):
        raise ValueError("audit selection has no data path")
    data_path = Path(data_value)
    if not data_path.is_file() or sha256_file(data_path) != plan.get("data_sha256"):
        raise ValueError("audit selection data hash does not match")
    selected = plan.get("selected")
    if not isinstance(selected, list) or len(selected) != 32:
        raise ValueError("audit selection must contain exactly 32 rows")
    lines = [
        item.get("line_number") if isinstance(item, dict) else None
        for item in selected
    ]
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in lines):
        raise ValueError("audit selection has invalid line numbers")
    line_numbers = [int(value) for value in lines]
    if len(set(line_numbers)) != len(line_numbers):
        raise ValueError("audit selection has duplicate line numbers")
    counts = plan.get("counts")
    expected_counts = {
        "baseline_rows": 1132,
        "u_failure_pressure": 21,
        "length_stratified_valid_controls": 11,
        "selected": 32,
        "generation_cells": 128,
    }
    if counts != expected_counts:
        raise ValueError("audit selection counts differ from registration")
    groups = [item.get("group") for item in selected]
    if (
        groups.count("u_failure_pressure") != 21
        or groups.count("length_stratified_valid_control") != 11
        or len(set(groups)) != 2
    ):
        raise ValueError("audit selection row groups differ from registration")
    baseline_path = Path(str(plan.get("baseline", "")))
    failure_path = Path(str(plan.get("failure_selection", "")))
    if (
        not baseline_path.is_file()
        or sha256_file(baseline_path) != plan.get("baseline_sha256")
        or not failure_path.is_file()
        or sha256_file(failure_path) != plan.get("failure_selection_sha256")
    ):
        raise ValueError("audit selection source artifacts do not match hashes")
    failures = read_object(failure_path).get("selected")
    failure_lines = {
        int(item["line_number"])
        for item in failures
        if isinstance(item, dict) and "line_number" in item
    } if isinstance(failures, list) else set()
    planned_failure_lines = {
        int(item["line_number"])
        for item in selected
        if item.get("group") == "u_failure_pressure"
    }
    if failure_lines != planned_failure_lines:
        raise ValueError("audit pressure rows differ from frozen U failures")
    del selection_path
    return data_path, line_numbers


def complete_manifest(
    manifest_path: Path,
    output: Path,
    manifest: dict[str, Any],
    *,
    expected_rows: int,
) -> None:
    rows = read_rows(output)
    if len(rows) != expected_rows:
        raise ValueError(f"audit output is incomplete: {len(rows)}/{expected_rows}")
    completion = {
        "status": "complete",
        "rows": len(rows),
        "output_sha256": sha256_file(output),
    }
    stored = read_object(manifest_path)
    if stored.get("status") == "complete":
        mismatches = [
            name
            for name, value in completion.items()
            if stored.get(name) != value
        ]
        if mismatches or not isinstance(stored.get("completed_at_utc"), str):
            raise ValueError(
                "completed audit manifest no longer matches evidence: "
                + ", ".join(mismatches or ["completed_at_utc"])
            )
        return
    complete = {
        **manifest,
        **completion,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-loader", choices=MODEL_LOADERS, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--num-beams", type=int, choices=(1, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=4)
    args = parser.parse_args()
    if args.progress_every <= 0:
        parser.error("progress-every must be positive")
    manifest_path = Path(str(args.output) + ".manifest.json")
    if not args.resume and (args.output.exists() or manifest_path.exists()):
        parser.error(f"output or manifest already exists: {args.output}")
    if args.output.exists() and not manifest_path.is_file():
        parser.error("resume output has no manifest")

    lock_path = Path(str(args.output) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_source = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_source.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error(f"another writer holds the audit lock: {lock_path}")

    try:
        plan = read_object(args.selection)
        data_path, line_numbers = validate_selection(plan, args.selection)
        records = select_sbn_records_by_line(
            iter_sbn_records(data_path), line_numbers
        )
        expected = scheduled_generation_identities(
            records, CONFIGURATIONS, schedule="rotate"
        )
        implementation = {
            "audit_runner_sha256": sha256_file(Path(__file__)),
            "adaptive_subword_sha256": sha256_file(
                REPOSITORY_ROOT / "src" / "automata_drs" / "adaptive_subword.py"
            ),
            "ranking_sha256": sha256_file(
                REPOSITORY_ROOT / "src" / "automata_drs" / "ranking.py"
            ),
            "frozen_subword_sha256": sha256_file(
                REPOSITORY_ROOT / "src" / "automata_drs" / "subword.py"
            ),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "status": "running",
            "selection": str(args.selection),
            "selection_sha256": sha256_file(args.selection),
            "data": str(data_path),
            "data_sha256": sha256_file(data_path),
            "model": args.model,
            "revision": args.revision,
            "model_loader": args.model_loader,
            "adapter": str(args.adapter),
            "adapter_hashes": adapter_hashes(args.adapter),
            "num_beams": args.num_beams,
            "configurations": list(CONFIGURATIONS),
            "schedule": "rotate",
            "max_input_tokens": 512,
            "max_new_tokens": 256,
            "completion_budget_factor": 1.0,
            "adaptive_initial_k": 32,
            "expected_examples": len(records),
            "expected_rows": len(expected),
            "same_logits_three_way_shadow": True,
            "returned_generation_path": "frozen_unstable_full_sort",
            "performance_threshold_used": False,
            "physical_gpu_index": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "implementation": implementation,
        }
        contract_fields = tuple(
            name for name in manifest if name != "status"
        )
        stored_status: str | None = None
        if manifest_path.exists():
            stored = read_object(manifest_path)
            stored_status = stored.get("status")
            if stored_status not in {"running", "complete"}:
                raise ValueError("audit manifest has an unsupported status")
            mismatches = [
                name
                for name in contract_fields
                if stored.get(name) != manifest.get(name)
            ]
            if mismatches:
                raise ValueError(
                    "resume manifest differs from request: " + ", ".join(mismatches)
                )
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("x", encoding="utf-8") as destination:
                json.dump(manifest, destination, ensure_ascii=False, indent=2)
                destination.write("\n")

        existing = read_rows(args.output)
        completed = validate_generation_resume_prefix(
            existing, expected, num_beams=args.num_beams
        )
        if stored_status == "complete" and completed != len(expected):
            raise ValueError("completed audit manifest has an incomplete output")
        if completed == len(expected):
            complete_manifest(
                manifest_path, args.output, manifest, expected_rows=len(expected)
            )
            print("adaptive-ranking integration audit is already complete")
            return 0

        try:
            import peft
            import torch
            import transformers
            from peft import PeftModel
            from transformers import AutoTokenizer, LogitsProcessorList
        except ImportError as error:
            parser.error("install the LoRA runtime with: pip install -e '.[lora]'")
            raise AssertionError("unreachable") from error

        load_kwargs: dict[str, Any] = {"revision": args.revision}
        if args.cache_dir:
            load_kwargs["cache_dir"] = str(args.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.adapter)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = load_generation_model(
            transformers,
            args.model,
            loader=args.model_loader,
            load_kwargs={
                **load_kwargs,
                "dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
            },
        )
        model = PeftModel.from_pretrained(base, args.adapter)
        model.to("cuda")
        model.eval()
        cuda = torch.cuda.get_device_properties(0)
        runtime = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "cuda_device_name": cuda.name,
            "cuda_device_total_bytes": int(cuda.total_memory),
        }
        vocabulary = ByteLevelVocabulary.from_tokenizer(tokenizer)
        mode = "a" if args.output.exists() else "x"
        with args.output.open(mode, encoding="utf-8") as destination:
            completed_cells = completed
            for record_index, record in enumerate(records):
                configuration_order = (
                    list(CONFIGURATIONS[record_index % 2 :])
                    + list(CONFIGURATIONS[: record_index % 2])
                )
                encoded = tokenizer(
                    build_sbn_prompt(record.source),
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to("cuda")
                input_tokens = int(encoded["input_ids"].shape[-1])
                prompt_ids = encoded["input_ids"][0].tolist()
                for configuration_index, configuration in enumerate(
                    configuration_order
                ):
                    scheduled_index = record_index * 2 + configuration_index
                    if scheduled_index < completed:
                        continue
                    base_configuration = {"BF": "B", "WF": "W"}[configuration]
                    audit = (
                        ThreeWaySubwordRankingAuditProcessor.from_shared_configuration(
                            tokenizer=tokenizer,
                            vocabulary=vocabulary,
                            prompt_token_ids=prompt_ids,
                            policy=SBNPolicy.for_configuration(base_configuration),
                            retained_tokens_per_row=(
                                1 if args.num_beams == 1 else 2 * args.num_beams
                            ),
                            max_new_tokens=256,
                            bounded_completion=True,
                            completion_budget_factor=1.0,
                            initial_k=32,
                        )
                    )
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        output = model.generate(
                            **encoded,
                            logits_processor=LogitsProcessorList([audit]),
                            do_sample=False,
                            num_beams=args.num_beams,
                            max_new_tokens=256,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    suffix = output[0, input_tokens:]
                    suffix_ids = [int(value) for value in suffix.tolist()]
                    raw_prediction = tokenizer.decode(
                        suffix,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    prediction = extract_one_line_sbn(raw_prediction)
                    structural = validate_sbn(prediction)
                    semantic = (
                        check_sbn_state_wellformedness(structural.state)
                        if structural.valid
                        else None
                    )
                    audit_stats = audit.stats()
                    frozen_stats = audit.frozen.stats()
                    stable_stats = audit.stable.stats()
                    adaptive_stats = audit.adaptive.stats()
                    ranking_stats = audit.adaptive.ranking_stats()
                    row = {
                        "id": record.identifier,
                        "input": record.source,
                        "reference": record.target,
                        "configuration": configuration,
                        "decode": {
                            "strategy": (
                                "greedy" if args.num_beams == 1 else "beam"
                            ),
                            "num_beams": args.num_beams,
                        },
                        "prediction": prediction,
                        "raw_prediction": raw_prediction,
                        "surface_token_ids": suffix_ids,
                        "validity": {
                            "sbn": structural.valid,
                            "sbn_reason": structural.reason,
                            "clf_referee": None if semantic is None else semantic.valid,
                            "clf_referee_reason": (
                                None if semantic is None else semantic.reason
                            ),
                        },
                        "timing": {
                            "three_way_generation_ms": elapsed * 1000,
                            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
                        },
                        "equivalence": asdict(audit_stats),
                        "processors": {
                            "frozen": asdict(frozen_stats),
                            "stable": asdict(stable_stats),
                            "adaptive": asdict(adaptive_stats),
                            "adaptive_ranking": asdict(ranking_stats),
                        },
                        "runtime": runtime,
                    }
                    destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                    destination.flush()
                    completed_cells += 1
                    if completed_cells == 1 or completed_cells % args.progress_every == 0:
                        print(
                            f"{completed_cells}/{len(expected)} {record.identifier} "
                            f"{configuration}: rows={audit_stats.rows} "
                            f"frozen_ties={audit_stats.frozen_stable_mismatches}",
                            flush=True,
                        )
        complete_manifest(
            manifest_path, args.output, manifest, expected_rows=len(expected)
        )
        print(f"adaptive-ranking integration audit complete: {args.output}")
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
        raise AssertionError("unreachable") from error


if __name__ == "__main__":
    raise SystemExit(main())
