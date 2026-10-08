#!/usr/bin/env python3
"""Generate layered SBN predictions from a causal-LM LoRA adapter."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import os
import platform
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import (  # noqa: E402
    ByteLevelVocabulary,
    SBNPolicy,
    SubwordSBNLogitsProcessor,
    WitnessCarryingSubwordSBNLogitsProcessor,
    check_sbn_state_wellformedness,
    validate_sbn,
)
from automata_drs.causal_lm import (  # noqa: E402
    build_sbn_prompt,
    extract_one_line_sbn,
    iter_sbn_records,
    scheduled_generation_identities,
    select_sbn_records_by_line,
    validate_generation_resume_prefix,
)
from automata_drs.generation_manifest import (  # noqa: E402
    KIND as GENERATION_MANIFEST_KIND,
    SCHEMA_VERSION as GENERATION_MANIFEST_SCHEMA_VERSION,
    complete_generation_manifest,
    sha256_file,
)
from automata_drs.model_loading import (  # noqa: E402
    MODEL_LOADERS,
    load_generation_model,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--model-loader", choices=MODEL_LOADERS, default="causal-lm"
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument(
        "--line-numbers",
        type=int,
        nargs="+",
        help="select exact one-based physical TSV line numbers in the given order",
    )
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="batching is restricted to unscored unconstrained greedy generation",
    )
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=[
            "U", "S", "R", "G", "B", "W", "BF", "WF", "BWCD", "WWCD"
        ],
        default=["U"],
    )
    parser.add_argument("--completion-budget-factor", type=float, default=1.0)
    parser.add_argument("--record-native-token-ids", action="store_true")
    parser.add_argument("--review-instrumentation", action="store_true")
    parser.add_argument("--review-fail-fast", action="store_true")
    parser.add_argument("--review-reuse", type=Path,
                        help="reuse locked predecessor rows with no all-masked event")
    parser.add_argument("--fresh-proposal-denial-rate", type=float, default=0.0)
    parser.add_argument(
        "--score-model-logprob",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "teacher-force each generated suffix to record its log-probability "
            "under the unconstrained LoRA model"
        ),
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append only when the existing JSONL is an exact scheduled prefix",
    )
    parser.add_argument(
        "--schedule",
        choices=["rotate", "fixed"],
        default="rotate",
        help="rotate configuration order by example to reduce timing-order bias",
    )
    args = parser.parse_args()

    manifest_path = Path(str(args.output) + ".manifest.json")
    if not args.resume and (args.output.exists() or manifest_path.exists()):
        parser.error(f"output or manifest already exists: {args.output}")
    if args.output.exists() and not manifest_path.is_file():
        parser.error(f"resume output has no generation manifest: {args.output}")
    if args.num_beams <= 0:
        parser.error("num_beams must be positive")
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.line_numbers is not None:
        if args.start != 0 or args.limit != -1:
            parser.error("line-numbers cannot be combined with start or limit")
        if any(line_number <= 0 for line_number in args.line_numbers):
            parser.error("line-numbers must be positive")
        if len(set(args.line_numbers)) != len(args.line_numbers):
            parser.error("line-numbers must not contain duplicates")
    if args.progress_every <= 0:
        parser.error("progress-every must be positive")
    if not 0 < args.completion_budget_factor <= 1:
        parser.error("completion-budget-factor must be in (0, 1]")
    if not 0 <= args.fresh_proposal_denial_rate <= 1:
        parser.error("fresh-proposal-denial-rate must be in [0, 1]")
    if args.fresh_proposal_denial_rate and not args.review_instrumentation:
        parser.error("proposal denial requires review instrumentation")
    if args.review_instrumentation and args.num_beams != 1:
        parser.error("review instrumentation is registered for greedy only")
    if (args.review_fail_fast or args.review_reuse) and (
        not args.review_instrumentation or args.num_beams != 1 or args.batch_size != 1
    ):
        parser.error("review recovery requires instrumented greedy batch-one")
    if args.review_reuse and not args.review_fail_fast:
        parser.error("review reuse requires the explicit fail-fast policy")
    if args.batch_size > 1 and (
        args.configurations != ["U"]
        or args.num_beams != 1
        or args.score_model_logprob
    ):
        parser.error(
            "batch-size > 1 requires U-only greedy generation and "
            "--no-score-model-logprob"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(args.output) + ".lock")
    lock_source = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_source.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error(f"another writer holds the generation lock: {lock_path}")

    try:
        import peft
        import torch
        import transformers
        from peft import PeftModel
        from transformers import AutoTokenizer, LogitsProcessorList
    except ImportError as error:
        parser.error("install the LoRA runtime with: pip install -e '.[lora]'")
        raise AssertionError("unreachable") from error

    load_kwargs = {"revision": args.revision}
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
    cuda_properties = torch.cuda.get_device_properties(0)

    if args.line_numbers is None:
        records = list(
            iter_sbn_records(args.path, start=args.start, limit=args.limit)
        )
    else:
        try:
            records = select_sbn_records_by_line(
                iter_sbn_records(args.path), args.line_numbers
            )
        except ValueError as error:
            parser.error(str(error))
    if not records:
        parser.error(f"no records found in {args.path}")
    subword_vocabulary = (
        None
        if args.configurations == ["U"]
        else ByteLevelVocabulary.from_tokenizer(tokenizer)
    )

    expected_identities = scheduled_generation_identities(
        records, args.configurations, schedule=args.schedule
    )
    manifest = {
        "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
        "kind": GENERATION_MANIFEST_KIND,
        "status": "running",
        "model": args.model,
        "model_loader": args.model_loader,
        "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "adapter": str(args.adapter),
        "path": str(args.path),
        "start": args.start,
        "limit": args.limit,
        "line_numbers": args.line_numbers,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "batch_size": args.batch_size,
        "configurations": args.configurations,
        "completion_budget_factor": args.completion_budget_factor,
        "record_native_token_ids": args.record_native_token_ids,
        "review_instrumentation": args.review_instrumentation,
        "review_fail_fast": args.review_fail_fast,
        "review_reuse": None if args.review_reuse is None else str(args.review_reuse),
        "review_reuse_sha256": None if args.review_reuse is None else sha256_file(args.review_reuse),
        "fresh_proposal_denial_rate": args.fresh_proposal_denial_rate,
        "score_model_logprob": args.score_model_logprob,
        "constraint_lifting": "exact-bytelevel-bpe-score-ranked",
        "schedule": args.schedule,
        "source_sha256": sha256_file(args.path),
        "expected_examples": len(records),
        "expected_rows": len(expected_identities),
        "physical_gpu_index": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": cuda_properties.name,
        "cuda_device_total_bytes": int(cuda_properties.total_memory),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }
    contract_fields = (
        "model",
        "model_loader",
        "requested_revision",
        "resolved_revision",
        "adapter",
        "path",
        "start",
        "limit",
        "line_numbers",
        "max_input_tokens",
        "max_new_tokens",
        "num_beams",
        "batch_size",
        "configurations",
        "completion_budget_factor",
        "record_native_token_ids",
        "review_instrumentation",
        "review_fail_fast",
        "review_reuse",
        "review_reuse_sha256",
        "fresh_proposal_denial_rate",
        "score_model_logprob",
        "constraint_lifting",
        "schedule",
        "physical_gpu_index",
        "cuda_device_name",
        "cuda_device_total_bytes",
    )
    versioned_contract_fields = (
        "schema_version",
        "kind",
        "source_sha256",
        "expected_examples",
        "expected_rows",
    )
    if manifest_path.exists():
        try:
            stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"cannot read resume manifest: {error}")
        mismatches = [
            field
            for field in contract_fields
            if stored_manifest.get(
                field,
                manifest.get(field) if field in {
                    "record_native_token_ids", "review_instrumentation", "fresh_proposal_denial_rate",
                    "review_fail_fast", "review_reuse", "review_reuse_sha256"
                } and not manifest.get(field) else None,
            ) != manifest.get(field)
        ]
        if stored_manifest.get("schema_version") is not None:
            mismatches.extend(
                field
                for field in versioned_contract_fields
                if stored_manifest.get(field) != manifest.get(field)
            )
        if mismatches:
            parser.error(
                "resume manifest differs from requested run: "
                + ", ".join(mismatches)
            )
    else:
        with manifest_path.open("x", encoding="utf-8") as sink:
            sink.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    existing_rows = []
    if args.output.exists():
        try:
            with args.output.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        parser.error(
                            f"blank resume row at {args.output}:{line_number}"
                        )
                    existing_rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            parser.error(f"invalid resume JSONL: {error}")
    try:
        completed_prefix = validate_generation_resume_prefix(
            existing_rows,
            expected_identities,
            num_beams=args.num_beams,
        )
    except ValueError as error:
        parser.error(str(error))
    if completed_prefix:
        print(
            f"validated resume prefix: {completed_prefix}/{len(expected_identities)}",
            flush=True,
        )
    if completed_prefix == len(expected_identities):
        complete_generation_manifest(args.output, manifest_path)
        print("generation output is already complete", flush=True)
        return 0
    output_mode = "a" if args.output.exists() else "x"

    reusable_rows = {}
    if args.review_reuse:
        from arr1012_review_recovery import load_reusable_rows
        reusable_rows = load_reusable_rows(args.review_reuse, manifest, expected_identities)
        print(f"verified predecessor rows eligible for reuse: {len(reusable_rows)}", flush=True)

    if args.batch_size > 1:
        tokenizer.padding_side = "left"
        total_runs = len(records)
        completed_runs = completed_prefix
        with args.output.open(output_mode, encoding="utf-8") as sink:
            for batch_start in range(0, len(records), args.batch_size):
                batch_records = records[batch_start : batch_start + args.batch_size]
                if completed_prefix >= batch_start + len(batch_records):
                    continue
                encoded = tokenizer(
                    [build_sbn_prompt(record.source) for record in batch_records],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_input_tokens,
                ).to("cuda")
                prompt_width = int(encoded["input_ids"].shape[-1])
                input_token_counts = [
                    int(count)
                    for count in encoded["attention_mask"].sum(dim=-1).tolist()
                ]
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    outputs = model.generate(
                        **encoded,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                torch.cuda.synchronize()
                batch_elapsed = time.perf_counter() - started
                peak_gpu_bytes = torch.cuda.max_memory_allocated()
                per_record_elapsed = batch_elapsed / len(batch_records)

                for batch_index, record in enumerate(batch_records):
                    if batch_start + batch_index < completed_prefix:
                        continue
                    suffix_ids = [
                        int(token_id)
                        for token_id in outputs[batch_index, prompt_width:].tolist()
                    ]
                    if tokenizer.eos_token_id in suffix_ids:
                        eos_index = suffix_ids.index(tokenizer.eos_token_id)
                        suffix_ids = suffix_ids[: eos_index + 1]
                    raw_prediction = tokenizer.decode(
                        suffix_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    prediction = extract_one_line_sbn(raw_prediction)
                    raw_structural = validate_sbn(raw_prediction)
                    raw_semantic_valid = None
                    raw_semantic_reason = None
                    if raw_structural.valid:
                        raw_semantic = check_sbn_state_wellformedness(
                            raw_structural.state
                        )
                        raw_semantic_valid = raw_semantic.valid
                        raw_semantic_reason = raw_semantic.reason
                    structural = (
                        raw_structural
                        if prediction == raw_prediction
                        else validate_sbn(prediction)
                    )
                    semantic_valid = None
                    semantic_reason = None
                    if prediction == raw_prediction:
                        semantic_valid = raw_semantic_valid
                        semantic_reason = raw_semantic_reason
                    elif structural.valid:
                        semantic = check_sbn_state_wellformedness(
                            structural.state
                        )
                        semantic_valid = semantic.valid
                        semantic_reason = semantic.reason
                    ended_with_eos = tokenizer.eos_token_id in suffix_ids
                    surface_token_ids = []
                    for token_id in suffix_ids:
                        if token_id == tokenizer.eos_token_id:
                            break
                        surface_token_ids.append(token_id)
                    canonical_token_ids = [
                        int(token_id)
                        for token_id in tokenizer(
                            raw_prediction, add_special_tokens=False
                        )["input_ids"]
                    ]
                    row = {
                        "id": record.identifier,
                        "input": record.source,
                        "reference": record.target,
                        "prediction": prediction,
                        "raw_prediction": raw_prediction,
                        "configuration": "U",
                        "decode": {"strategy": "greedy", "num_beams": 1},
                        "validity": {
                            "raw_sbn": raw_structural.valid,
                            "raw_sbn_reason": raw_structural.reason,
                            "raw_clf_referee": raw_semantic_valid,
                            "raw_clf_referee_reason": raw_semantic_reason,
                            "sbn": structural.valid,
                            "sbn_reason": structural.reason,
                            "clf_referee": semantic_valid,
                            "clf_referee_reason": semantic_reason,
                        },
                        "timing": {
                            "generation_ms": per_record_elapsed * 1000,
                            "model_score_ms": 0.0,
                            "constraint_ms": 0.0,
                        },
                        "score": {
                            "raw_model_logprob": None,
                            "raw_model_mean_logprob": None,
                            "scored_tokens": 0,
                        },
                        "generation": {
                            "input_tokens": input_token_counts[batch_index],
                            "output_tokens_including_eos": len(suffix_ids),
                            "decoder_steps": len(suffix_ids),
                            "ended_with_eos": ended_with_eos,
                            "canonical_tokenization": (
                                surface_token_ids == canonical_token_ids
                            ),
                            "hit_max_new_tokens": (
                                len(suffix_ids) >= args.max_new_tokens
                                and not ended_with_eos
                            ),
                            "peak_gpu_bytes": peak_gpu_bytes,
                        },
                        "constraint": None,
                    }
                    if args.record_native_token_ids:
                        row["generation"]["native_token_ids"] = suffix_ids
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                    completed_runs += 1
                    if (
                        completed_runs == 1
                        or completed_runs % args.progress_every == 0
                    ):
                        print(
                            f"{completed_runs}/{total_runs} {record.identifier} "
                            f"U: sbn={structural.valid} tokens={len(suffix_ids)} "
                            f"amortized_time={per_record_elapsed:.2f}s",
                            flush=True,
                        )
                del outputs, encoded
        complete_generation_manifest(args.output, manifest_path)
        print(f"generation complete: {total_runs} rows", flush=True)
        return 0

    with args.output.open(output_mode, encoding="utf-8") as sink:
        completed_runs = completed_prefix
        total_runs = len(records) * len(args.configurations)
        configuration_names = list(args.configurations)
        for record_index, record in enumerate(records):
            record_start = record_index * len(configuration_names)
            if completed_prefix >= record_start + len(configuration_names):
                continue
            encoded = tokenizer(
                build_sbn_prompt(record.source),
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
            ).to("cuda")
            input_tokens = int(encoded["input_ids"].shape[-1])
            prompt_token_ids = encoded["input_ids"][0].tolist()
            if args.schedule == "rotate":
                shift = record_index % len(configuration_names)
                record_configurations = (
                    configuration_names[shift:] + configuration_names[:shift]
                )
            else:
                record_configurations = configuration_names
            for configuration_index, configuration in enumerate(record_configurations):
                scheduled_index = record_start + configuration_index
                if scheduled_index < completed_prefix:
                    continue
                reused = reusable_rows.get((record.identifier, configuration))
                if reused is not None:
                    if reused["input"] != record.source or reused["reference"] != record.target:
                        raise RuntimeError("predecessor content differs from frozen input")
                    sink.write(json.dumps(reused, ensure_ascii=False) + "\n")
                    sink.flush()
                    completed_runs += 1
                    continue
                processor = None
                if configuration != "U":
                    base_configuration = {
                        "BF": "B",
                        "WF": "W",
                        "BWCD": "B",
                        "WWCD": "W",
                    }.get(configuration, configuration)
                    processor_arguments = {
                        "tokenizer": tokenizer,
                        "vocabulary": subword_vocabulary,
                        "prompt_token_ids": prompt_token_ids,
                        "policy": SBNPolicy.for_configuration(base_configuration),
                        "retained_tokens_per_row": (
                            1 if args.num_beams == 1 else 2 * args.num_beams
                        ),
                        "max_new_tokens": args.max_new_tokens,
                    }
                    local_processor_class = SubwordSBNLogitsProcessor
                    carrying_processor_class = WitnessCarryingSubwordSBNLogitsProcessor
                    if args.review_instrumentation:
                        from arr1012_review_runtime import InstrumentedWF, InstrumentedAPD
                        local_processor_class = InstrumentedWF
                        carrying_processor_class = InstrumentedAPD
                        processor_arguments.update(
                            review_identifier=record.identifier,
                            review_denial_rate=args.fresh_proposal_denial_rate,
                            review_fail_fast=args.review_fail_fast,
                        )
                    if configuration in {"BWCD", "WWCD"}:
                        processor = carrying_processor_class(
                            **processor_arguments,
                            max_prefix_certificates=20_000,
                        )
                    else:
                        processor = local_processor_class(
                            **processor_arguments,
                            bounded_completion=configuration in {"BF", "WF"},
                            completion_budget_factor=(
                                args.completion_budget_factor
                            ),
                        )
                logits_processors = LogitsProcessorList(
                    [] if processor is None else [processor]
                )
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                decoder_failure = None
                with torch.inference_mode():
                    generation_kwargs = dict(
                        logits_processor=logits_processors, do_sample=False,
                        num_beams=args.num_beams, max_new_tokens=args.max_new_tokens,
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
                    if args.review_fail_fast:
                        from arr1012_review_runtime import generate_with_failure_capture
                        suffix, decoder_failure = generate_with_failure_capture(
                            model, encoded, input_tokens, **generation_kwargs)
                    else:
                        output = model.generate(**encoded, **generation_kwargs)
                        suffix = output[0, input_tokens:]
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                raw_model_logprob = None
                raw_model_mean_logprob = None
                score_elapsed = 0.0
                score_token_count = int(suffix.shape[-1])
                if args.score_model_logprob and score_token_count:
                    score_input_ids = torch.cat(
                        (encoded["input_ids"], suffix.unsqueeze(0)), dim=-1
                    )
                    score_attention_mask = torch.cat(
                        (
                            encoded["attention_mask"],
                            torch.ones_like(suffix).unsqueeze(0),
                        ),
                        dim=-1,
                    )
                    labels = score_input_ids.clone()
                    labels[:, :input_tokens] = -100
                    torch.cuda.synchronize()
                    score_started = time.perf_counter()
                    with torch.inference_mode():
                        scored = model(
                            input_ids=score_input_ids,
                            attention_mask=score_attention_mask,
                            labels=labels,
                        )
                    torch.cuda.synchronize()
                    score_elapsed = time.perf_counter() - score_started
                    raw_model_mean_logprob = -float(scored.loss.item())
                    raw_model_logprob = (
                        raw_model_mean_logprob * score_token_count
                    )
                raw_prediction = tokenizer.decode(
                    suffix,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                prediction = extract_one_line_sbn(raw_prediction)
                raw_structural = validate_sbn(raw_prediction)
                raw_semantic_valid = None
                raw_semantic_reason = None
                if raw_structural.valid:
                    raw_semantic = check_sbn_state_wellformedness(
                        raw_structural.state
                    )
                    raw_semantic_valid = raw_semantic.valid
                    raw_semantic_reason = raw_semantic.reason
                structural = (
                    raw_structural
                    if prediction == raw_prediction
                    else validate_sbn(prediction)
                )
                semantic_valid = None
                semantic_reason = None
                if prediction == raw_prediction:
                    semantic_valid = raw_semantic_valid
                    semantic_reason = raw_semantic_reason
                elif structural.valid:
                    semantic = check_sbn_state_wellformedness(structural.state)
                    semantic_valid = semantic.valid
                    semantic_reason = semantic.reason
                ended_with_eos = tokenizer.eos_token_id in suffix.tolist()
                surface_token_ids = []
                for token_id in suffix.tolist():
                    if token_id == tokenizer.eos_token_id:
                        break
                    surface_token_ids.append(int(token_id))
                canonical_token_ids = [
                    int(token_id)
                    for token_id in tokenizer(
                        raw_prediction, add_special_tokens=False
                    )["input_ids"]
                ]
                processor_stats = (
                    None if processor is None else processor.stats()
                )
                cache = (
                    None if processor is None else processor.controller.cache_info()
                )
                completion_cache = (
                    None if processor is None else processor.completion_cache_info()
                )
                constraint = None
                if (
                    processor_stats is not None
                    and cache is not None
                    and completion_cache is not None
                ):
                    constraint = {
                        **asdict(processor_stats),
                        "allowed_mean": (
                            processor_stats.allowed_total / processor_stats.rows
                            if processor_stats.rows
                            else None
                        ),
                        "candidates_tested_per_row": (
                            processor_stats.candidates_tested / processor_stats.rows
                            if processor_stats.rows
                            else None
                        ),
                        "cache_hits": cache.hits,
                        "cache_misses": cache.misses,
                        "cache_entries": cache.size,
                        "cache_evictions": cache.evictions,
                        "completion_cache_hits": completion_cache.hits,
                        "completion_cache_misses": completion_cache.misses,
                        "completion_cache_entries": completion_cache.size,
                        "completion_cache_evictions": completion_cache.evictions,
                    }
                row = {
                    "id": record.identifier,
                    "input": record.source,
                    "reference": record.target,
                    "prediction": prediction,
                    "raw_prediction": raw_prediction,
                    "configuration": configuration,
                    "decode": {
                        "strategy": "greedy" if args.num_beams == 1 else "beam",
                        "num_beams": args.num_beams,
                    },
                    "validity": {
                        "raw_sbn": raw_structural.valid,
                        "raw_sbn_reason": raw_structural.reason,
                        "raw_clf_referee": raw_semantic_valid,
                        "raw_clf_referee_reason": raw_semantic_reason,
                        "sbn": structural.valid,
                        "sbn_reason": structural.reason,
                        "clf_referee": semantic_valid,
                        "clf_referee_reason": semantic_reason,
                    },
                    "timing": {
                        "generation_ms": elapsed * 1000,
                        "model_score_ms": score_elapsed * 1000,
                        "constraint_ms": (
                            0.0
                            if processor_stats is None
                            else processor_stats.seconds * 1000
                        ),
                    },
                    "score": {
                        "raw_model_logprob": raw_model_logprob,
                        "raw_model_mean_logprob": raw_model_mean_logprob,
                        "scored_tokens": score_token_count,
                    },
                    "generation": {
                        "input_tokens": input_tokens,
                        "output_tokens_including_eos": int(suffix.shape[-1]),
                        "decoder_steps": int(suffix.shape[-1]),
                        "ended_with_eos": ended_with_eos,
                        "canonical_tokenization": (
                            surface_token_ids == canonical_token_ids
                        ),
                        "hit_max_new_tokens": (
                            int(suffix.shape[-1]) >= args.max_new_tokens
                            and not ended_with_eos
                        ),
                        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                    },
                    "constraint": constraint,
                }
                if args.record_native_token_ids:
                    row["generation"]["native_token_ids"] = [int(i) for i in suffix.tolist()]
                if args.review_fail_fast:
                    row["generation"]["decoder_failure"] = decoder_failure
                    row["generation"]["decoder_steps"] = len(suffix) + int(decoder_failure is not None)
                    row["generation"]["all_masked_policy"] = "terminate-at-first-all-masked-v1"
                if args.review_instrumentation and processor is not None:
                    row["review_operations"] = processor.review_report()
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                completed_runs += 1
                if (
                    completed_runs == 1
                    or completed_runs % args.progress_every == 0
                ):
                    print(
                        f"{completed_runs}/{total_runs} {record.identifier} "
                        f"{configuration}: sbn={structural.valid} "
                        f"tokens={int(suffix.shape[-1])} time={elapsed:.2f}s",
                        flush=True,
                    )
                if args.score_model_logprob and score_token_count:
                    del scored, score_input_ids, score_attention_mask, labels
    complete_generation_manifest(args.output, manifest_path)
    print(f"generation complete: {total_runs} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
