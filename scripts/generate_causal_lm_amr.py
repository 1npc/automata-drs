#!/usr/bin/env python3
"""Generate paired unconstrained/PENMAN-constrained AMR predictions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import os
import platform
from pathlib import Path
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import (  # noqa: E402
    ByteLevelConstraintLogitsProcessor,
    ByteLevelStructuralCompletionStoppingCriteria,
    ByteLevelVocabulary,
    PENMANConstraintState,
    validate_penman,
)
from automata_drs.amr_causal_lm import (  # noqa: E402
    AMR_OUTPUT_ENVELOPES,
    AMR_PROMPT_TEMPLATES,
    build_amr_generation_prompt,
    extract_first_penman_graph,
    iter_amr_task_records,
    remove_amr_output_envelope,
)
from automata_drs.causal_lm import (  # noqa: E402
    scheduled_generation_identities,
    validate_generation_resume_prefix,
)
from automata_drs.generation_manifest import (  # noqa: E402
    KIND as GENERATION_MANIFEST_KIND,
    SCHEMA_VERSION as GENERATION_MANIFEST_SCHEMA_VERSION,
    complete_generation_manifest,
    sha256_file,
)
from automata_drs.model_loading import MODEL_LOADERS, load_generation_model  # noqa: E402


def _safe_reason(error: BaseException) -> str:
    first_line = str(error).splitlines()[0].strip()
    return (first_line or type(error).__name__)[:240]


def _independent_penman_valid(surface: str, penman: object) -> tuple[bool, str | None]:
    if not surface.strip():
        return False, "empty surface"
    try:
        penman.decode(surface)  # type: ignore[attr-defined]
    except Exception as error:
        return False, _safe_reason(error)
    return True, None


def _read_existing_rows(path: Path, parser: argparse.ArgumentParser) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    parser.error(f"blank resume row at {path}:{line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    parser.error(f"resume row is not an object at {path}:{line_number}")
                rows.append(value)
    except json.JSONDecodeError as error:
        parser.error(f"invalid resume JSONL: {error}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-loader", choices=MODEL_LOADERS, default="causal-lm")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--prompt-template", choices=AMR_PROMPT_TEMPLATES, default="frozen-plain"
    )
    parser.add_argument(
        "--output-envelope", choices=AMR_OUTPUT_ENVELOPES, default="none"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument(
        "--configurations", nargs="+", choices=("U", "P"), default=("U", "P")
    )
    parser.add_argument(
        "--score-model-logprob", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--stop-on-structural-completion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "for constrained rows, stop after the token that first makes the "
            "representation EOS-acceptable, before any transport envelope"
        ),
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--schedule", choices=("rotate", "fixed"), default="rotate")
    args = parser.parse_args()

    manifest_path = Path(str(args.output) + ".manifest.json")
    if not args.resume and (args.output.exists() or manifest_path.exists()):
        parser.error(f"output or manifest already exists: {args.output}")
    if args.output.exists() and not manifest_path.is_file():
        parser.error(f"resume output has no generation manifest: {args.output}")
    if args.start < 0:
        parser.error("start must be non-negative")
    if args.limit == 0 or args.limit < -1:
        parser.error("limit must be -1 or positive")
    if args.max_input_tokens <= 0 or args.max_new_tokens <= 0:
        parser.error("token limits must be positive")
    if args.num_beams <= 0:
        parser.error("num-beams must be positive")
    if args.progress_every <= 0:
        parser.error("progress-every must be positive")
    if len(set(args.configurations)) != len(args.configurations):
        parser.error("configurations must not contain duplicates")
    if (
        args.prompt_template == "viamr-qwen3-graph-prefill"
        and args.output_envelope != "viamr-amr-tag"
    ):
        parser.error(
            "viamr-qwen3-graph-prefill requires output-envelope=viamr-amr-tag"
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
        import penman
        import torch
        import transformers
        from peft import PeftModel
        from transformers import (
            AutoTokenizer,
            LogitsProcessorList,
            StoppingCriteriaList,
        )
    except ImportError as error:
        parser.error("install the LoRA and score runtimes before AMR generation")
        raise AssertionError("unreachable") from error

    load_kwargs = {"revision": args.revision}
    if args.cache_dir:
        load_kwargs["cache_dir"] = str(args.cache_dir)
    tokenizer_source = args.adapter if args.adapter is not None else args.model
    tokenizer_kwargs = {} if args.adapter is not None else load_kwargs
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    if tokenizer.eos_token_id is None:
        parser.error("adapter tokenizer has no EOS token")
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
    model = (
        PeftModel.from_pretrained(base, args.adapter)
        if args.adapter is not None
        else base
    )
    model.to("cuda")
    model.eval()
    cuda_properties = torch.cuda.get_device_properties(0)

    try:
        records = list(
            iter_amr_task_records(args.path, start=args.start, limit=args.limit)
        )
    except ValueError as error:
        parser.error(str(error))
    if not records:
        parser.error(f"no records found in {args.path}")
    vocabulary = (
        None
        if args.configurations == ["U"]
        else ByteLevelVocabulary.from_tokenizer(tokenizer)
    )
    expected_identities = scheduled_generation_identities(
        records, args.configurations, schedule=args.schedule
    )

    experiment_config = REPOSITORY_ROOT / "configs" / "amr_transfer_experiment.json"
    manifest = {
        "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
        "kind": GENERATION_MANIFEST_KIND,
        "task": "amr",
        "status": "running",
        "model": args.model,
        "model_loader": args.model_loader,
        "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "adapter": None if args.adapter is None else str(args.adapter),
        "path": str(args.path),
        "start": args.start,
        "limit": args.limit,
        "line_numbers": None,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "batch_size": 1,
        "configurations": list(args.configurations),
        "score_model_logprob": args.score_model_logprob,
        "constraint_lifting": (
            "exact-bytelevel-bpe-score-ranked-penman-automaton-stop"
            if args.stop_on_structural_completion
            else "exact-bytelevel-bpe-score-ranked-penman"
        ),
        "schedule": args.schedule,
        "source_sha256": sha256_file(args.path),
        "experiment_config_sha256": sha256_file(experiment_config),
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
        "penman": penman.__version__,
    }
    if args.prompt_template != "frozen-plain":
        manifest["prompt_template"] = args.prompt_template
    if args.output_envelope != "none":
        manifest["output_envelope"] = args.output_envelope
    if args.stop_on_structural_completion:
        manifest["stop_on_structural_completion"] = True
    contract_fields = (
        "schema_version",
        "kind",
        "task",
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
        "score_model_logprob",
        "constraint_lifting",
        "schedule",
        "source_sha256",
        "experiment_config_sha256",
        "expected_examples",
        "expected_rows",
        "physical_gpu_index",
        "cuda_device_name",
        "cuda_device_total_bytes",
    )
    if args.prompt_template != "frozen-plain":
        contract_fields += ("prompt_template",)
    if args.output_envelope != "none":
        contract_fields += ("output_envelope",)
    if args.stop_on_structural_completion:
        contract_fields += ("stop_on_structural_completion",)
    if manifest_path.exists():
        try:
            stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"cannot read resume manifest: {error}")
        mismatches = [
            field
            for field in contract_fields
            if stored_manifest.get(field) != manifest.get(field)
        ]
        if mismatches:
            parser.error(
                "resume manifest differs from requested run: " + ", ".join(mismatches)
            )
    else:
        with manifest_path.open("x", encoding="utf-8") as destination:
            json.dump(manifest, destination, ensure_ascii=False, indent=2)
            destination.write("\n")

    existing_rows = _read_existing_rows(args.output, parser)
    try:
        completed_prefix = validate_generation_resume_prefix(
            existing_rows, expected_identities, num_beams=args.num_beams
        )
    except ValueError as error:
        parser.error(str(error))
    if completed_prefix == len(expected_identities):
        complete_generation_manifest(args.output, manifest_path)
        print("generation output is already complete", flush=True)
        return 0
    if completed_prefix:
        print(
            f"validated resume prefix: {completed_prefix}/{len(expected_identities)}",
            flush=True,
        )

    output_mode = "a" if args.output.exists() else "x"
    configuration_names = list(args.configurations)
    with args.output.open(output_mode, encoding="utf-8") as destination:
        completed_runs = completed_prefix
        total_runs = len(expected_identities)
        for record_index, record in enumerate(records):
            record_start = record_index * len(configuration_names)
            if completed_prefix >= record_start + len(configuration_names):
                continue
            encoded = tokenizer(
                build_amr_generation_prompt(
                    record.source, template=args.prompt_template
                ),
                return_tensors="pt",
            )
            input_tokens = int(encoded["input_ids"].shape[-1])
            if input_tokens > args.max_input_tokens:
                parser.error(
                    f"{record.identifier} prompt has {input_tokens} tokens, "
                    f"exceeding max-input-tokens={args.max_input_tokens}"
                )
            encoded = encoded.to("cuda")
            prompt_token_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
            if args.schedule == "rotate":
                shift = record_index % len(configuration_names)
                record_configurations = configuration_names[shift:] + configuration_names[:shift]
            else:
                record_configurations = configuration_names

            for configuration_index, configuration in enumerate(record_configurations):
                scheduled_index = record_start + configuration_index
                if scheduled_index < completed_prefix:
                    continue
                processor = None
                if configuration == "P":
                    processor = ByteLevelConstraintLogitsProcessor(
                        tokenizer=tokenizer,
                        vocabulary=vocabulary,
                        prompt_token_ids=prompt_token_ids,
                        initial_state=PENMANConstraintState.initial(),
                        incomplete_utf8_is_extendable=(
                            PENMANConstraintState.incomplete_utf8_is_extendable
                        ),
                        retained_tokens_per_row=(
                            1 if args.num_beams == 1 else 2 * args.num_beams
                        ),
                    )
                logits_processors = LogitsProcessorList(
                    [] if processor is None else [processor]
                )
                structural_stopper = (
                    None
                    if processor is None or not args.stop_on_structural_completion
                    else ByteLevelStructuralCompletionStoppingCriteria(processor)
                )
                stopping_criteria = StoppingCriteriaList(
                    [] if structural_stopper is None else [structural_stopper]
                )
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(
                        **encoded,
                        logits_processor=logits_processors,
                        stopping_criteria=stopping_criteria,
                        do_sample=False,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                suffix = output[0, input_tokens:]
                suffix_ids = [int(value) for value in suffix.tolist()]
                ended_with_eos = tokenizer.eos_token_id in suffix_ids
                ended_with_structural_completion = bool(
                    structural_stopper is not None
                    and processor is not None
                    and processor.prefix_can_finish(output[0].tolist())
                )

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
                    raw_model_logprob = raw_model_mean_logprob * score_token_count

                model_suffix = tokenizer.decode(
                    suffix,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                raw_prediction, envelope_removed = remove_amr_output_envelope(
                    model_suffix, envelope=args.output_envelope
                )
                prediction = extract_first_penman_graph(raw_prediction)
                validation_started = time.perf_counter()
                raw_local = validate_penman(raw_prediction)
                prediction_local = validate_penman(prediction)
                raw_official, raw_official_reason = _independent_penman_valid(
                    raw_prediction, penman
                )
                prediction_official, prediction_official_reason = (
                    _independent_penman_valid(prediction, penman)
                )
                validation_elapsed = time.perf_counter() - validation_started

                surface_token_ids = []
                for token_id in suffix_ids:
                    if token_id == tokenizer.eos_token_id:
                        break
                    surface_token_ids.append(token_id)
                canonical_token_ids = [
                    int(value)
                    for value in tokenizer(
                        model_suffix, add_special_tokens=False
                    )["input_ids"]
                ]
                processor_stats = None if processor is None else processor.stats()
                cache = None if processor is None else processor.controller.cache_info()
                constraint = None
                if processor_stats is not None and cache is not None:
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
                    }

                row = {
                    "task": "amr",
                    "id": record.identifier,
                    "input": record.source,
                    "reference": record.target,
                    "prediction": prediction,
                    "raw_prediction": raw_prediction,
                    "postprocessed": prediction != raw_prediction,
                    "configuration": configuration,
                    "decode": {
                        "strategy": "greedy" if args.num_beams == 1 else "beam",
                        "num_beams": args.num_beams,
                    },
                    "validity": {
                        "raw_local_penman": raw_local.valid,
                        "raw_local_penman_reason": raw_local.reason,
                        "raw_official_penman": raw_official,
                        "raw_official_penman_reason": raw_official_reason,
                        "prediction_local_penman": prediction_local.valid,
                        "prediction_local_penman_reason": prediction_local.reason,
                        "prediction_official_penman": prediction_official,
                        "prediction_official_penman_reason": prediction_official_reason,
                    },
                    "timing": {
                        "generation_ms": elapsed * 1000,
                        "model_score_ms": score_elapsed * 1000,
                        "validation_ms": validation_elapsed * 1000,
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
                        "output_tokens_including_eos": len(suffix_ids),
                        "decoder_steps": len(suffix_ids),
                        "ended_with_eos": ended_with_eos,
                        "ended_with_structural_completion": (
                            ended_with_structural_completion
                        ),
                        "canonical_tokenization": surface_token_ids == canonical_token_ids,
                        "hit_max_new_tokens": (
                            len(suffix_ids) >= args.max_new_tokens and not ended_with_eos
                        ),
                        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                    },
                    "constraint": constraint,
                }
                if args.output_envelope != "none":
                    row["transport_envelope"] = {
                        "kind": args.output_envelope,
                        "model_suffix": model_suffix,
                        "removed": envelope_removed,
                    }
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                destination.flush()
                completed_runs += 1
                if completed_runs == 1 or completed_runs % args.progress_every == 0:
                    print(
                        f"{completed_runs}/{total_runs} {record.identifier} "
                        f"{configuration}: raw_penman={raw_official} "
                        f"tokens={len(suffix_ids)} time={elapsed:.2f}s",
                        flush=True,
                    )
                if args.score_model_logprob and score_token_count:
                    del scored, score_input_ids, score_attention_mask, labels

    complete_generation_manifest(args.output, manifest_path)
    print(f"generation complete: {len(expected_identities)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
