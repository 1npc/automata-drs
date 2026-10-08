#!/usr/bin/env python3
"""Generate post-primary AMR PF or WCD without changing the frozen U/P CLI."""

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
    ByteLevelFiniteHorizonConstraintLogitsProcessor,
    ByteLevelStructuralCompletionStoppingCriteria,
    ByteLevelVocabulary,
    ByteLevelWitnessCarryingLogitsProcessor,
    PENMANConstraintState,
    construct_penman_completion_suffix,
    validate_penman,
)
from automata_drs.amr_causal_lm import (  # noqa: E402
    build_amr_generation_prompt,
    extract_first_penman_graph,
    iter_amr_task_records,
    remove_amr_output_envelope,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-loader", choices=MODEL_LOADERS, default="causal-lm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--method", choices=("PF", "WCD"), default="PF")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument(
        "--stop-on-structural-completion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="stop at the PENMAN boundary before the model's transport envelope",
    )
    args = parser.parse_args()

    manifest_path = Path(str(args.output) + ".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error(f"output or manifest already exists: {args.output}")
    if args.start < 0 or args.limit <= 0:
        parser.error("start must be non-negative and limit must be positive")
    if args.max_input_tokens <= 0 or args.max_new_tokens <= 0:
        parser.error("token limits must be positive")
    if args.progress_every <= 0:
        parser.error("progress-every must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(args.output) + ".lock")
    lock_source = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_source.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error(f"another writer holds the generation lock: {lock_path}")

    try:
        import penman
        import torch
        import transformers
        from transformers import (
            AutoTokenizer,
            LogitsProcessorList,
            StoppingCriteriaList,
        )
    except ImportError as error:
        parser.error(f"PF generation dependencies are unavailable: {error}")

    load_kwargs: dict[str, object] = {"revision": args.revision}
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = str(args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    if tokenizer.eos_token_id is None:
        parser.error("model tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_generation_model(
        transformers,
        args.model,
        loader=args.model_loader,
        load_kwargs={
            **load_kwargs,
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        },
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
    vocabulary = ByteLevelVocabulary.from_tokenizer(tokenizer)

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
        "adapter": None,
        "path": str(args.path),
        "start": args.start,
        "limit": args.limit,
        "line_numbers": None,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": 1,
        "batch_size": 1,
        "configurations": [args.method],
        "score_model_logprob": False,
        "constraint_lifting": (
            "exact-bytelevel-bpe-score-ranked-penman-"
            + (
                "witness-carrying"
                if args.method == "WCD"
                else "positive-finite-horizon"
            )
            + ("-automaton-stop" if args.stop_on_structural_completion else "")
        ),
        "stop_on_structural_completion": args.stop_on_structural_completion,
        "schedule": "fixed",
        "prompt_template": "viamr-qwen3-graph-prefill",
        "output_envelope": "viamr-amr-tag",
        "source_sha256": sha256_file(args.path),
        "experiment_config_sha256": sha256_file(experiment_config),
        "expected_examples": len(records),
        "expected_rows": len(records),
        "physical_gpu_index": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": cuda_properties.name,
        "cuda_device_total_bytes": int(cuda_properties.total_memory),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "penman": penman.__version__,
    }
    with manifest_path.open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2)
        destination.write("\n")

    with args.output.open("x", encoding="utf-8") as destination:
        for record_index, record in enumerate(records, start=1):
            prompt = build_amr_generation_prompt(
                record.source, template="viamr-qwen3-graph-prefill"
            )
            encoded = tokenizer(prompt, return_tensors="pt")
            input_tokens = int(encoded["input_ids"].shape[-1])
            if input_tokens > args.max_input_tokens:
                parser.error(
                    f"{record.identifier} prompt has {input_tokens} tokens, "
                    f"exceeding max-input-tokens={args.max_input_tokens}"
                )
            encoded = encoded.to("cuda")
            prompt_token_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
            processor_class = (
                ByteLevelWitnessCarryingLogitsProcessor
                if args.method == "WCD"
                else ByteLevelFiniteHorizonConstraintLogitsProcessor
            )
            processor = processor_class(
                tokenizer=tokenizer,
                vocabulary=vocabulary,
                prompt_token_ids=prompt_token_ids,
                initial_state=PENMANConstraintState.initial(),
                incomplete_utf8_is_extendable=(
                    PENMANConstraintState.incomplete_utf8_is_extendable
                ),
                construct_surface_suffix=construct_penman_completion_suffix,
                max_new_tokens=args.max_new_tokens,
                retained_tokens_per_row=1,
                max_completion_cache_entries=20_000,
            )
            structural_stopper = (
                ByteLevelStructuralCompletionStoppingCriteria(processor)
                if args.stop_on_structural_completion
                else None
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
                    logits_processor=LogitsProcessorList([processor]),
                    stopping_criteria=stopping_criteria,
                    do_sample=False,
                    num_beams=1,
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
                and processor.prefix_can_finish(output[0].tolist())
            )
            model_suffix = tokenizer.decode(
                suffix,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            raw_prediction, envelope_removed = remove_amr_output_envelope(
                model_suffix, envelope="viamr-amr-tag"
            )
            prediction = extract_first_penman_graph(raw_prediction)
            raw_local = validate_penman(raw_prediction)
            prediction_local = validate_penman(prediction)
            raw_official, raw_official_reason = _independent_penman_valid(
                raw_prediction, penman
            )
            prediction_official, prediction_official_reason = (
                _independent_penman_valid(prediction, penman)
            )
            processor_stats = processor.stats()
            transition_cache = processor.controller.cache_info()
            completion_cache = processor.completion_cache_info()
            surface_token_ids = []
            for token_id in suffix_ids:
                if token_id == tokenizer.eos_token_id:
                    break
                surface_token_ids.append(token_id)
            canonical_token_ids = [
                int(value)
                for value in tokenizer(model_suffix, add_special_tokens=False)[
                    "input_ids"
                ]
            ]
            row = {
                "task": "amr",
                "id": record.identifier,
                "input": record.source,
                "reference": record.target,
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "postprocessed": prediction != raw_prediction,
                "configuration": args.method,
                "decode": {"strategy": "greedy", "num_beams": 1},
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
                    "model_score_ms": 0.0,
                    "validation_ms": 0.0,
                    "constraint_ms": processor_stats.seconds * 1000,
                },
                "score": {
                    "raw_model_logprob": None,
                    "raw_model_mean_logprob": None,
                    "scored_tokens": len(suffix_ids),
                },
                "generation": {
                    "input_tokens": input_tokens,
                    "output_tokens_including_eos": len(suffix_ids),
                    "decoder_steps": len(suffix_ids),
                    "ended_with_eos": ended_with_eos,
                    "ended_with_structural_completion": (
                        ended_with_structural_completion
                    ),
                    "canonical_tokenization": (
                        surface_token_ids == canonical_token_ids
                    ),
                    "hit_max_new_tokens": (
                        len(suffix_ids) >= args.max_new_tokens and not ended_with_eos
                    ),
                    "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                },
                "constraint": {
                    **asdict(processor_stats),
                    "transition_cache": asdict(transition_cache),
                    "completion_cache": asdict(completion_cache),
                },
                "transport_envelope": {
                    "kind": "viamr-amr-tag",
                    "model_suffix": model_suffix,
                    "removed": envelope_removed,
                },
            }
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            destination.flush()
            if record_index == 1 or record_index % args.progress_every == 0:
                print(
                    f"{record_index}/{len(records)} {record.identifier} "
                    f"{args.method}: "
                    f"raw_penman={raw_official} tokens={len(suffix_ids)} "
                    f"time={elapsed:.2f}s all_masked="
                    f"{processor_stats.all_masked_rows}",
                    flush=True,
                )

    complete_generation_manifest(args.output, manifest_path)
    print(f"{args.method} generation complete: {len(records)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
