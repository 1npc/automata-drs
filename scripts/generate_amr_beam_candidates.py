#!/usr/bin/env python3
"""Generate a resumable, memory-bounded U-beam candidate grid for AMR."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
import platform
from pathlib import Path
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import validate_penman  # noqa: E402
from automata_drs.amr_beam_selection import (  # noqa: E402
    BEAM_RANKING_METHOD,
    U_BEAM_CANDIDATE_CONFIGURATION,
    validate_amr_beam_candidate_resume_prefix,
)
from automata_drs.amr_causal_lm import (  # noqa: E402
    build_amr_prompt,
    extract_first_penman_graph,
    iter_amr_task_records,
)
from automata_drs.generation_manifest import sha256_file  # noqa: E402
from automata_drs.model_loading import MODEL_LOADERS, load_generation_model  # noqa: E402


KIND = "amr-unconstrained-beam-candidates"


def _write_manifest(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_rows(path: Path, parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    parser.error(f"blank AMR candidate row at {path}:{line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    parser.error(f"AMR candidate row {line_number} is not an object")
                rows.append(value)
    except json.JSONDecodeError as error:
        parser.error(f"invalid AMR candidate JSONL: {error}")
    return rows


def _safe_reason(error: BaseException) -> str:
    first_line = str(error).splitlines()[0].strip()
    return (first_line or type(error).__name__)[:240]


def _official_penman_valid(surface: str, penman: Any) -> tuple[bool, str | None]:
    if not surface.strip():
        return False, "empty surface"
    try:
        penman.decode(surface)
    except Exception as error:
        return False, _safe_reason(error)
    return True, None


def _trim_suffix(sequence: Any, input_tokens: int, eos_token_id: int) -> Any:
    suffix = sequence[input_tokens:]
    values = [int(value) for value in suffix.tolist()]
    if eos_token_id in values:
        suffix = suffix[: values.index(eos_token_id) + 1]
    return suffix


def _logprob_sum(torch: Any, logits: Any, targets: Any) -> float:
    """Score target IDs without materializing a persistent log-softmax tensor."""

    selected = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    normalizer = torch.logsumexp(logits.float(), dim=-1)
    return float((selected.float() - normalizer).sum().item())


def _stream_score_suffix(
    model: Any,
    torch: Any,
    prompt_ids: Any,
    suffix: Any,
    *,
    chunk_tokens: int,
) -> tuple[float, float]:
    """Teacher-force one suffix with fixed-size vocabulary-logit chunks.

    Only the last logit row is retained between calls. This avoids the
    ``steps * vocabulary`` score tuple produced by generation with
    ``output_scores=True`` and also avoids a full-sequence logits tensor.
    """

    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    suffix_length = int(suffix.shape[-1])
    if suffix_length <= 0:
        raise ValueError("cannot score an empty generated suffix")

    past = None
    carry = None
    with torch.inference_mode():
        for start in range(0, int(prompt_ids.shape[-1]), chunk_tokens):
            chunk = prompt_ids[:, start : start + chunk_tokens]
            output = model(
                input_ids=chunk,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = output.past_key_values
            carry = output.logits[:, -1:, :].detach().clone()
            del output
        if carry is None:
            raise ValueError("cannot score with an empty prompt")

        total = 0.0
        for start in range(0, suffix_length, chunk_tokens):
            targets = suffix[start : start + chunk_tokens].unsqueeze(0)
            total += _logprob_sum(torch, carry, targets[:, :1])
            output = model(
                input_ids=targets,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            if int(targets.shape[-1]) > 1:
                total += _logprob_sum(
                    torch, output.logits[:, :-1, :], targets[:, 1:]
                )
            past = output.past_key_values
            carry = output.logits[:, -1:, :].detach().clone()
            del output
    return total, total / suffix_length


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-loader", choices=MODEL_LOADERS, default="causal-lm")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--score-chunk-tokens", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.start < 0:
        parser.error("start must be non-negative")
    if args.limit == 0 or args.limit < -1:
        parser.error("limit must be -1 or positive")
    if args.num_beams < 2:
        parser.error("num-beams must be at least two")
    if min(args.max_input_tokens, args.max_new_tokens, args.score_chunk_tokens) <= 0:
        parser.error("token limits and score-chunk-tokens must be positive")
    if args.progress_every <= 0:
        parser.error("progress-every must be positive")

    manifest_path = Path(str(args.output) + ".manifest.json")
    if not args.resume and (args.output.exists() or manifest_path.exists()):
        parser.error(f"output or manifest already exists: {args.output}")
    if args.output.exists() and not manifest_path.is_file():
        parser.error(f"resume output has no candidate manifest: {args.output}")
    try:
        records = list(
            iter_amr_task_records(args.path, start=args.start, limit=args.limit)
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not records:
        parser.error(f"no AMR records found in {args.path}")
    identifiers = [record.identifier for record in records]

    experiment_config = REPOSITORY_ROOT / "configs" / "amr_transfer_experiment.json"
    requested_contract: dict[str, Any] = {
        "schema_version": 1,
        "kind": KIND,
        "task": "amr",
        "status": "running",
        "model": args.model,
        "model_loader": args.model_loader,
        "requested_revision": args.revision,
        "adapter": str(args.adapter),
        "path": str(args.path),
        "start": args.start,
        "limit": args.limit,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_beams,
        "configuration": U_BEAM_CANDIDATE_CONFIGURATION,
        "length_penalty": 1.0,
        "output_scores": False,
        "ranking_method": BEAM_RANKING_METHOD,
        "score_policy": "chunked-teacher-forcing-for-u-top1-and-postp-only",
        "score_chunk_tokens": args.score_chunk_tokens,
        "source_sha256": sha256_file(args.path),
        "experiment_config_sha256": sha256_file(experiment_config),
        "expected_examples": len(records),
        "expected_rows": len(records) * args.num_beams,
    }
    contract_fields = tuple(key for key in requested_contract if key != "status")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(args.output) + ".lock")
    lock_source = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_source.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error(f"another writer holds the AMR candidate lock: {lock_path}")

    stored_manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        try:
            stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"cannot read AMR candidate manifest: {error}")
        mismatches = [
            key
            for key in contract_fields
            if stored_manifest.get(key) != requested_contract.get(key)
        ]
        if mismatches:
            parser.error("AMR candidate manifest differs: " + ", ".join(mismatches))
    else:
        _write_manifest(manifest_path, requested_contract)

    existing_rows = _read_rows(args.output, parser)
    try:
        completed_examples = validate_amr_beam_candidate_resume_prefix(
            existing_rows, identifiers, num_beams=args.num_beams
        )
    except ValueError as error:
        parser.error(str(error))
    if completed_examples:
        print(
            f"validated AMR candidate resume: {completed_examples}/{len(records)}",
            flush=True,
        )
    if stored_manifest is not None and stored_manifest.get("status") == "complete":
        if completed_examples != len(records):
            parser.error("complete AMR candidate manifest has incomplete output")
        if stored_manifest.get("output_sha256") != sha256_file(args.output):
            parser.error("complete AMR candidate hash differs from output")
        print("AMR candidate output is already complete", flush=True)
        return 0
    if completed_examples == len(records):
        parser.error("complete AMR candidate rows have no complete manifest")

    try:
        import peft
        import penman
        import torch
        import transformers
        from peft import PeftModel
        from transformers import AutoTokenizer
    except ImportError as error:
        parser.error(f"AMR beam runtime is unavailable: {error}")
        raise AssertionError("unreachable") from error

    load_kwargs: dict[str, Any] = {"revision": args.revision}
    if args.cache_dir:
        load_kwargs["cache_dir"] = str(args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
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
    model = PeftModel.from_pretrained(base, args.adapter)
    model.to("cuda")
    model.eval()
    cuda_properties = torch.cuda.get_device_properties(0)
    runtime_manifest = {
        **requested_contract,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
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
    if stored_manifest is not None:
        for key in ("resolved_revision", "transformers", "torch", "peft", "penman"):
            if key in stored_manifest and stored_manifest.get(key) != runtime_manifest.get(key):
                parser.error(f"AMR candidate runtime changed on resume: {key}")
    _write_manifest(manifest_path, runtime_manifest)

    output_mode = "a" if args.output.exists() else "x"
    with args.output.open(output_mode, encoding="utf-8") as destination:
        for record_index, record in enumerate(records):
            if record_index < completed_examples:
                continue
            encoded = tokenizer(build_amr_prompt(record.source), return_tensors="pt")
            input_tokens = int(encoded["input_ids"].shape[-1])
            if input_tokens > args.max_input_tokens:
                parser.error(
                    f"{record.identifier} prompt has {input_tokens} tokens, "
                    f"exceeding max-input-tokens={args.max_input_tokens}"
                )
            encoded = encoded.to("cuda")

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            generation_started = time.perf_counter()
            with torch.inference_mode():
                sequences = model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    return_dict_in_generate=False,
                    output_scores=False,
                    length_penalty=1.0,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            torch.cuda.synchronize()
            generation_elapsed = time.perf_counter() - generation_started
            if int(sequences.shape[0]) != args.num_beams:
                raise RuntimeError("generation did not return the complete AMR beam group")
            suffixes = [
                _trim_suffix(sequence, input_tokens, int(tokenizer.eos_token_id))
                for sequence in sequences
            ]

            candidate_values: list[dict[str, Any]] = []
            valid_ranks: list[int] = []
            for rank, suffix in enumerate(suffixes):
                suffix_ids = [int(value) for value in suffix.tolist()]
                raw_prediction = tokenizer.decode(
                    suffix_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                prediction = extract_first_penman_graph(raw_prediction)
                validation_started = time.perf_counter()
                raw_local = validate_penman(raw_prediction)
                prediction_local = validate_penman(prediction)
                raw_official, raw_official_reason = _official_penman_valid(
                    raw_prediction, penman
                )
                prediction_official, prediction_official_reason = _official_penman_valid(
                    prediction, penman
                )
                validation_ms = (time.perf_counter() - validation_started) * 1000
                if raw_local.valid and raw_official:
                    valid_ranks.append(rank)
                candidate_values.append(
                    {
                        "suffix": suffix,
                        "suffix_ids": suffix_ids,
                        "raw_prediction": raw_prediction,
                        "prediction": prediction,
                        "validation_ms": validation_ms,
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
                    }
                )

            selected_rank = valid_ranks[0] if valid_ranks else 0
            required_score_ranks = {0, selected_rank}
            for rank, value in enumerate(candidate_values):
                value["raw_model_logprob"] = None
                value["raw_model_mean_logprob"] = None
                value["model_score_ms"] = 0.0
                if rank not in required_score_ranks:
                    continue
                torch.cuda.synchronize()
                score_started = time.perf_counter()
                raw_total, raw_mean = _stream_score_suffix(
                    model,
                    torch,
                    encoded["input_ids"],
                    value["suffix"],
                    chunk_tokens=args.score_chunk_tokens,
                )
                torch.cuda.synchronize()
                value["model_score_ms"] = (time.perf_counter() - score_started) * 1000
                value["raw_model_logprob"] = raw_total
                value["raw_model_mean_logprob"] = raw_mean
            peak_gpu_bytes = torch.cuda.max_memory_allocated()

            rows: list[dict[str, Any]] = []
            for rank, value in enumerate(candidate_values):
                suffix_ids = value["suffix_ids"]
                ended_with_eos = int(tokenizer.eos_token_id) in suffix_ids
                surface_ids: list[int] = []
                for token_id in suffix_ids:
                    if token_id == int(tokenizer.eos_token_id):
                        break
                    surface_ids.append(token_id)
                canonical_ids = [
                    int(token_id)
                    for token_id in tokenizer(
                        value["raw_prediction"], add_special_tokens=False
                    )["input_ids"]
                ]
                score_required = rank in required_score_ranks
                rows.append(
                    {
                        "task": "amr",
                        "id": record.identifier,
                        "input": record.source,
                        "reference": record.target,
                        "prediction": value["prediction"],
                        "raw_prediction": value["raw_prediction"],
                        "postprocessed": value["prediction"] != value["raw_prediction"],
                        "configuration": U_BEAM_CANDIDATE_CONFIGURATION,
                        "decode": {
                            "strategy": "beam",
                            "num_beams": args.num_beams,
                            "num_return_sequences": args.num_beams,
                            "output_scores": False,
                            "length_penalty": 1.0,
                        },
                        "beam_candidate": {
                            "rank": rank,
                            "ranking_method": BEAM_RANKING_METHOD,
                            "sequence_score": None,
                            "validation_ms": value["validation_ms"],
                        },
                        "validity": value["validity"],
                        "timing": {
                            "generation_ms": generation_elapsed * 1000,
                            "model_score_ms": value["model_score_ms"],
                            "validation_ms": value["validation_ms"],
                            "constraint_ms": 0.0,
                        },
                        "score": {
                            "raw_model_logprob": value["raw_model_logprob"],
                            "raw_model_mean_logprob": value["raw_model_mean_logprob"],
                            "scored_tokens": len(suffix_ids) if score_required else None,
                            "required_for_derived_matrix": score_required,
                            "method": (
                                "exact-chunked-teacher-forcing"
                                if score_required
                                else "not-scored-not-needed-for-selection"
                            ),
                            "chunk_tokens": args.score_chunk_tokens if score_required else None,
                        },
                        "generation": {
                            "input_tokens": input_tokens,
                            "output_tokens_including_eos": len(suffix_ids),
                            "decoder_steps": len(suffix_ids),
                            "ended_with_eos": ended_with_eos,
                            "canonical_tokenization": surface_ids == canonical_ids,
                            "hit_max_new_tokens": (
                                len(suffix_ids) >= args.max_new_tokens and not ended_with_eos
                            ),
                            "peak_gpu_bytes": peak_gpu_bytes,
                        },
                        "constraint": None,
                    }
                )
            destination.write(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            )
            destination.flush()
            completed = record_index + 1
            if completed == 1 or completed % args.progress_every == 0:
                print(
                    f"{completed}/{len(records)} {record.identifier}: "
                    f"valid={len(valid_ranks)}/{args.num_beams} selected={selected_rank} "
                    f"time={generation_elapsed:.2f}s",
                    flush=True,
                )

    complete_manifest = {
        **runtime_manifest,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "examples": len(records),
        "rows": len(records) * args.num_beams,
        "output_sha256": sha256_file(args.output),
    }
    _write_manifest(manifest_path, complete_manifest)
    print(f"AMR beam candidate generation complete: {len(records)} examples", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
