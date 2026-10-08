#!/usr/bin/env python3
"""Generate all unconstrained beam candidates for the frozen PostW baseline."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import (  # noqa: E402
    check_sbn_state_wellformedness,
    validate_sbn,
)
from automata_drs.beam_selection import (  # noqa: E402
    U_BEAM_CANDIDATE_CONFIGURATION,
    validate_beam_candidate_resume_prefix,
)
from automata_drs.causal_lm import (  # noqa: E402
    build_sbn_prompt,
    extract_one_line_sbn,
    iter_sbn_records,
)
from automata_drs.model_loading import (  # noqa: E402
    MODEL_LOADERS,
    load_generation_model,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _trim_suffix(sequence: Any, input_tokens: int, eos_token_id: int) -> Any:
    suffix = sequence[input_tokens:]
    values = [int(token_id) for token_id in suffix.tolist()]
    if eos_token_id in values:
        suffix = suffix[: values.index(eos_token_id) + 1]
    return suffix


def _score_suffixes(
    model: Any,
    torch: Any,
    prompt_ids: Any,
    suffixes: list[Any],
    *,
    pad_token_id: int,
) -> tuple[list[float], list[float]]:
    """Teacher-force a beam group in one forward pass under raw model logits."""

    prompt_length = int(prompt_ids.shape[-1])
    lengths = [int(suffix.shape[-1]) for suffix in suffixes]
    width = prompt_length + max(lengths)
    batch = torch.full(
        (len(suffixes), width),
        int(pad_token_id),
        dtype=prompt_ids.dtype,
        device=prompt_ids.device,
    )
    attention = torch.zeros_like(batch)
    for index, suffix in enumerate(suffixes):
        length = lengths[index]
        batch[index, :prompt_length] = prompt_ids[0]
        batch[index, prompt_length : prompt_length + length] = suffix
        attention[index, : prompt_length + length] = 1

    with torch.inference_mode():
        scored = model(input_ids=batch, attention_mask=attention)
    logits = scored.logits[:, :-1, :].float()
    targets = batch[:, 1:]
    token_logprob = torch.log_softmax(logits, dim=-1).gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for index, length in enumerate(lengths):
        mask[index, prompt_length - 1 : prompt_length - 1 + length] = True
    totals = (token_logprob * mask).sum(dim=-1)
    total_values = [float(value) for value in totals.tolist()]
    mean_values = [
        total / length for total, length in zip(total_values, lengths, strict=True)
    ]
    return total_values, mean_values


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
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.num_beams < 2:
        parser.error("num-beams must be at least two")
    if args.progress_every <= 0:
        parser.error("progress-every must be positive")
    manifest_path = Path(str(args.output) + ".manifest.json")
    if not args.resume and (args.output.exists() or manifest_path.exists()):
        parser.error(f"output or manifest already exists: {args.output}")
    if args.output.exists() and not manifest_path.is_file():
        parser.error(f"resume output has no generation manifest: {args.output}")

    records = list(iter_sbn_records(args.path, start=args.start, limit=args.limit))
    if not records:
        parser.error(f"no records found in {args.path}")
    identifiers = [record.identifier for record in records]
    requested_contract: dict[str, Any] = {
        "schema_version": 1,
        "kind": "unconstrained-beam-candidates",
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
    }
    contract_fields = tuple(requested_contract)
    contract_fields = tuple(field for field in contract_fields if field != "status")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(args.output) + ".lock")
    lock_source = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_source.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error(f"another writer holds the beam candidate lock: {lock_path}")

    stored_manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        try:
            stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"cannot read beam candidate manifest: {error}")
        mismatches = [
            field
            for field in contract_fields
            if stored_manifest.get(field) != requested_contract.get(field)
        ]
        if mismatches:
            parser.error(
                "beam candidate manifest differs from request: "
                + ", ".join(mismatches)
            )
    else:
        _write_manifest(manifest_path, requested_contract)

    existing_rows: list[dict[str, Any]] = []
    if args.output.exists():
        try:
            with args.output.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        parser.error(
                            f"blank beam candidate row at {args.output}:{line_number}"
                        )
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        parser.error(
                            f"beam candidate row is not an object at line {line_number}"
                        )
                    existing_rows.append(value)
        except json.JSONDecodeError as error:
            parser.error(f"invalid beam candidate resume JSONL: {error}")
    try:
        completed_records = validate_beam_candidate_resume_prefix(
            existing_rows, identifiers, num_beams=args.num_beams
        )
    except ValueError as error:
        parser.error(str(error))
    if completed_records:
        print(
            f"validated beam candidate resume: {completed_records}/{len(records)}",
            flush=True,
        )
    if (
        stored_manifest is not None
        and stored_manifest.get("status") == "complete"
        and completed_records != len(records)
    ):
        parser.error("complete beam candidate manifest has incomplete output")
    if completed_records == len(records):
        if stored_manifest is None or stored_manifest.get("status") != "complete":
            parser.error("complete beam rows have no complete manifest")
        if stored_manifest.get("output_sha256") != _sha256(args.output):
            parser.error("complete beam candidate output hash differs from manifest")
        print("beam candidate output is already complete", flush=True)
        return 0

    try:
        import peft
        import torch
        import transformers
        from peft import PeftModel
        from transformers import AutoTokenizer
    except ImportError as error:
        parser.error("install the LoRA runtime with: pip install -e '.[lora]'")
        raise AssertionError("unreachable") from error

    load_kwargs: dict[str, Any] = {"revision": args.revision}
    if args.cache_dir:
        load_kwargs["cache_dir"] = str(args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.eos_token_id is None:
        parser.error("tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        parser.error("tokenizer has no usable padding token")
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

    runtime_manifest = {
        **requested_contract,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }
    _write_manifest(manifest_path, runtime_manifest)

    output_mode = "a" if args.output.exists() else "x"
    with args.output.open(output_mode, encoding="utf-8") as sink:
        for record_index, record in enumerate(records):
            if record_index < completed_records:
                continue
            encoded = tokenizer(
                build_sbn_prompt(record.source),
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
            ).to("cuda")
            input_tokens = int(encoded["input_ids"].shape[-1])
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            torch.cuda.synchronize()
            generation_elapsed = time.perf_counter() - started
            sequences = generated.sequences
            sequence_scores = generated.sequences_scores
            if int(sequences.shape[0]) != args.num_beams or sequence_scores is None:
                raise RuntimeError("generation did not return the complete beam group")
            suffixes = [
                _trim_suffix(
                    sequences[rank], input_tokens, int(tokenizer.eos_token_id)
                )
                for rank in range(args.num_beams)
            ]

            torch.cuda.synchronize()
            score_started = time.perf_counter()
            raw_totals, raw_means = _score_suffixes(
                model,
                torch,
                encoded["input_ids"],
                suffixes,
                pad_token_id=int(tokenizer.pad_token_id),
            )
            torch.cuda.synchronize()
            score_elapsed = time.perf_counter() - score_started
            peak_gpu_bytes = torch.cuda.max_memory_allocated()

            rows: list[dict[str, Any]] = []
            previous_sequence_score = math.inf
            for rank, suffix in enumerate(suffixes):
                sequence_score = float(sequence_scores[rank].item())
                if not math.isfinite(sequence_score):
                    raise RuntimeError("generation returned a non-finite beam score")
                if sequence_score > previous_sequence_score + 1e-12:
                    raise RuntimeError("generation returned beams outside score order")
                previous_sequence_score = sequence_score
                suffix_ids = [int(token_id) for token_id in suffix.tolist()]
                raw_prediction = tokenizer.decode(
                    suffix_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                prediction = extract_one_line_sbn(raw_prediction)
                validation_started = time.perf_counter()
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
                validation_ms = (time.perf_counter() - validation_started) * 1000
                ended_with_eos = int(tokenizer.eos_token_id) in suffix_ids
                surface_ids = [
                    token_id
                    for token_id in suffix_ids
                    if token_id != int(tokenizer.eos_token_id)
                ]
                if int(tokenizer.eos_token_id) in suffix_ids:
                    surface_ids = surface_ids[
                        : suffix_ids.index(int(tokenizer.eos_token_id))
                    ]
                canonical_ids = [
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
                    "configuration": U_BEAM_CANDIDATE_CONFIGURATION,
                    "decode": {
                        "strategy": "beam",
                        "num_beams": args.num_beams,
                        "num_return_sequences": args.num_beams,
                    },
                    "beam_candidate": {
                        "rank": rank,
                        "sequence_score": sequence_score,
                        "validation_ms": validation_ms,
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
                        "generation_ms": generation_elapsed * 1000,
                        "model_score_ms": score_elapsed * 1000,
                        "constraint_ms": 0.0,
                    },
                    "score": {
                        "raw_model_logprob": raw_totals[rank],
                        "raw_model_mean_logprob": raw_means[rank],
                        "scored_tokens": len(suffix_ids),
                    },
                    "generation": {
                        "input_tokens": input_tokens,
                        "output_tokens_including_eos": len(suffix_ids),
                        "decoder_steps": len(suffix_ids),
                        "ended_with_eos": ended_with_eos,
                        "canonical_tokenization": surface_ids == canonical_ids,
                        "hit_max_new_tokens": (
                            len(suffix_ids) >= args.max_new_tokens
                            and not ended_with_eos
                        ),
                        "peak_gpu_bytes": peak_gpu_bytes,
                    },
                    "constraint": None,
                }
                rows.append(row)
            sink.write(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            )
            sink.flush()
            if record_index == 0 or (record_index + 1) % args.progress_every == 0:
                valid_count = sum(
                    row["validity"]["raw_sbn"] is True
                    and row["validity"]["raw_clf_referee"] is True
                    for row in rows
                )
                print(
                    f"{record_index + 1}/{len(records)} {record.identifier}: "
                    f"W-valid candidates={valid_count}/{args.num_beams} "
                    f"time={generation_elapsed:.2f}s",
                    flush=True,
                )
            del generated, sequences, sequence_scores, suffixes, encoded

    complete_manifest = {
        **runtime_manifest,
        "status": "complete",
        "records": len(records),
        "rows": len(records) * args.num_beams,
        "output_sha256": _sha256(args.output),
    }
    _write_manifest(manifest_path, complete_manifest)
    print(f"beam candidate generation complete: {len(records)} records", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
