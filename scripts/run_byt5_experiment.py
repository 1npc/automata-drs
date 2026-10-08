#!/usr/bin/env python3
"""Run reproducible unconstrained/structural/semantic ByT5 experiments."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import (  # noqa: E402
    ByT5SBNLogitsProcessor,
    SBNPolicy,
    check_sbn_state_wellformedness,
    validate_sbn,
)


def iter_records(
    path: Path, limit: int, start: int = 0
) -> Iterator[dict[str, str]]:
    emitted = 0
    available = 0
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if "\t" not in line:
                raise ValueError(f"expected source<TAB>SBN at {path}:{line_number}")
            if available < start:
                available += 1
                continue
            source_text, reference = line.split("\t", 1)
            yield {
                "id": f"{path}:{line_number}",
                "input": source_text,
                "reference": reference.rstrip("\r"),
            }
            emitted += 1
            if limit >= 0 and emitted >= limit:
                return


def resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def policy_for(configuration: str) -> SBNPolicy | None:
    if configuration == "U":
        return None
    base = {"BF": "B", "WF": "W"}.get(configuration, configuration)
    return SBNPolicy.for_configuration(base)


def strategy_parameters(strategy: str) -> dict[str, Any]:
    if strategy == "greedy":
        return {"strategy": "greedy", "num_beams": 1}
    if strategy.startswith("beam") and strategy[4:].isdigit():
        beams = int(strategy[4:])
        if beams < 2:
            raise ValueError("beam width must be at least 2")
        return {"strategy": "beam", "num_beams": beams}
    raise ValueError(f"invalid strategy {strategy!r}; use greedy or beamN")


class GenerationStepCounter:
    """Count decoder calls without retaining every step's logits tensor."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        self.calls += 1
        return scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="TSV source/reference corpus")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="XiaoZhang98/byT5-DRS")
    parser.add_argument("--revision")
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=["U", "S", "R", "G", "B", "BF", "W", "WF"],
        default=["U", "B", "W"],
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["greedy", "beam4"],
        help="greedy and/or beamN entries",
    )
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="skip this many non-empty corpus records before applying --limit",
    )
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--bounded-completion-window",
        type=int,
        default=None,
        help=(
            "for BF/WF, require a certified accepting completion during the "
            "last N decoder steps; default is the full max-new-tokens budget"
        ),
    )
    parser.add_argument(
        "--certified-candidates-per-row",
        type=int,
        default=None,
        help=(
            "lazy BF/WF certificate limit after the first step; default is 1 "
            "for greedy and 2 per beam row, while 0 exhaustively certifies all"
        ),
    )
    parser.add_argument(
        "--completion-budget-factor",
        type=float,
        default=1.0,
        help=(
            "BF/WF completion reserve in (0,1]; smaller values require the "
            "certificate to fit inside this fraction of the remaining budget"
        ),
    )
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument(
        "--score-model-logprob",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="teacher-force each prediction to record its raw LM log-probability",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="warm model kernels and constraint tables before timed records",
    )
    parser.add_argument(
        "--schedule",
        choices=["rotate", "fixed"],
        default="rotate",
        help="rotate config/strategy order by example to reduce timing-order bias",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    strategies = {name: strategy_parameters(name) for name in args.strategies}
    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if (
        args.bounded_completion_window is not None
        and args.bounded_completion_window <= 0
    ):
        parser.error("--bounded-completion-window must be positive")
    if (
        args.certified_candidates_per_row is not None
        and args.certified_candidates_per_row < 0
    ):
        parser.error("--certified-candidates-per-row must be non-negative")
    if not 0 < args.completion_budget_factor <= 1:
        parser.error("--completion-budget-factor must be in (0, 1]")
    effective_bounded_window = (
        args.max_new_tokens
        if args.bounded_completion_window is None
        else args.bounded_completion_window
    )
    bounded_global_invariant = effective_bounded_window >= args.max_new_tokens
    if (
        any(configuration in {"BF", "WF"} for configuration in args.configurations)
        and not bounded_global_invariant
    ):
        print(
            "warning: a bounded-completion window smaller than max-new-tokens "
            "is a late-repair ablation and cannot guarantee that the prefix "
            "is still completable when the window activates",
            file=sys.stderr,
        )
    records = list(iter_records(args.path, args.limit, args.start))
    if not records:
        parser.error(f"no source/reference records found in {args.path}")
    if args.output.exists() and not args.resume:
        parser.error(f"output already exists; use --resume: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    completed: set[tuple[str, str, str]] = set()
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                decode = row["decode"]
                strategy_key = (
                    "greedy"
                    if decode["strategy"] == "greedy"
                    else f"beam{decode['num_beams']}"
                )
                completed.add((row["id"], row["configuration"], strategy_key))

    try:
        import torch
        import transformers
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            LogitsProcessorList,
        )
    except ImportError as error:
        parser.error("install the model runtime with: pip install -e '.[hf]'")
        raise AssertionError("unreachable") from error

    device = resolve_device(torch, args.device)
    load_kwargs = {"revision": args.revision} if args.revision else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, **load_kwargs)
    model.to(device)
    model.eval()

    expected = {
        "vocab_size": 384,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "decoder_start_token_id": 0,
    }
    actual = {
        "vocab_size": model.config.vocab_size,
        "pad_token_id": model.config.pad_token_id,
        "eos_token_id": model.config.eos_token_id,
        "decoder_start_token_id": model.config.decoder_start_token_id,
    }
    if actual != expected or tokenizer.__class__.__name__ != "ByT5Tokenizer":
        raise ValueError(
            "the exact byte adapter does not match this checkpoint: "
            f"tokenizer={tokenizer.__class__.__name__}, config={actual}"
        )

    warmup_seconds = 0.0
    if args.warmup:
        warmup_started = time.perf_counter()
        # Prime the global Unicode-prefix tables and an initial exact byte
        # mask outside the timed region.
        warmup_processor = ByT5SBNLogitsProcessor(
            policy=SBNPolicy(),
            decoder_start_token_id=model.config.decoder_start_token_id,
        )
        warmup_processor.allowed_token_ids_for_prefix(
            [model.config.decoder_start_token_id]
        )

        warmup_inputs = tokenizer(
            records[0]["input"],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(device)
        # Warm each distinct beam shape used by the requested matrix. The cap
        # bounds setup cost while exercising the same decoder kernels.
        for beams in sorted({decode["num_beams"] for decode in strategies.values()}):
            with torch.inference_mode():
                warmup_output = model.generate(
                    **warmup_inputs,
                    do_sample=False,
                    num_beams=beams,
                    max_new_tokens=min(args.max_new_tokens, 64),
                    length_penalty=args.length_penalty,
                )
            del warmup_output
        if args.score_model_logprob:
            reference_tokens = tokenizer(
                text_target=records[0]["reference"],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_new_tokens,
            )["input_ids"].to(device)
            reference_tokens[reference_tokens == model.config.pad_token_id] = -100
            with torch.inference_mode():
                warmup_score = model(**warmup_inputs, labels=reference_tokens)
            del warmup_score
        synchronize(torch, device)
        if device == "mps":
            torch.mps.empty_cache()
        warmup_seconds = time.perf_counter() - warmup_started

    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest = {
        "status": "running",
        "started_at_unix": time.time(),
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer": tokenizer.__class__.__name__,
        "model_config": actual,
        "data": str(args.path),
        "configurations": args.configurations,
        "strategies": strategies,
        "limit": args.limit,
        "start": args.start,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "requested_bounded_completion_window": args.bounded_completion_window,
        "effective_bounded_completion_window": effective_bounded_window,
        "bounded_completion_global_invariant": bounded_global_invariant,
        "requested_certified_candidates_per_row": (
            args.certified_candidates_per_row
        ),
        "completion_budget_factor": args.completion_budget_factor,
        "length_penalty": args.length_penalty,
        "score_model_logprob": args.score_model_logprob,
        "warmup": args.warmup,
        "warmup_seconds": warmup_seconds,
        "schedule": args.schedule,
        "device": device,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    mode = "a" if args.resume else "x"
    written = 0
    with args.output.open(mode, encoding="utf-8") as sink:
        configuration_names = list(args.configurations)
        strategy_names = list(strategies)
        for record_index, record in enumerate(records):
            inputs = tokenizer(
                record["input"],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
            ).to(device)
            if args.schedule == "rotate":
                config_shift = record_index % len(configuration_names)
                strategy_shift = record_index % len(strategy_names)
                record_configurations = (
                    configuration_names[config_shift:]
                    + configuration_names[:config_shift]
                )
                record_strategies = (
                    strategy_names[strategy_shift:]
                    + strategy_names[:strategy_shift]
                )
            else:
                record_configurations = configuration_names
                record_strategies = strategy_names
            for configuration in record_configurations:
                policy = policy_for(configuration)
                bounded_window = (
                    effective_bounded_window
                    if configuration in {"BF", "WF"}
                    else 0
                )
                for strategy_key in record_strategies:
                    decode = strategies[strategy_key]
                    certified_candidate_limit: int | None = None
                    initial_certified_candidate_limit: int | None = None
                    if bounded_window:
                        if args.certified_candidates_per_row is None:
                            certified_candidate_limit = (
                                1
                                if decode["num_beams"] == 1
                                else 2
                            )
                        elif args.certified_candidates_per_row > 0:
                            certified_candidate_limit = (
                                args.certified_candidates_per_row
                            )
                        if certified_candidate_limit is not None:
                            initial_certified_candidate_limit = (
                                certified_candidate_limit
                                if decode["num_beams"] == 1
                                else max(
                                    certified_candidate_limit,
                                    2 * decode["num_beams"],
                                )
                            )
                    key = (record["id"], configuration, strategy_key)
                    if key in completed:
                        continue

                    processor = (
                        None
                        if policy is None
                        else ByT5SBNLogitsProcessor(
                            policy=policy,
                            decoder_start_token_id=model.config.decoder_start_token_id,
                            max_new_tokens=(
                                args.max_new_tokens if bounded_window else None
                            ),
                            bounded_completion_window=bounded_window,
                            max_certified_candidates=certified_candidate_limit,
                            initial_max_certified_candidates=(
                                initial_certified_candidate_limit
                            ),
                            completion_budget_factor=(
                                args.completion_budget_factor
                            ),
                        )
                    )
                    step_counter = GenerationStepCounter()
                    logits_processors = LogitsProcessorList(
                        [step_counter]
                        if processor is None
                        else [step_counter, processor]
                    )
                    if device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats()
                    synchronize(torch, device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            logits_processor=logits_processors,
                            do_sample=False,
                            num_beams=decode["num_beams"],
                            max_new_tokens=args.max_new_tokens,
                            length_penalty=args.length_penalty,
                            return_dict_in_generate=True,
                        )
                    synchronize(torch, device)
                    elapsed = time.perf_counter() - started
                    output = generated.sequences
                    output_token_ids = [int(token_id) for token_id in output[0].tolist()]
                    ended_with_eos = model.config.eos_token_id in output_token_ids[1:]

                    raw_model_logprob: float | None = None
                    raw_model_mean_logprob: float | None = None
                    score_elapsed = 0.0
                    score_token_count = max(0, int(output.shape[-1]) - 1)
                    if args.score_model_logprob and score_token_count:
                        labels = output[:, 1:].clone()
                        labels[labels == model.config.pad_token_id] = -100
                        score_token_count = int((labels != -100).sum().item())
                        synchronize(torch, device)
                        score_started = time.perf_counter()
                        with torch.inference_mode():
                            scored = model(**inputs, labels=labels)
                        synchronize(torch, device)
                        score_elapsed = time.perf_counter() - score_started
                        raw_model_mean_logprob = -float(scored.loss.item())
                        raw_model_logprob = (
                            raw_model_mean_logprob * score_token_count
                        )

                    prediction = tokenizer.decode(
                        output[0],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    structural = validate_sbn(prediction)
                    semantic_valid: bool | None = None
                    semantic_reason: str | None = None
                    if structural.valid:
                        semantic = check_sbn_state_wellformedness(structural.state)
                        semantic_valid = semantic.valid
                        semantic_reason = semantic.reason

                    processor_stats = processor.stats() if processor is not None else None
                    cache = processor.controller.cache_info() if processor is not None else None
                    completion_cache = (
                        processor.completion_cache_info()
                        if processor is not None
                        else None
                    )
                    generated_tokens = max(0, int(output.shape[-1]) - 1)
                    decoder_steps = step_counter.calls
                    result = {
                        **record,
                        "prediction": prediction,
                        "configuration": configuration,
                        "decode": decode,
                        "validity": {
                            "sbn": structural.valid,
                            "sbn_reason": structural.reason,
                            "clf_referee": semantic_valid,
                            "clf_referee_reason": semantic_reason,
                        },
                        "timing": {
                            "generation_ms": elapsed * 1000,
                            "constraint_ms": (
                                processor_stats.seconds * 1000
                                if processor_stats is not None
                                else 0.0
                            ),
                            "model_score_ms": score_elapsed * 1000,
                        },
                        "score": {
                            "raw_model_logprob": raw_model_logprob,
                            "raw_model_mean_logprob": raw_model_mean_logprob,
                            "scored_tokens": score_token_count,
                        },
                        "generation": {
                            "output_tokens_including_eos": generated_tokens,
                            "decoder_steps": decoder_steps,
                            "ended_with_eos": ended_with_eos,
                            "hit_max_new_tokens": (
                                decoder_steps >= args.max_new_tokens
                                and not ended_with_eos
                            ),
                        },
                        "constraint": (
                            None
                            if (
                                processor_stats is None
                                or cache is None
                                or completion_cache is None
                            )
                            else {
                                **asdict(processor_stats),
                                "allowed_mean": (
                                    processor_stats.allowed_total / processor_stats.rows
                                    if processor_stats.rows
                                    else None
                                ),
                                "cache_hits": cache.hits,
                                "cache_misses": cache.misses,
                                "cache_evictions": cache.evictions,
                                "cache_size": cache.size,
                                "allowed_cache_size": (
                                    processor.controller.allowed_cache_size
                                ),
                                "completion_cache_hits": completion_cache.hits,
                                "completion_cache_misses": completion_cache.misses,
                                "completion_cache_evictions": completion_cache.evictions,
                                "completion_cache_size": completion_cache.size,
                                "prefix_cache_size": processor.prefix_states.size,
                                "bounded_completion_window": bounded_window,
                                "bounded_completion_global_invariant": (
                                    bounded_global_invariant
                                    if bounded_window
                                    else False
                                ),
                                "max_certified_candidates": (
                                    certified_candidate_limit
                                ),
                                "initial_max_certified_candidates": (
                                    initial_certified_candidate_limit
                                ),
                                "completion_budget_factor": (
                                    args.completion_budget_factor
                                ),
                            }
                        ),
                        "peak_memory_bytes": (
                            int(torch.cuda.max_memory_allocated())
                            if device.startswith("cuda")
                            else None
                        ),
                    }
                    sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                    sink.flush()
                    written += 1
                    print(
                        f"{record['id']} {configuration} {strategy_key}: "
                        f"sbn={structural.valid} referee={semantic_valid} "
                        f"{elapsed:.3f}s",
                        flush=True,
                    )
                    # The allowed-mask probes can materialize many temporary
                    # immutable states. Drop representation caches after each
                    # matrix cell; device allocations are released only after
                    # the full rotated cell set so timing remains comparable.
                    if processor is not None:
                        processor.prefix_states.clear()
                        processor.controller.clear_cache()
                    del generated, output, logits_processors, step_counter
                    if args.score_model_logprob and score_token_count:
                        del scored
            if device == "mps":
                torch.mps.empty_cache()

    manifest.update(
        {
            "status": "complete",
            "finished_at_unix": time.time(),
            "new_records_written": written,
            "output": str(args.output),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
