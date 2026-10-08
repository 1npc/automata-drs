#!/usr/bin/env python3
"""Benchmark full ranking versus exact tie-closed adaptive top-k.

The predicate and score tensors are synthetic, so this isolates ranking and
host-transfer behavior. It is not an end-to-end generation speed claim.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.ranking import adaptive_certified_tokens  # noqa: E402


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def full_ranked_tokens(
    scores: Any,
    certify: Callable[[int], bool],
    *,
    required: int,
) -> tuple[tuple[int, ...], int]:
    # Stable sorting over the natural token-ID order implements the same
    # documented total order as adaptive_certified_tokens.
    ranked = scores.argsort(descending=True, stable=True).tolist()
    accepted: list[int] = []
    tests = 0
    for value in ranked:
        token_id = int(value)
        tests += 1
        if certify(token_id):
            accepted.append(token_id)
            if len(accepted) >= required:
                break
    return tuple(accepted), tests


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = round((len(ordered) - 1) * probability)
    return ordered[position]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocabulary-size", type=int, default=250_000)
    parser.add_argument(
        "--mapped-token-count",
        type=int,
        help="optional contiguous tokenizer-ID count; higher logits always reject",
    )
    parser.add_argument("--required", type=int, default=1)
    parser.add_argument("--initial-k", type=int, default=32)
    parser.add_argument("--certificate-modulus", type=int, default=97)
    parser.add_argument("--certificate-residues", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--source-model")
    parser.add_argument("--source-revision")
    parser.add_argument("--source-config-sha256")
    parser.add_argument("--source-tokenizer-sha256")
    parser.add_argument("--source-tokenizer-config-sha256")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in (
        "vocabulary_size",
        "required",
        "initial_k",
        "certificate_modulus",
        "certificate_residues",
        "repeats",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if (
        args.mapped_token_count is not None
        and not 0 < args.mapped_token_count <= args.vocabulary_size
    ):
        parser.error(
            "--mapped-token-count must be positive and no larger than the vocabulary"
        )
    if args.certificate_residues > args.certificate_modulus:
        parser.error("certificate residues cannot exceed the modulus")
    source_values = (
        args.source_model,
        args.source_revision,
        args.source_config_sha256,
        args.source_tokenizer_sha256,
        args.source_tokenizer_config_sha256,
    )
    if any(value is not None for value in source_values) and not all(
        value is not None for value in source_values
    ):
        parser.error(
            "all source model/revision/config/tokenizer fields must be supplied together"
        )
    if args.source_revision is not None and (
        len(args.source_revision) != 40
        or any(char not in "0123456789abcdef" for char in args.source_revision)
    ):
        parser.error("--source-revision must be a lowercase 40-hex commit")
    for name in (
        "source_config_sha256",
        "source_tokenizer_sha256",
        "source_tokenizer_config_sha256",
    ):
        value = getattr(args, name)
        if value is not None and (
            len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            parser.error(
                f"--{name.replace('_', '-')} must be lowercase 64-hex"
            )
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")

    try:
        import torch
    except ImportError as error:
        parser.error("this benchmark requires the optional Torch runtime")
        raise AssertionError("unreachable") from error
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    scores = torch.randn(
        args.vocabulary_size,
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).to(dtype=dtype)

    def certify(token_id: int) -> bool:
        if (
            args.mapped_token_count is not None
            and token_id >= args.mapped_token_count
        ):
            return False
        mixed = (token_id * 2_654_435_761 + args.seed) & 0xFFFFFFFF
        return mixed % args.certificate_modulus < args.certificate_residues

    def run_full() -> tuple[tuple[int, ...], int, float]:
        synchronize(torch, device)
        started = time.perf_counter()
        selected, tests = full_ranked_tokens(
            scores, certify, required=args.required
        )
        synchronize(torch, device)
        return selected, tests, time.perf_counter() - started

    def run_adaptive():
        synchronize(torch, device)
        started = time.perf_counter()
        selected, stats = adaptive_certified_tokens(
            scores,
            certify,
            required=args.required,
            initial_k=args.initial_k,
        )
        synchronize(torch, device)
        return selected, stats, time.perf_counter() - started

    for _ in range(args.warmup):
        reference, _, _ = run_full()
        observed, _, _ = run_adaptive()
        if observed != reference:
            parser.error("adaptive ranking disagreed during warmup")

    full_times: list[float] = []
    adaptive_times: list[float] = []
    full_tests: list[int] = []
    adaptive_tests: list[int] = []
    peak_pools: list[int] = []
    cumulative_pools: list[int] = []
    adaptive_python_values: list[int] = []
    rounds: list[int] = []
    tie_additions: list[int] = []
    fallbacks = 0
    mismatches: list[dict[str, object]] = []
    selected_reference: tuple[int, ...] = ()
    for repeat in range(args.repeats):
        if repeat % 2:
            observed, adaptive_stats, adaptive_seconds = run_adaptive()
            reference, reference_tests, full_seconds = run_full()
        else:
            reference, reference_tests, full_seconds = run_full()
            observed, adaptive_stats, adaptive_seconds = run_adaptive()
        selected_reference = reference
        full_times.append(full_seconds)
        adaptive_times.append(adaptive_seconds)
        full_tests.append(reference_tests)
        adaptive_tests.append(adaptive_stats.certificate_tests)
        peak_pools.append(adaptive_stats.peak_candidate_pool)
        cumulative_pools.append(adaptive_stats.cumulative_candidate_pool)
        adaptive_python_values.append(
            adaptive_stats.python_scalar_values_materialized
        )
        rounds.append(adaptive_stats.topk_rounds)
        tie_additions.append(adaptive_stats.boundary_ties_added)
        fallbacks += int(adaptive_stats.full_vocabulary_reached)
        if observed != reference:
            mismatches.append(
                {"repeat": repeat, "reference": reference, "adaptive": observed}
            )

    full_median = statistics.median(full_times)
    adaptive_median = statistics.median(adaptive_times)
    report = {
        "schema_version": 4,
        "benchmark": "tie-closed-adaptive-token-ranking",
        "source_contract": {
            "model": args.source_model,
            "revision": args.source_revision,
            "config_sha256": args.source_config_sha256,
            "tokenizer_sha256": args.source_tokenizer_sha256,
            "tokenizer_config_sha256": args.source_tokenizer_config_sha256,
        },
        "scope": {
            "synthetic_scores_and_certificate_predicate": True,
            "end_to_end_generation_speed_claim": False,
            "integrated_into_frozen_publication_generator": False,
        },
        "ranking_order": {
            "primary": "descending_score",
            "tie_break": "ascending_token_id",
            "full_reference_argsort_stable": True,
            # The frozen publication processor calls argsort without
            # stable=True.  This benchmark therefore proves equivalence to
            # its deterministic reference, not yet to that processor on tied
            # model logits.
            "frozen_publication_argsort_stable": False,
            "frozen_publication_equivalence_claim": False,
        },
        "parameters": {
            "vocabulary_size": args.vocabulary_size,
            "mapped_token_count": args.mapped_token_count,
            "unmapped_logit_positions": (
                0
                if args.mapped_token_count is None
                else args.vocabulary_size - args.mapped_token_count
            ),
            "required": args.required,
            "initial_k": args.initial_k,
            "certificate_modulus": args.certificate_modulus,
            "certificate_residues": args.certificate_residues,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "seed": args.seed,
            "device": str(device),
            "dtype": args.dtype,
        },
        "environment": {
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "device_total_memory_bytes": (
                int(torch.cuda.get_device_properties(device).total_memory)
                if device.type == "cuda"
                else None
            ),
            "device_compute_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "correctness": {
            "mismatches": len(mismatches),
            "examples": mismatches[:5],
            "selected_token_ids": selected_reference,
        },
        "full_ranking": {
            "python_scalar_values_materialized_per_repeat": args.vocabulary_size,
            "materialization_contents": "token_ids",
            "certificate_tests_median": statistics.median(full_tests),
            "seconds_median": full_median,
            "seconds_p95": percentile(full_times, 0.95),
        },
        "adaptive": {
            "peak_candidate_pool_median": statistics.median(peak_pools),
            "peak_candidate_pool_max": max(peak_pools),
            "cumulative_candidate_pool_median": statistics.median(
                cumulative_pools
            ),
            "python_scalar_values_materialized_median": statistics.median(
                adaptive_python_values
            ),
            "materialization_contents": "candidate_ids_and_scores_each_round",
            "certificate_tests_median": statistics.median(adaptive_tests),
            "topk_rounds_median": statistics.median(rounds),
            "boundary_ties_added_max": max(tie_additions),
            "full_vocabulary_fallbacks": fallbacks,
            "seconds_median": adaptive_median,
            "seconds_p95": percentile(adaptive_times, 0.95),
        },
        "comparison": {
            "median_wall_time_speedup": (
                full_median / adaptive_median if adaptive_median else None
            ),
            "candidate_id_transfer_reduction": (
                1.0
                - statistics.median(cumulative_pools)
                / args.vocabulary_size
            ),
            "python_scalar_materialization_reduction": (
                1.0
                - statistics.median(adaptive_python_values)
                / args.vocabulary_size
            ),
            "device_to_host_scalar_transfer_reduction": (
                1.0
                - statistics.median(adaptive_python_values)
                / args.vocabulary_size
                if device.type == "cuda"
                else None
            ),
            "device_to_host_transfer_applies": device.type == "cuda",
        },
        "implementation": {
            "ranking_sha256": file_sha256(
                REPOSITORY_ROOT / "src" / "automata_drs" / "ranking.py"
            ),
            "benchmark_sha256": file_sha256(Path(__file__)),
            "publication_processor_sha256": file_sha256(
                REPOSITORY_ROOT / "src" / "automata_drs" / "subword.py"
            ),
            "publication_generator_sha256": file_sha256(
                REPOSITORY_ROOT / "scripts" / "generate_causal_lm_baseline.py"
            ),
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(rendered)
    print(rendered, end="")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
