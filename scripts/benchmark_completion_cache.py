#!/usr/bin/env python3
"""Microbenchmark exact-budget versus dominance completion memoization.

This isolates cache behavior from model inference. It is intentionally labeled
as a repeated-state workload rather than an end-to-end decoding speed claim.
Publication runs record the same cache counters on real generations.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import (  # noqa: E402
    ByT5SBNLogitsProcessor,
    ByT5Vocabulary,
    SBNPolicy,
)


PREFIXES = (
    "",
    "male.n.02 gracefully.",
    'male.n.02 Name "Gabi',
    "male.n.02 Attrib",
    "person.n.01 Proposition >1 C",
    "time.n.08 EQU now EQU +1 time.n.08 ClockTime 0",
)


def encode_byt5(text: str, vocabulary: ByT5Vocabulary) -> list[int]:
    return [vocabulary.token_for_byte(byte) for byte in text.encode("utf-8")]


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def make_processor(max_cache_entries: int) -> ByT5SBNLogitsProcessor:
    return ByT5SBNLogitsProcessor(
        policy=SBNPolicy.for_configuration("W"),
        max_new_tokens=256,
        bounded_completion_window=256,
        max_completion_cache_entries=max_cache_entries,
    )


def states_for(processor: ByT5SBNLogitsProcessor):
    vocabulary = processor.controller.vocabulary
    states = []
    for text in PREFIXES:
        state = processor.prefix_states.state_for(
            [0, *encode_byt5(text, vocabulary)]
        )
        if state is None:
            raise AssertionError(f"benchmark prefix is invalid: {text!r}")
        states.append(state)
    return states


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--maximum-budget", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.maximum_budget <= 0:
        parser.error("--maximum-budget must be positive")

    budgets = tuple(range(args.maximum_budget, 0, -1))
    exact_results: list[int | None] = []
    exact_planner_traversals = 0
    exact_started = time.perf_counter()
    for _ in range(args.rounds):
        processor = make_processor(0)
        exact_cache: dict[tuple[object, int], int | None] = {}
        for state in states_for(processor):
            for budget in budgets:
                key = (state, budget)
                if key not in exact_cache:
                    exact_cache[key] = processor._completion_distance_within(
                        state, budget
                    )
                    exact_planner_traversals += 1
                exact_results.append(exact_cache[key])
    exact_seconds = time.perf_counter() - exact_started

    dominance_results: list[int | None] = []
    dominance_hits = 0
    dominance_misses = 0
    dominance_evictions = 0
    dominance_max_entries = 0
    dominance_started = time.perf_counter()
    for _ in range(args.rounds):
        processor = make_processor(100_000)
        for state in states_for(processor):
            for budget in budgets:
                dominance_results.append(
                    processor._completion_distance_within(state, budget)
                )
        info = processor.completion_cache_info()
        dominance_hits += info.hits
        dominance_misses += info.misses
        dominance_evictions += info.evictions
        dominance_max_entries = max(dominance_max_entries, info.size)
    dominance_seconds = time.perf_counter() - dominance_started

    mismatches = sum(
        expected != observed
        for expected, observed in zip(exact_results, dominance_results)
    )
    query_count = len(exact_results)
    report = {
        "schema_version": 1,
        "benchmark": "completion-cache-dominance",
        "scope": {
            "synthetic_repeated_state_workload": True,
            "end_to_end_generation_speed_claim": False,
            "policy": "W",
            "prefix_states": len(PREFIXES),
            "budgets_per_state": len(budgets),
            "budget_order": "descending",
            "rounds": args.rounds,
            "queries": query_count,
        },
        "exact_state_budget_cache": {
            "planner_traversals": exact_planner_traversals,
            "elapsed_seconds": exact_seconds,
        },
        "monotone_state_cache": {
            "planner_traversals": dominance_misses,
            "cache_hits": dominance_hits,
            "cache_evictions": dominance_evictions,
            "maximum_entries_per_round": dominance_max_entries,
            "elapsed_seconds": dominance_seconds,
        },
        "comparison": {
            "decision_mismatches": mismatches,
            "planner_traversal_reduction": (
                1.0 - dominance_misses / exact_planner_traversals
                if exact_planner_traversals
                else None
            ),
            "wall_time_speedup": (
                exact_seconds / dominance_seconds
                if dominance_seconds
                else None
            ),
        },
        "implementation": {
            "decoding_sha256": file_sha256(
                REPOSITORY_ROOT / "src" / "automata_drs" / "decoding.py"
            ),
            "byt5_sha256": file_sha256(
                REPOSITORY_ROOT / "src" / "automata_drs" / "byt5.py"
            ),
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(rendered)
    print(rendered, end="")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
