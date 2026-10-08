#!/usr/bin/env python3
"""Measure dependency-free ByT5 constraint-mask overhead on gold SBN."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import (  # noqa: E402
    ByT5ConstraintState,
    ByT5TokenConstraintController,
    ByT5Vocabulary,
    SBNPolicy,
)


def iter_sbn(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            line = line.rstrip("\n")
            if not line:
                continue
            yield line.split("\t", 1)[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=REPOSITORY_ROOT / "data/en/dev/standard.sbn",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--mask-every-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enumerate the full 384-token allowed set at every byte step",
    )
    parser.add_argument(
        "--configuration",
        choices=["S", "R", "G", "B", "W"],
        default="B",
    )
    args = parser.parse_args()

    vocabulary = ByT5Vocabulary()
    controller = ByT5TokenConstraintController(vocabulary)
    sequence_count = 0
    step_count = 0
    allowed_counts: list[int] = []
    started = time.perf_counter()
    policy = SBNPolicy.for_configuration(args.configuration)

    for sbn in iter_sbn(args.path):
        if args.limit >= 0 and sequence_count >= args.limit:
            break
        state = ByT5ConstraintState.initial(policy)
        for byte in sbn.encode("utf-8"):
            if args.mask_every_step:
                allowed_counts.append(len(controller.allowed_token_ids(state)))
            token_id = vocabulary.token_for_byte(byte)
            state = controller.transition(state, token_id)  # type: ignore[assignment]
            if state is None:
                raise RuntimeError(
                    f"gold prefix unexpectedly rejected in sequence {sequence_count + 1}"
                )
            step_count += 1
        if vocabulary.eos_token_id not in controller.allowed_token_ids(
            state, (vocabulary.eos_token_id,)
        ):
            raise RuntimeError(
                f"EOS unexpectedly rejected in sequence {sequence_count + 1}"
            )
        sequence_count += 1

    elapsed = time.perf_counter() - started
    cache = controller.cache_info()
    print(f"file: {args.path}")
    print(f"sequences: {sequence_count}")
    print(f"byte steps: {step_count}")
    print(f"full mask each step: {args.mask_every_step}")
    print(f"configuration: {args.configuration}")
    print(f"elapsed seconds: {elapsed:.6f}")
    print(f"steps/second: {step_count / elapsed:.1f}")
    print(f"milliseconds/step: {1000 * elapsed / max(step_count, 1):.4f}")
    if allowed_counts:
        print(f"allowed tokens mean: {statistics.fmean(allowed_counts):.2f}")
        print(f"allowed tokens min: {min(allowed_counts)}")
        print(f"allowed tokens max: {max(allowed_counts)}")
    print(f"transition cache hits: {cache.hits}")
    print(f"transition cache misses: {cache.misses}")
    print(f"transition cache size: {cache.size}")
    print(f"transition cache evictions: {cache.evictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
