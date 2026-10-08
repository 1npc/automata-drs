#!/usr/bin/env python3
"""CPU-only validation and completion of a causal generation manifest."""

from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.generation_manifest import (  # noqa: E402
    complete_generation_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()

    lock_path = Path(str(args.predictions) + ".lock")
    lock_source = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_source.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error(f"generation is still being written: {args.predictions}")
    try:
        manifest = complete_generation_manifest(args.predictions)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(
        "completed generation manifest: "
        f"{manifest['rows']}/{manifest['expected_rows']} rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
