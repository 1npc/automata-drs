#!/usr/bin/env python3
"""Score experiment predictions with the external PMB5 SBN Smatch script."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.scoring import parse_sbn_smatch_output  # noqa: E402


SEEDED_SCORER_RUNNER = """
import random
import runpy
import sys

seed = int(sys.argv[1])
scorer = sys.argv[2]
scorer_args = sys.argv[3:]
random.seed(seed)
try:
    import numpy
except ImportError:
    pass
else:
    numpy.random.seed(seed % (2 ** 32))
sys.argv = [scorer, *scorer_args]
runpy.run_path(scorer, run_name="__main__")
"""


def strategy_key(row):
    decode = row["decode"]
    return (
        "greedy"
        if decode["strategy"] == "greedy"
        else f"beam{decode['num_beams']}"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enclosing_git_revision(path: Path) -> str | None:
    for directory in (path.parent, *path.parents):
        if not (directory / ".git").exists():
            continue
        completed = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="experiment JSONL")
    parser.add_argument("--scorer", type=Path, required=True, help="PMB5 sbn_smatch.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.scorer.is_file():
        parser.error(f"scorer not found: {args.scorer}")
    manifest_path = Path(str(args.output) + ".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error(f"output or manifest already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    input_bytes = args.path.read_bytes()
    input_digest = hashlib.sha256(input_bytes).hexdigest()
    input_text = input_bytes.decode("utf-8")
    records = [json.loads(line) for line in input_text.splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in records:
        groups[(str(row["configuration"]), strategy_key(row))].append(row)

    # Smatch uses randomized hill climbing.  Scoring each configuration in a
    # separate subprocess therefore allowed byte-identical reference/prediction
    # pairs to receive different scores.  Score every unique semantic pair once
    # under one seeded process, then reuse that value for every experimental
    # cell.  This makes paired method contrasts invariant to configuration
    # names, scheduling, and duplicate surfaces.
    unique_pairs: list[tuple[str, str]] = []
    pair_index: dict[tuple[str, str], int] = {}
    record_pair_indices: list[int] = []
    for row in records:
        pair = (str(row["reference"]), str(row["prediction"]))
        index = pair_index.get(pair)
        if index is None:
            index = len(unique_pairs)
            pair_index[pair] = index
            unique_pairs.append(pair)
        record_pair_indices.append(index)

    unique_scores: list[float] = []
    with tempfile.TemporaryDirectory(prefix="automata-drs-smatch-") as temp:
        temp_dir = Path(temp)
        if unique_pairs:
            references = temp_dir / "unique.reference.sbn"
            predictions = temp_dir / "unique.prediction.sbn"
            references.write_text(
                "\n".join(reference for reference, _ in unique_pairs) + "\n",
                encoding="utf-8",
            )
            predictions.write_text(
                "\n".join(prediction for _, prediction in unique_pairs) + "\n",
                encoding="utf-8",
            )
            scorer_environment = dict(os.environ)
            scorer_environment["PYTHONHASHSEED"] = str(args.seed % (2**32))
            completed = subprocess.run(
                [
                    args.python,
                    "-c",
                    SEEDED_SCORER_RUNNER,
                    str(args.seed),
                    str(args.scorer.resolve()),
                    "-s1",
                    str(references),
                    "-s2",
                    str(predictions),
                ],
                cwd=args.scorer.parent,
                env=scorer_environment,
                text=True,
                capture_output=True,
            )
            if completed.returncode:
                raise RuntimeError(
                    "SBN Smatch failed for the deduplicated semantic pairs:\n"
                    f"{completed.stdout}\n{completed.stderr}"
                )
            unique_scores = parse_sbn_smatch_output(
                completed.stdout, len(unique_pairs)
            )

    score_rows = []
    grouped_scores = defaultdict(list)
    for row, index in zip(records, record_pair_indices):
        configuration = str(row["configuration"])
        strategy = strategy_key(row)
        score = unique_scores[index]
        grouped_scores[(configuration, strategy)].append(score)
        score_rows.append(
            {
                "id": row["id"],
                "configuration": configuration,
                "decode": row["decode"],
                "semantic_scores": {"sbn_smatch": score},
            }
        )
    for (configuration, strategy), scores in sorted(grouped_scores.items()):
        print(
            f"{configuration}/{strategy}: {sum(scores) / len(scores):.6f}",
            flush=True,
        )

    if sha256(args.path) != input_digest:
        raise RuntimeError(
            "experiment JSONL changed while SBN Smatch was running; "
            "score only a completed artifact"
        )
    with args.output.open("x", encoding="utf-8") as sink:
        for row in score_rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    scorer = args.scorer.resolve()
    manifest = {
        "input": str(args.path),
        "input_sha256": input_digest,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "records": len(records),
        "scoring_protocol": "seeded-deduplicated-semantic-pairs-v2",
        "seed": args.seed,
        "unique_semantic_pairs": len(unique_pairs),
        "reused_score_records": len(records) - len(unique_pairs),
        "pair_key": ["reference", "prediction"],
        "scorer_invocations": 1 if unique_pairs else 0,
        "identical_pair_score_consistency": True,
        "groups": [
            {
                "configuration": configuration,
                "strategy": strategy,
                "records": len(group),
            }
            for (configuration, strategy), group in sorted(groups.items())
        ],
        "scorer": str(scorer),
        "scorer_sha256": sha256(scorer),
        "scorer_git_revision": enclosing_git_revision(scorer),
        "python_executable": args.python,
        "python": sys.version,
        "platform": platform.platform(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
