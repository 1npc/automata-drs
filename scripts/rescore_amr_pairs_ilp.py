"""Exact-ILP Smatch++ for (id, arm, prediction) rows against a reference JSONL.

Invalid rows (valid == false) score zero matches and zero predicted triples.
Uses the registered scorer configuration (AMR standardizer, ILP solver).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from automata_drs.amr_scoring import create_amr_smatchpp, score_amr_pair  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--references", type=Path, required=True, help="JSONL with id and reference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-solver-seconds", type=int, default=240)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output exists: {args.output}")
    references = {row["id"]: row["reference"] for row in map(json.loads, args.references.open())}
    measure = create_amr_smatchpp(max_solver_seconds=args.max_solver_seconds)
    with args.output.open("x") as sink:
        for row in map(json.loads, args.predictions.open()):
            reference = references[row["id"]]
            prediction = row["prediction"] if row["valid"] else None
            score = score_amr_pair(measure, prediction, reference)
            sink.write(json.dumps({"id": row["id"], "arm": row["arm"], "matches": score["matches"],
                                   "predicted_triples": score["predicted_triples"],
                                   "gold_triples": score["gold_triples"],
                                   "alignment_optimal": score.get("alignment_optimal", True)}) + "\n")


if __name__ == "__main__":
    main()
