from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from automata_drs.amr_scoring import (
    create_amr_smatchpp,
    score_amr_pair,
    smatchpp_scoreable,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "score_amr_predictions.py"


class AMRScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.measure = create_amr_smatchpp(max_solver_seconds=30)

    def test_pair_scoring_is_exact_for_identity_and_zero_for_sentinel(self):
        graph = "(a / alpha)"
        exact = score_amr_pair(self.measure, graph, graph)
        self.assertEqual(1.0, exact["f1"])
        self.assertTrue(exact["alignment_optimal"])
        self.assertEqual(
            0.0,
            score_amr_pair(self.measure, None, graph)["f1"],
        )

    def test_penman_surface_without_concept_is_not_smatchpp_scoreable(self):
        self.assertFalse(smatchpp_scoreable(self.measure, "(ARG0)"))
        self.assertEqual(
            0.0,
            score_amr_pair(self.measure, "(ARG0)", "(a / alpha)")["f1"],
        )

    def test_complete_matrix_scores_raw_invalid_without_dropping_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "dev.amr"
            corpus.write_text(
                "A.\n\n(a / alpha)\n\n\nB.\n\n(b / beta)\n",
                encoding="utf-8",
            )
            predictions = root / "matrix.jsonl"
            rows = []
            for ordinal, graph in ((1, "(a / alpha)"), (2, "(b / beta)")):
                for configuration in ("U", "P"):
                    raw = (
                        "not an AMR"
                        if ordinal == 2 and configuration == "U"
                        else graph
                    )
                    rows.append(
                        {
                            "task": "amr",
                            "id": f"{corpus}:record:{ordinal}",
                            "reference": graph,
                            "prediction": raw if raw.startswith("(") else "",
                            "raw_prediction": raw,
                            "configuration": configuration,
                            "decode": {"strategy": "greedy", "num_beams": 1},
                        }
                    )
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            manifest = Path(str(predictions) + ".manifest.json")
            manifest.write_text(
                json.dumps(
                    {
                        "task": "amr",
                        "status": "running",
                        "path": str(corpus),
                        "start": 0,
                        "limit": -1,
                        "line_numbers": None,
                        "configurations": ["U", "P"],
                        "schedule": "fixed",
                        "num_beams": 1,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "scores.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(predictions),
                    "--output",
                    str(output),
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            verified = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(predictions),
                    "--verify-existing",
                    str(output),
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, verified.returncode, verified.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(report["gate"]["passed"])
        self.assertEqual("smatchpp", report["scorer"]["name"])
        self.assertEqual("ilp", report["scorer"]["solver"])
        self.assertTrue(report["gate"]["all_alignments_optimal"])
        self.assertEqual(1, report["groups"]["U"]["raw_official_penman"])
        self.assertEqual(2, report["groups"]["P"]["raw_official_penman"])
        self.assertEqual(1.0, report["groups"]["P"]["raw_primary_smatch"]["f1"])
        self.assertLess(report["groups"]["U"]["raw_primary_smatch"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
