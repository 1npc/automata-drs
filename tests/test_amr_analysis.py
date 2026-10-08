from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCORER = REPOSITORY_ROOT / "scripts" / "score_amr_predictions.py"
ANALYZER = REPOSITORY_ROOT / "scripts" / "analyze_amr_paired.py"
PYTHON = Path(sys.executable)


class AMRPairedAnalysisTest(unittest.TestCase):
    def test_reports_validity_repair_and_paired_corpus_delta(self):
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
                    raw = "invalid" if ordinal == 2 and configuration == "U" else graph
                    valid = raw.startswith("(")
                    rows.append(
                        {
                            "task": "amr",
                            "id": f"{corpus}:record:{ordinal}",
                            "reference": graph,
                            "prediction": raw if valid else "",
                            "raw_prediction": raw,
                            "configuration": configuration,
                            "decode": {"strategy": "greedy", "num_beams": 1},
                            "validity": {
                                "raw_official_penman_reason": None if valid else "invalid"
                            },
                            "timing": {
                                "generation_ms": 10.0 if configuration == "U" else 12.0,
                                "constraint_ms": 0.0 if configuration == "U" else 1.0,
                            },
                            "score": {
                                "raw_model_mean_logprob": -1.0
                                if configuration == "U"
                                else -1.1
                            },
                            "generation": {"hit_max_new_tokens": False},
                            "constraint": (
                                None
                                if configuration == "U"
                                else {"all_masked_rows": 0}
                            ),
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
            scores = root / "scores.json"
            scoring = subprocess.run(
                [str(PYTHON), str(SCORER), str(predictions), "--output", str(scores)],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, scoring.returncode, scoring.stderr)
            report_path = root / "paired.json"
            analysis = subprocess.run(
                [
                    str(PYTHON),
                    str(ANALYZER),
                    str(predictions),
                    "--scores",
                    str(scores),
                    "--output",
                    str(report_path),
                    "--resamples",
                    "100",
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, analysis.returncode, analysis.stderr)
            verified = subprocess.run(
                [
                    str(PYTHON),
                    str(ANALYZER),
                    str(predictions),
                    "--scores",
                    str(scores),
                    "--verify-existing",
                    str(report_path),
                    "--resamples",
                    "100",
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, verified.returncode, verified.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(1, report["validity"]["repairs"])
        self.assertEqual(0, report["validity"]["regressions"])
        self.assertGreater(report["smatch"]["delta_p_minus_u"], 0.0)
        self.assertEqual(0, report["search"]["p_all_masked_rows"])
        self.assertAlmostEqual(
            -0.1,
            report["probability"]["mean_raw_model_logprob_delta_p_minus_u"],
        )


if __name__ == "__main__":
    unittest.main()
