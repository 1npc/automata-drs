from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from automata_drs.mechanism_contrasts import analyze_mechanism_contrasts


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def row(identifier: str, configuration: str, value: float) -> dict:
    return {
        "id": identifier,
        "configuration": configuration,
        "decode": {"strategy": "greedy", "num_beams": 1},
        "prediction": f"{identifier}-{configuration}",
        "reference": f"reference-{identifier}",
        "validity": {
            "raw_sbn": value > 0.0,
            "sbn": value > 0.0,
            "raw_clf_referee": value > 0.0,
            "clf_referee": value > 0.0,
        },
        "semantic_scores": {"sbn_smatch": value},
        "score": {
            "raw_model_logprob": -2.0,
            "raw_model_mean_logprob": -0.2 - value,
        },
        "timing": {
            "generation_ms": 10.0 + value,
            "constraint_ms": value,
        },
        "generation": {"decoder_steps": 10.0 + value},
    }


class MechanismContrastTest(unittest.TestCase):
    def test_right_minus_left_contrasts_are_paired_and_deterministic(self):
        rows = []
        for identifier in ("1", "2", "3", "4"):
            rows.extend(
                [
                    row(identifier, "U", 0.0),
                    row(identifier, "S", 0.0),
                    row(identifier, "R", 0.5),
                    row(identifier, "B", 0.5),
                    row(identifier, "BF", 1.0),
                ]
            )
        kwargs = {
            "contrasts": (("U", "S"), ("S", "R"), ("B", "BF")),
            "bootstrap_resamples": 100,
            "seed": 42,
        }
        first = analyze_mechanism_contrasts(rows, **kwargs)
        second = analyze_mechanism_contrasts(rows, **kwargs)
        self.assertEqual(first, second)
        comparisons = {
            value["contrast"]: value for value in first["comparisons"]
        }
        self.assertEqual(
            0.0,
            comparisons["S-minus-U"]["metrics"]["sbn_smatch"]["delta_mean"],
        )
        self.assertEqual(
            0.5,
            comparisons["R-minus-S"]["metrics"]["sbn_smatch"]["delta_mean"],
        )
        self.assertEqual(
            0.5,
            comparisons["BF-minus-B"]["metrics"]["sbn_smatch"]["delta_mean"],
        )
        self.assertTrue(first["diagnostics"]["shared_population_across_contrasts"])
        validity = comparisons["BF-minus-B"]["metrics"]["wellformed_valid"]
        self.assertEqual(0, validity["regressed"])
        self.assertGreaterEqual(
            validity["mcnemar_exact_p_holm"], validity["mcnemar_exact_p"]
        )

    def test_missing_or_unpaired_endpoint_fails_closed(self):
        rows = [row("1", "U", 0.0), row("1", "S", 1.0)]
        with self.assertRaisesRegex(ValueError, "requested configurations are absent"):
            analyze_mechanism_contrasts(
                rows,
                contrasts=(("S", "R"),),
                bootstrap_resamples=10,
            )
        rows.extend([row("2", "U", 0.0), row("2", "S", 1.0)])
        rows.pop()
        with self.assertRaisesRegex(ValueError, "unpaired population"):
            analyze_mechanism_contrasts(
                rows,
                contrasts=(("U", "S"),),
                bootstrap_resamples=10,
            )

    def test_cli_binds_hashes_and_refuses_overwrite(self):
        prediction_rows = [
            row("1", "U", 0.0),
            row("1", "S", 0.0),
            row("1", "R", 1.0),
        ]
        score_rows = []
        for prediction in prediction_rows:
            score_rows.append(
                {
                    "id": prediction["id"],
                    "configuration": prediction["configuration"],
                    "decode": prediction["decode"],
                    "semantic_scores": prediction.pop("semantic_scores"),
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "matrix.jsonl"
            scores = root / "matrix.smatch.jsonl"
            output = root / "matrix.mechanism-contrasts.json"
            predictions.write_text(
                "".join(json.dumps(value) + "\n" for value in prediction_rows),
                encoding="utf-8",
            )
            scores.write_text(
                "".join(json.dumps(value) + "\n" for value in score_rows),
                encoding="utf-8",
            )
            Path(str(predictions) + ".manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "rows": len(prediction_rows),
                        "expected_rows": len(prediction_rows),
                        "output_sha256": sha256(predictions.read_bytes()).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            Path(str(scores) + ".manifest.json").write_text(
                json.dumps(
                    {
                        "records": len(score_rows),
                        "input_sha256": sha256(predictions.read_bytes()).hexdigest(),
                        "output_sha256": sha256(scores.read_bytes()).hexdigest(),
                        "scoring_protocol": (
                            "seeded-deduplicated-semantic-pairs-v2"
                        ),
                        "identical_pair_score_consistency": True,
                        "unique_semantic_pairs": len(score_rows),
                        "reused_score_records": 0,
                    }
                ),
                encoding="utf-8",
            )
            command = [
                sys.executable,
                "scripts/analyze_mechanism_contrasts.py",
                str(predictions),
                "--scores",
                str(scores),
                "--output",
                str(output),
                "--contrast",
                "U:S",
                "--contrast",
                "S:R",
            ]
            completed = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["gate"]["passed"])
            self.assertEqual(
                sha256(predictions.read_bytes()).hexdigest(),
                report["source"]["predictions_sha256"],
            )
            repeated = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(0, repeated.returncode)
            self.assertIn("refusing to overwrite", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
