from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from automata_drs.paired_analysis import (
    analyze_paired_experiment,
    exact_mcnemar_p,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def row(
    identifier: str,
    configuration: str,
    *,
    valid: bool,
    smatch: float,
    prediction: str | None = None,
) -> dict:
    prediction = prediction or f"prediction-{identifier}-{configuration}"
    return {
        "id": identifier,
        "configuration": configuration,
        "decode": {"strategy": "greedy", "num_beams": 1},
        "reference": f"reference-{identifier}",
        "prediction": prediction,
        "validity": {
            "raw_sbn": valid,
            "sbn": valid,
            "clf_referee": valid,
        },
        "semantic_scores": {"sbn_smatch": smatch},
        "score": {
            "raw_model_logprob": -2.0,
            "raw_model_mean_logprob": -0.2 if configuration == "U" else -0.3,
        },
        "timing": {
            "generation_ms": 10.0 if configuration == "U" else 12.0,
            "constraint_ms": 0.0 if configuration == "U" else 1.0,
        },
        "generation": {
            "decoder_steps": 10 if configuration == "U" else 11,
        },
    }


class PairedAnalysisTest(unittest.TestCase):
    def test_exact_mcnemar_edge_cases(self):
        self.assertEqual(1.0, exact_mcnemar_p(0, 0))
        self.assertEqual(1.0, exact_mcnemar_p(1, 1))
        self.assertEqual(0.0625, exact_mcnemar_p(5, 0))
        with self.assertRaises(ValueError):
            exact_mcnemar_p(-1, 0)

    def test_paired_bootstrap_is_deterministic_and_holm_adjusted(self):
        records = []
        for index in range(1, 5):
            identifier = str(index)
            records.append(
                row(identifier, "U", valid=False, smatch=0.0)
            )
            records.append(
                row(
                    identifier,
                    "B",
                    valid=index <= 2,
                    smatch=0.5 if index <= 2 else 0.0,
                )
            )
            records.append(
                row(identifier, "W", valid=True, smatch=0.75)
            )

        first = analyze_paired_experiment(
            records,
            configurations=("B", "W"),
            bootstrap_resamples=200,
            confidence=0.95,
            seed=42,
        )
        second = analyze_paired_experiment(
            records,
            configurations=("B", "W"),
            bootstrap_resamples=200,
            confidence=0.95,
            seed=42,
        )
        self.assertEqual(first, second)
        comparisons = {
            value["configuration"]: value
            for value in first["comparisons"]
        }
        b_validity = comparisons["B"]["metrics"]["raw_sbn_valid"]
        w_validity = comparisons["W"]["metrics"]["raw_sbn_valid"]
        self.assertEqual(2, b_validity["fixed"])
        self.assertEqual(0, b_validity["regressed"])
        self.assertEqual(0.5, b_validity["delta_mean"])
        self.assertEqual(4, w_validity["fixed"])
        self.assertGreaterEqual(
            b_validity["mcnemar_exact_p_holm"],
            b_validity["mcnemar_exact_p"],
        )
        smatch = comparisons["W"]["metrics"]["sbn_smatch"]
        self.assertEqual(0.75, smatch["delta_mean"])
        self.assertEqual(0.75, smatch["delta_ci_lower"])
        self.assertEqual(0.75, smatch["delta_ci_upper"])

        duplicate = analyze_paired_experiment(
            [
                row("1", "U", valid=False, smatch=0.0),
                row("1", "B", valid=True, smatch=0.5),
                row("1", "W", valid=True, smatch=0.5),
            ],
            configurations=("B", "W"),
            bootstrap_resamples=25,
            seed=42,
        )
        by_name = {
            value["configuration"]: value
            for value in duplicate["comparisons"]
        }
        self.assertEqual(
            by_name["B"]["bootstrap_seed"],
            by_name["W"]["bootstrap_seed"],
        )
        self.assertEqual(
            by_name["B"]["metrics"]["sbn_smatch"],
            by_name["W"]["metrics"]["sbn_smatch"],
        )

    def test_unpaired_or_missing_metrics_fail_closed(self):
        records = [
            row("1", "U", valid=True, smatch=0.5),
            row("2", "U", valid=True, smatch=0.5),
            row("1", "B", valid=True, smatch=0.5),
        ]
        with self.assertRaisesRegex(ValueError, "unpaired population"):
            analyze_paired_experiment(records, bootstrap_resamples=10)

        records.append(row("2", "B", valid=True, smatch=0.5))
        del records[-1]["semantic_scores"]
        with self.assertRaisesRegex(ValueError, "semantic_scores"):
            analyze_paired_experiment(records, bootstrap_resamples=10)

    def test_custom_baseline_and_metric_subset(self):
        records = []
        for identifier in ("1", "2"):
            baseline = row(identifier, "W", valid=False, smatch=0.0)
            treatment = row(identifier, "WWCD", valid=True, smatch=0.5)
            baseline["score"]["raw_model_mean_logprob"] = None
            treatment["score"]["raw_model_mean_logprob"] = None
            records.extend((baseline, treatment))

        report = analyze_paired_experiment(
            records,
            configurations=("WWCD",),
            baseline_configuration="W",
            metric_names=("raw_sbn_valid", "sbn_smatch"),
            bootstrap_resamples=25,
            seed=42,
        )
        self.assertEqual("W", report["parameters"]["baseline_configuration"])
        comparison = report["comparisons"][0]
        self.assertEqual(
            {"raw_sbn_valid", "sbn_smatch"}, set(comparison["metrics"])
        )
        self.assertEqual(2, comparison["metrics"]["raw_sbn_valid"]["fixed"])

        with self.assertRaisesRegex(ValueError, "exclude the baseline"):
            analyze_paired_experiment(
                records,
                configurations=("W",),
                baseline_configuration="W",
                metric_names=("sbn_smatch",),
                bootstrap_resamples=10,
            )

    def test_primary_wellformed_metric_requires_raw_usable_output(self):
        unconstrained = row("1", "U", valid=True, smatch=0.5)
        unconstrained["validity"].update(
            {
                "raw_sbn": False,
                "raw_clf_referee": None,
                "sbn": True,
                "clf_referee": True,
            }
        )
        constrained = row("1", "B", valid=True, smatch=0.5)
        constrained["validity"]["raw_clf_referee"] = True
        report = analyze_paired_experiment(
            [unconstrained, constrained],
            configurations=("B",),
            bootstrap_resamples=25,
            seed=42,
        )
        metric = report["comparisons"][0]["metrics"]["wellformed_valid"]
        self.assertEqual(0.0, metric["unconstrained_mean"])
        self.assertEqual(1.0, metric["constrained_mean"])
        self.assertEqual(1.0, metric["delta_mean"])
        self.assertEqual(1, metric["fixed"])

    def test_cli_verifies_hashes_and_writes_immutable_sidecar(self):
        prediction_rows = [
            row("1", "U", valid=False, smatch=0.0),
            row("1", "BF", valid=True, smatch=0.5),
            row("2", "U", valid=True, smatch=0.8, prediction="same"),
            row("2", "BF", valid=True, smatch=0.8, prediction="same"),
        ]
        score_rows = []
        for prediction in prediction_rows:
            score_rows.append(
                {
                    "id": prediction["id"],
                    "configuration": prediction["configuration"],
                    "decode": prediction["decode"],
                    "semantic_scores": prediction["semantic_scores"],
                }
            )
            prediction.pop("semantic_scores")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "matrix.jsonl"
            scores = root / "matrix.smatch.jsonl"
            output = root / "matrix.paired-statistics.json"
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
                        "output_sha256": sha256(
                            predictions.read_bytes()
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            Path(str(scores) + ".manifest.json").write_text(
                json.dumps(
                    {
                        "records": len(prediction_rows),
                        "input_sha256": sha256(predictions.read_bytes()).hexdigest(),
                        "output_sha256": sha256(scores.read_bytes()).hexdigest(),
                        "scoring_protocol": (
                            "seeded-deduplicated-semantic-pairs-v2"
                        ),
                        "identical_pair_score_consistency": True,
                        "unique_semantic_pairs": 3,
                        "reused_score_records": 1,
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/analyze_paired_statistics.py",
                    str(predictions),
                    "--scores",
                    str(scores),
                    "--output",
                    str(output),
                    "--configurations",
                    "BF",
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["gate"]["passed"])
            self.assertTrue(
                report["gate"]["identical_pair_score_consistency"]
            )
            self.assertEqual(
                3, report["semantic_pair_diagnostics"]["unique_semantic_pairs"]
            )
            self.assertEqual(2, report["comparisons"][0]["paired"])
            self.assertEqual(
                sha256(predictions.read_bytes()).hexdigest(),
                report["source"]["predictions_sha256"],
            )

    def test_cli_accepts_registered_non_u_baseline_and_metric_subset(self):
        prediction_rows = [
            row("1", "W", valid=False, smatch=0.0),
            row("1", "WWCD", valid=True, smatch=0.5),
            row("2", "W", valid=True, smatch=0.8, prediction="same"),
            row("2", "WWCD", valid=True, smatch=0.8, prediction="same"),
        ]
        score_rows = []
        for prediction in prediction_rows:
            score_rows.append(
                {
                    "id": prediction["id"],
                    "configuration": prediction["configuration"],
                    "decode": prediction["decode"],
                    "semantic_scores": prediction["semantic_scores"],
                }
            )
            prediction.pop("semantic_scores")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "matrix.jsonl"
            scores = root / "matrix.smatch.jsonl"
            output = root / "matrix.paired-statistics.json"
            config = root / "paired-config.json"
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
                        "rows": 4,
                        "expected_rows": 4,
                        "output_sha256": sha256(
                            predictions.read_bytes()
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            Path(str(scores) + ".manifest.json").write_text(
                json.dumps(
                    {
                        "records": 4,
                        "input_sha256": sha256(predictions.read_bytes()).hexdigest(),
                        "output_sha256": sha256(scores.read_bytes()).hexdigest(),
                        "scoring_protocol": (
                            "seeded-deduplicated-semantic-pairs-v2"
                        ),
                        "identical_pair_score_consistency": True,
                        "unique_semantic_pairs": 3,
                        "reused_score_records": 1,
                    }
                ),
                encoding="utf-8",
            )
            config.write_text(
                json.dumps(
                    {
                        "method_experiment": {
                            "paired_statistics": {
                                "registration_timing": "post-hoc-descriptive",
                                "paired_unit": "example",
                                "baseline_configuration": "W",
                                "bootstrap": "paired-percentile",
                                "bootstrap_resamples": 50,
                                "confidence": 0.95,
                                "seed": 7,
                                "mcnemar": (
                                    "two-sided-exact-conditional-binomial"
                                ),
                                "multiplicity": (
                                    "Holm within each binary metric across comparisons"
                                ),
                                "primary_metrics": [
                                    "raw_sbn_valid",
                                    "sbn_smatch"
                                ],
                                "secondary_metrics": []
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/analyze_paired_statistics.py",
                    str(predictions),
                    "--scores",
                    str(scores),
                    "--output",
                    str(output),
                    "--config",
                    str(config),
                    "--configurations",
                    "WWCD",
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("W", report["parameters"]["baseline_configuration"])
            self.assertEqual(
                {"raw_sbn_valid", "sbn_smatch"},
                set(report["comparisons"][0]["metrics"]),
            )
            self.assertFalse(report["gate"]["preregistered_parameters_used"])
            self.assertTrue(report["gate"]["registered_parameters_used"])


if __name__ == "__main__":
    unittest.main()
