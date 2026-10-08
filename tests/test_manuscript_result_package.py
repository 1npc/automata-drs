from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "build_manuscript_result_package.py"
SPEC = importlib.util.spec_from_file_location("manuscript_result_package", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ManuscriptResultPackageTest(unittest.TestCase):
    @staticmethod
    def write(path: Path, value: object) -> None:
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    @staticmethod
    def group(configuration: str, *, strategy: str = "greedy") -> dict[str, object]:
        return {
            "configuration": configuration,
            "strategy": strategy,
            "total": 2,
            "raw_sbn_valid": 2,
            "raw_referee_valid": 1 if configuration == "U" else 2,
            "sbn_smatch_mean": 0.70 if configuration == "U" else 0.71,
            "raw_model_mean_logprob_mean": -1.0 if configuration == "U" else -1.1,
            "generation_ms_mean": 10.0 if configuration == "U" else 12.0,
            "all_masked_rows": 0,
            "postw_fallbacks": 1 if configuration == "PostW" else 0,
        }

    @staticmethod
    def statistics() -> dict[str, object]:
        metric = {"delta_mean": 0.01, "delta_ci_lower": 0.0, "delta_ci_upper": 0.02}
        return {
            "metrics": {
                "sbn_smatch": metric,
                "raw_model_mean_logprob": {
                    "delta_mean": -0.1,
                    "delta_ci_lower": -0.2,
                    "delta_ci_upper": 0.0,
                },
                "generation_ms": {
                    "delta_mean": 2.0,
                    "delta_ci_lower": 1.0,
                    "delta_ci_upper": 3.0,
                },
            }
        }

    def drs_suite(self) -> dict[str, object]:
        baselines = []
        comparisons = []
        for evaluation, configurations in MODULE.DRS_EVALUATION_CONFIGURATIONS.items():
            strategy = "beam4" if evaluation == "dev400-beam4" else "greedy"
            baselines.append(
                {
                    "evaluation": evaluation,
                    "configuration": "U",
                    "strategy": strategy,
                    "group": self.group("U", strategy=strategy),
                }
            )
            for configuration in configurations:
                comparisons.append(
                    {
                        "evaluation": evaluation,
                        "configuration": configuration,
                        "strategy": strategy,
                        "group": self.group(configuration, strategy=strategy),
                        "paired": {
                            "prediction_changed": 1,
                            "wellformed_fixed": 1,
                            "wellformed_regressed": 0,
                        },
                        "statistics": self.statistics(),
                    }
                )
        releases = []
        for split in ("dev", "test", "tax"):
            releases.append(
                {
                    "split": split,
                    "audit": {
                        "population": {"examples": 2},
                        "release": {
                            "guarantee_counts": {
                                "finite-horizon-structural-policy-membership": 2,
                                "finite-horizon-structural-and-referee-policy-membership": 2,
                            }
                        },
                    },
                }
            )
        return {
            "host_selection": {"selected_key": "m"},
            "models": [
                {
                    "key": "m",
                    "method_baselines": baselines,
                    "comparisons": comparisons,
                }
            ],
            "certified_releases": releases,
        }

    @staticmethod
    def amr_release() -> dict[str, object]:
        return {
            "releasable_rows": 1,
            "refused_rows": 1,
            "guarantee_counts": {
                "final-local-and-independent-penman-membership": 1
            },
            "reason_counts": {"generation-did-not-end-with-eos": 1},
            "refusals": [
                {
                    "id": "example-2",
                    "reasons": ["generation-did-not-end-with-eos"],
                }
            ],
        }

    def amr_suite(self) -> dict[str, object]:
        matrices = []
        for split in ("pilot", "dev", "test"):
            matrices.append(
                {
                    "split": split,
                    "examples": 2,
                    "u": {"raw_official_penman": 1},
                    "p": {"raw_official_penman": 1},
                    "paired": {
                        "validity": {"repairs": 1, "regressions": 1},
                        "smatch": {
                            "u_corpus_f1": 0.70,
                            "p_corpus_f1": 0.71,
                            "delta_p_minus_u": 0.01,
                            "delta_ci_lower": 0.0,
                            "delta_ci_upper": 0.02,
                        },
                        "probability": {
                            "mean_raw_model_logprob_delta_p_minus_u": -0.1
                        },
                        "search": {
                            "u_generation_ms_mean": 10.0,
                            "p_generation_ms_mean": 12.0,
                            "p_all_masked_rows": 1,
                        },
                    },
                    "release": self.amr_release(),
                }
            )
        comparison = {
            "corpus_smatch_delta": 0.01,
            "delta_ci_lower": 0.0,
            "delta_ci_upper": 0.02,
            "repairs": 1,
            "regressions": 0,
            "exact_mcnemar_p": 0.5,
            "holm_adjusted_mcnemar_p": 1.0,
        }
        return {
            "matrices": matrices,
            "beam": {
                "comparisons": {
                    name: comparison
                    for name in ("postp_minus_u", "p_minus_u", "p_minus_postp")
                },
                "selection": {"fallbacks": 1},
                "search": {"online_p_all_masked_rows": 1},
                "release": self.amr_release(),
            },
        }

    @staticmethod
    def ready_lock(_drs: Path, _amr: Path) -> dict[str, object]:
        return {
            "state": "all-suite-evidence-ready-for-result-package",
            "drs": {"ready": True},
            "amr": {"ready": True},
        }

    def test_package_uses_only_suite_cells_and_separates_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drs = root / "drs.json"
            amr = root / "amr.json"
            self.write(drs, self.drs_suite())
            self.write(amr, self.amr_suite())
            report = MODULE.build_package(
                drs, amr, lock_builder=self.ready_lock
            )
            markdown = MODULE.to_markdown(report)

        self.assertEqual(9, len(report["drs"]["publication"]))
        self.assertEqual(8, len(report["drs"]["mechanism_ladder"]))
        self.assertEqual(4, len(report["drs"]["beam_diagnostic"]))
        self.assertTrue(all(
            row["certified_release"]["coverage"] == 1.0
            for row in report["drs"]["publication"]
            if row["condition"] in {"BF", "WF"}
        ))
        self.assertTrue(all(
            row["p_release"]["coverage"] == 0.5
            and row["p_release"]["certified_policy_validity_among_released"] == 1.0
            for row in report["amr"]["matrices"]
        ))
        self.assertIn("1/2 (50.00%) | 100.00%", markdown)
        self.assertIn("framing remains a manual decision", markdown)
        self.assertEqual(
            "outcome-dependent-do-not-assume",
            report["claim_evidence"]["amr_task_improvement"],
        )

    def test_package_refuses_missing_evaluation_specific_u(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drs_value = self.drs_suite()
            drs_value["models"][0]["method_baselines"].pop()
            drs = root / "drs.json"
            amr = root / "amr.json"
            self.write(drs, drs_value)
            self.write(amr, self.amr_suite())
            with self.assertRaisesRegex(ValueError, "U baselines"):
                MODULE.build_package(drs, amr, lock_builder=self.ready_lock)


if __name__ == "__main__":
    unittest.main()
