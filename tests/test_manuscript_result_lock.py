from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "check_manuscript_result_lock.py"
SPEC = importlib.util.spec_from_file_location("manuscript_result_lock", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ManuscriptResultLockTest(unittest.TestCase):
    @staticmethod
    def write(path: Path, value: object) -> None:
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def test_missing_suites_keep_every_result_family_locked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = MODULE.build_report(root / "drs.json", root / "amr.json")
        self.assertTrue(report["manuscript_lock"]["passed"])
        self.assertEqual(
            "locked-waiting-for-final-suite-evidence", report["state"]
        )
        self.assertFalse(report["can_integrate"]["drs_results"])
        self.assertFalse(report["can_integrate"]["amr_results"])
        self.assertFalse(
            report["manuscript_lock"]["manual_log_or_partial_jsonl_numbers_allowed"]
        )

    def test_partial_or_superficial_suite_cannot_unlock_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drs = root / "drs.json"
            amr = root / "amr.json"
            self.write(
                drs,
                {
                    "schema_version": 7,
                    "scope": "full",
                    "host_selection": {"status": "selected", "selected_key": "m"},
                    "models": [{"key": "m", "comparisons": []}],
                    "method_gates": [],
                    "certified_releases": [],
                    "evidence": {"input_sha256": {}},
                },
            )
            self.write(
                amr,
                {
                    "schema_version": 1,
                    "audit": "amr-transfer-suite",
                    "gate": {"passed": True, "performance_threshold_used": False},
                    "matrices": [],
                },
            )
            report = MODULE.build_report(drs, amr)
        self.assertEqual(
            "locked-waiting-for-final-suite-evidence", report["state"]
        )
        self.assertFalse(report["drs"]["ready"])
        self.assertFalse(report["amr"]["ready"])
        self.assertFalse(report["drs"]["checks"]["all_method_stages"])
        self.assertFalse(report["amr"]["checks"]["exact_matrix_chain"])

    def test_complete_collector_contracts_unlock_only_their_result_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drs = root / "drs.json"
            amr = root / "amr.json"
            drs_hashes = {
                "runs/lora/selected-host.json": "a" * 64,
                "runs/lora/selected-method-resource-plan.json": "a" * 64,
                "runs/penman/m-grapes-token-lifting.json": "a" * 64,
            }
            method_gates = []
            for stage in MODULE.EXPECTED_DRS_STAGES:
                gate_path = f"runs/gates/{stage}.json"
                drs_hashes[gate_path] = "a" * 64
                method_gates.append(
                    {"stage": stage, "path": gate_path, "passed": True}
                )
            certified_releases = []
            for split in MODULE.EXPECTED_DRS_RELEASE_SPLITS:
                audit_path = f"runs/release/{split}.json"
                predictions_path = f"runs/predictions/{split}.jsonl"
                manifest_path = f"{predictions_path}.manifest.json"
                for source_path in (audit_path, predictions_path, manifest_path):
                    drs_hashes[source_path] = "a" * 64
                certified_releases.append(
                    {
                        "split": split,
                        "path": audit_path,
                        "audit": {
                            "schema_version": 2,
                            "audit": "sbn-certified-release",
                            "claim_boundary": MODULE.DRS_RELEASE_CLAIM_BOUNDARY,
                            "gate": {
                                "passed": True,
                                "performance_threshold_used": False,
                            },
                            "input": {
                                "configurations": ["BF", "WF"],
                                "completion_budget_factor": 1.0,
                                "predictions_sha256": "a" * 64,
                                "manifest_sha256": "a" * 64,
                            },
                            "implementation": {
                                "audit_sha256": "b" * 64,
                                "release_sha256": "c" * 64,
                            },
                            "population": {
                                "examples": 1,
                                "rows": 2,
                                "unique_identities": 2,
                            },
                            "release": {
                                "releasable_rows": 2,
                                "refused_rows": 0,
                                "failures": [],
                                "reason_counts": {},
                                "guarantee_counts": {
                                    "finite-horizon-structural-and-referee-policy-membership": 1,
                                    "finite-horizon-structural-policy-membership": 1,
                                },
                            },
                        },
                    }
                )
            comparisons = []
            for evaluation, configurations in MODULE.EXPECTED_DRS_COMPARISONS.items():
                summary_path = f"runs/summaries/{evaluation}.json"
                statistics_path = f"runs/statistics/{evaluation}.json"
                drs_hashes[summary_path] = "a" * 64
                drs_hashes[statistics_path] = "a" * 64
                for configuration in configurations:
                    comparisons.append(
                        {
                            "evaluation": evaluation,
                            "configuration": configuration,
                            "strategy": (
                                "beam4"
                                if evaluation == "dev400-beam4"
                                else "greedy"
                            ),
                            "path": summary_path,
                            "statistics_path": statistics_path,
                            "group": {"total": 1},
                            "paired": {"pairs": 1},
                            "statistics": {"bootstrap_samples": 1},
                        }
                    )
            method_baselines = []
            for evaluation in MODULE.EXPECTED_DRS_EVALUATIONS:
                summary_path = f"runs/summaries/{evaluation}.json"
                method_baselines.append(
                    {
                        "evaluation": evaluation,
                        "configuration": "U",
                        "strategy": (
                            "beam4"
                            if evaluation == "dev400-beam4"
                            else "greedy"
                        ),
                        "path": summary_path,
                        "group": {"total": 1},
                    }
                )
            drs_suite = {
                "schema_version": 7,
                "scope": "full",
                "host_selection": {"status": "selected", "selected_key": "m"},
                "method_resource_plan": {
                    "kind": "selected-lora-method-resource-plan",
                    "selected_key": "m",
                    "gate": {"passed": True},
                },
                "method_gates": method_gates,
                "certified_releases": certified_releases,
                "cross_representation": {
                    "path": "runs/penman/m-grapes-token-lifting.json",
                    "audit": {
                        "audit": "penman-bytelevel-token-lifting",
                        "source": {"records": 1471},
                        "tokenizer": {
                            "model": "example/m",
                            "requested_revision": "revision",
                        },
                        "coverage": {
                            "denominator": 1471,
                            "exact_byte_roundtrip": 1471,
                            "accepted_with_eos": 1471,
                        },
                        "gate": {"passed": True},
                    },
                },
                "models": [
                    {
                        "key": "m",
                        "model": "example/m",
                        "revision": "revision",
                        "method_baselines": method_baselines,
                        "comparisons": comparisons,
                    }
                ],
                "evidence": {
                    "config_sha256": "d" * 64,
                    "input_sha256": drs_hashes,
                },
            }
            self.write(drs, drs_suite)
            amr_releases = {
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
            amr_hashes = {
                **{
                    f"runs/amr/{split}.gate.json": "e" * 64
                    for split in ("pilot", "dev", "test")
                },
                "runs/amr/beam.analysis.json": "e" * 64,
                "runs/amr/resource-plan.json": "e" * 64,
            }
            amr_suite = {
                "schema_version": 2,
                "audit": "amr-transfer-suite",
                "gate": {"passed": True, "performance_threshold_used": False},
                "resource_plan": {
                    "audit": "amr-full-run-resource-plan",
                    "gate": {
                        "passed": True,
                        "performance_threshold_used": False,
                    },
                },
                "training": {"train_records": 2},
                "matrices": [
                    {
                        "split": split,
                        "examples": 2,
                        "gate": f"runs/amr/{split}.gate.json",
                        "u": {"records": 2},
                        "p": {"records": 2},
                        "paired": {"population": 2},
                        "release": amr_releases,
                    }
                    for split in ("pilot", "dev", "test")
                ],
                "beam": {
                    "report_path": "runs/amr/beam.analysis.json",
                    "groups": {
                        name: {"records": 2} for name in ("U", "PostP", "P")
                    },
                    "comparisons": {
                        name: {"corpus_smatch_delta": 0.0}
                        for name in MODULE.EXPECTED_AMR_BEAM_COMPARISONS
                    },
                    "selection": {"fallbacks": 0},
                    "search": {
                        "online_p_all_masked_rows": 1,
                        "online_p_accepting_rows": 1,
                    },
                    "release": amr_releases,
                },
                "certification": {
                    "orthogonal_to_deterministic_search": True,
                    "non_guarantees": [
                        "AMR task-semantic correctness",
                        "availability of a releasable graph for every decoding attempt",
                    ],
                },
                "evidence": amr_hashes,
            }
            self.write(amr, amr_suite)
            report = MODULE.build_report(drs, amr)
            result_package = root / "result-package.json"
            package_value = {
                "schema_version": 1,
                "audit": "manuscript-result-package",
                "sources": {
                    "drs": {"sha256": MODULE._sha256(drs)},
                    "amr": {"sha256": MODULE._sha256(amr)},
                },
                "claim_boundary": {
                    "certified_policy_validity_is_conditional_on_release": True,
                    "release_coverage_is_reported_over_attempts": True,
                    "structural_certification_implies_task_semantic_correctness": False,
                    "performance_threshold_used": False,
                    "automatic_paper_framing_used": False,
                },
                "drs": {
                    "publication": [{} for _ in range(9)],
                    "mechanism_ladder": [{} for _ in range(8)],
                    "beam_diagnostic": [{} for _ in range(4)],
                },
                "amr": {
                    "matrices": [
                        {"split": split} for split in ("pilot", "dev", "test")
                    ],
                    "beam": {"comparisons": [{} for _ in range(3)]},
                },
                "gate": {"passed": True, "performance_threshold_used": False},
            }
            self.write(result_package, package_value)
            integration_report = MODULE.build_report(
                drs,
                amr,
                result_package,
                result_package_builder=lambda _drs, _amr: package_value,
            )

            drifted_package = json.loads(
                result_package.read_text(encoding="utf-8")
            )
            drifted_package["sources"]["amr"]["sha256"] = "0" * 64
            self.write(result_package, drifted_package)
            drifted_package_report = MODULE.build_report(
                drs,
                amr,
                result_package,
                result_package_builder=lambda _drs, _amr: package_value,
            )

            unbound_drs = json.loads(drs.read_text(encoding="utf-8"))
            unbound_drs["method_gates"][0]["path"] = "manual-gate.json"
            self.write(drs, unbound_drs)
            unbound_drs_report = MODULE.build_report(drs, amr)
            self.write(drs, drs_suite)

            unbound_amr = json.loads(amr.read_text(encoding="utf-8"))
            unbound_amr["matrices"][0]["gate"] = "manual-gate.json"
            self.write(amr, unbound_amr)
            unbound_amr_report = MODULE.build_report(drs, amr)
            self.write(amr, amr_suite)

            tampered_amr = json.loads(amr.read_text(encoding="utf-8"))
            tampered_amr["matrices"][0]["release"]["releasable_rows"] = 2
            self.write(amr, tampered_amr)
            tampered_report = MODULE.build_report(drs, amr)
        self.assertEqual(
            "all-suite-evidence-ready-for-result-package", report["state"]
        )
        self.assertTrue(report["can_build_result_package"])
        self.assertFalse(report["can_integrate"]["drs_results"])
        self.assertFalse(report["can_integrate"]["amr_results"])
        self.assertEqual(
            "all-suite-evidence-and-result-package-ready-for-integration",
            integration_report["state"],
        )
        self.assertTrue(integration_report["can_integrate"]["drs_results"])
        self.assertTrue(integration_report["can_integrate"]["amr_results"])
        self.assertFalse(drifted_package_report["result_package"]["ready"])
        self.assertTrue(all(report["drs"]["checks"].values()))
        self.assertTrue(all(report["amr"]["checks"].values()))
        self.assertFalse(unbound_drs_report["drs"]["checks"]["all_method_stages"])
        self.assertFalse(unbound_amr_report["amr"]["checks"]["exact_matrix_chain"])
        self.assertFalse(tampered_report["amr"]["ready"])
        self.assertFalse(tampered_report["can_integrate"]["amr_results"])


if __name__ == "__main__":
    unittest.main()
