from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from automata_drs.lora_suite import (
    collect_lora_suite,
    lora_suite_to_markdown,
    verify_lora_suite_outputs,
)


class LoRASuiteTest(unittest.TestCase):
    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    @staticmethod
    def group(configuration: str) -> dict[str, object]:
        return {
            "configuration": configuration,
            "strategy": "greedy",
            "total": 20,
            "raw_sbn_valid": 20,
            "referee_valid": 20,
            "sbn_smatch_mean": 0.9,
            "exact_serialization": 10,
            "canonical_tokenization": 20,
            "truncated": 0,
        }

    def test_rejects_unknown_artifact_scope(self):
        with self.assertRaisesRegex(ValueError, "scope must be baseline or full"):
            collect_lora_suite({}, Path("."), scope="unknown")

    def test_full_strict_scope_rejects_missing_selected_method_evidence(self):
        config = {
            "shared_recipe": {"seed": 42},
            "models": [{"key": "m", "priority": 1}],
        }
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "lora"
            runs_root.mkdir()
            self.write_json(
                runs_root / "selected-host.json",
                {"status": "selected", "selected_key": "m"},
            )
            with self.assertRaisesRegex(
                ValueError, "missing selected-host method summary"
            ):
                collect_lora_suite(
                    config,
                    runs_root,
                    scope="full",
                    require_complete_methods=True,
                )
        with self.assertRaisesRegex(ValueError, "requires full suite scope"):
            collect_lora_suite(
                {},
                Path("."),
                scope="baseline",
                require_complete_methods=True,
            )

    def test_collects_baseline_and_paired_tables(self):
        config = {
            "shared_recipe": {"seed": 42},
            "method_experiment": {
                "pilot": {"records": 400},
                "publication_matrices": [
                    {"split": "dev", "records": 1132},
                    {"split": "test", "records": 1132},
                    {"split": "tax", "records": 117},
                ],
            },
            "models": [
                {
                    "key": "m",
                    "repo_id": "example/m",
                    "revision": "abc",
                    "initialization": "base",
                    "parameters_billions": 3.0,
                    "checkpoint_parameters_billions": 3.5,
                    "baseline_generation_contract": "batch8-unscored-v1",
                    "priority": 1,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "lora"
            runs_root.mkdir()
            run_root = runs_root / "m-r16-seed42"
            self.write_json(run_root / "train_manifest.json", {"status": "complete"})
            self.write_json(
                run_root / "train_metrics.json",
                {
                    "train": {"train_loss": 1.0},
                    "eval": {"eval_loss": 0.5},
                    "wall_seconds": 60,
                },
            )
            for split in ("dev", "test", "tax"):
                self.write_json(
                    run_root / f"{split}-u-greedy.summary.json",
                    {"groups": [self.group("U")], "paired_vs_u": []},
                )
            paired_summary = {
                "groups": [self.group("U"), self.group("BF"), self.group("WF")],
                "paired_vs_u": [
                    {
                        "configuration": configuration,
                        "strategy": "greedy",
                        "raw_sbn_fixed": 2,
                        "raw_sbn_regressed": 0,
                        "sbn_smatch_delta_mean": 0.1,
                        "raw_model_mean_logprob_delta_mean": -0.2,
                        "generation_ms_delta_mean": 10,
                    }
                    for configuration in ("BF", "WF")
                ],
            }
            self.write_json(
                run_root / "dev400-u-bf-wf-greedy.summary.json", paired_summary
            )
            self.write_json(
                run_root / "tax-u-bf-wf-greedy.summary.json", paired_summary
            )
            for stem in (
                "dev1132-u-bf-wf-greedy",
                "test1132-u-bf-wf-greedy",
            ):
                self.write_json(run_root / f"{stem}.summary.json", paired_summary)
            fine_summary = {
                "groups": [
                    self.group(configuration)
                    for configuration in ("U", "S", "R", "G", "B", "W", "BF", "WF")
                ],
                "paired_vs_u": [
                    {
                        "configuration": configuration,
                        "strategy": "greedy",
                        "raw_sbn_fixed": 2,
                        "raw_sbn_regressed": 0,
                        "sbn_smatch_delta_mean": 0.1,
                        "raw_model_mean_logprob_delta_mean": -0.2,
                        "generation_ms_delta_mean": 10,
                    }
                    for configuration in ("S", "R", "G", "B", "W", "BF", "WF")
                ],
            }
            self.write_json(
                run_root / "dev400-u-s-r-g-b-w-bf-wf-greedy.summary.json",
                fine_summary,
            )
            paired_statistics = {
                "comparisons": [
                    {
                        "configuration": configuration,
                        "strategy": "greedy",
                        "metrics": {
                            "wellformed_valid": {
                                "delta_mean": 0.1,
                                "delta_ci_lower": 0.0,
                                "delta_ci_upper": 0.2,
                                "mcnemar_exact_p": 0.5,
                                "mcnemar_exact_p_holm": 1.0,
                            },
                            "sbn_smatch": {
                                "delta_mean": 0.1,
                                "delta_ci_lower": 0.05,
                                "delta_ci_upper": 0.15,
                            },
                            "raw_model_mean_logprob": {
                                "delta_mean": -0.2,
                                "delta_ci_lower": -0.3,
                                "delta_ci_upper": -0.1,
                            },
                            "generation_ms": {
                                "delta_mean": 10.0,
                                "delta_ci_lower": 8.0,
                                "delta_ci_upper": 12.0,
                            },
                        },
                    }
                    for configuration in ("BF", "WF")
                ]
            }
            self.write_json(
                run_root / "dev400-u-bf-wf-greedy.paired-statistics.json",
                paired_statistics,
            )
            self.write_json(
                run_root / "tax-u-bf-wf-greedy.paired-statistics.json",
                paired_statistics,
            )
            for stem in (
                "dev1132-u-bf-wf-greedy",
                "test1132-u-bf-wf-greedy",
            ):
                self.write_json(
                    run_root / f"{stem}.paired-statistics.json",
                    paired_statistics,
                )
            fine_statistics = {
                "comparisons": [
                    {**item, "configuration": configuration}
                    for configuration in ("S", "R", "G", "B", "W", "BF", "WF")
                    for item in paired_statistics["comparisons"][:1]
                ]
            }
            self.write_json(
                run_root
                / "dev400-u-s-r-g-b-w-bf-wf-greedy.paired-statistics.json",
                fine_statistics,
            )
            beam_groups = []
            beam_pairs = []
            for configuration in ("U", "PostW", "W", "WF"):
                group = self.group(configuration)
                group["strategy"] = "beam4"
                if configuration == "PostW":
                    group["postw_selection_records"] = 20
                    group["postw_fallbacks"] = 2
                    group["postw_w_valid_candidates_mean"] = 2.5
                beam_groups.append(group)
                if configuration != "U":
                    beam_pairs.append(
                        {
                            "configuration": configuration,
                            "strategy": "beam4",
                            "raw_sbn_fixed": 1,
                            "raw_sbn_regressed": 0,
                        }
                    )
            self.write_json(
                run_root / "dev400-u-postw-w-wf-beam4.summary.json",
                {"groups": beam_groups, "paired_vs_u": beam_pairs},
            )
            beam_statistics = {
                "comparisons": [
                    {
                        **paired_statistics["comparisons"][0],
                        "configuration": configuration,
                        "strategy": "beam4",
                    }
                    for configuration in ("PostW", "W", "WF")
                ]
            }
            self.write_json(
                run_root / "dev400-u-postw-w-wf-beam4.paired-statistics.json",
                beam_statistics,
            )
            conditioned_summary = {
                "groups": [
                    self.group(configuration)
                    for configuration in ("U", "B", "W", "BF", "WF")
                ],
                "paired_vs_u": [
                    {
                        "configuration": configuration,
                        "strategy": "greedy",
                        "wellformed_fixed": 2,
                        "wellformed_regressed": 0,
                    }
                    for configuration in ("B", "W", "BF", "WF")
                ],
            }
            conditioned_statistics = {
                "comparisons": [
                    {**paired_statistics["comparisons"][0], "configuration": configuration}
                    for configuration in ("B", "W", "BF", "WF")
                ]
            }
            for split in ("dev", "test", "tax"):
                stem = f"{split}-u-invalid-u-b-w-bf-wf-greedy-v2"
                self.write_json(
                    run_root / f"{stem}.summary.json",
                    conditioned_summary,
                )
                self.write_json(
                    run_root / f"{stem}.paired-statistics.json",
                    conditioned_statistics,
                )
            self.write_json(
                runs_root / "selected-host.json",
                {
                    "status": "selected",
                    "selected_key": "m",
                    "ranking": ["m"],
                    "candidates": [
                        {
                            "key": "m",
                            "parameters_billions": 3.0,
                            "metrics": {
                                "sbn_smatch_mean": 0.91,
                                "referee_valid_rate_total": 0.99,
                                "raw_sbn_valid_rate": 0.98,
                            },
                        }
                    ],
                },
            )
            selection_sha = hashlib.sha256(
                (runs_root / "selected-host.json").read_bytes()
            ).hexdigest()
            self.write_json(
                runs_root / "selected-method-resource-plan.json",
                {
                    "kind": "selected-lora-method-resource-plan",
                    "selected_key": "m",
                    "evidence": {
                        "config_sha256": "config-hash",
                        "selection_sha256": selection_sha,
                    },
                    "runtime_limits": {
                        "concurrent_model_processes": 1,
                        "cpu_math_threads_per_model_process": 2,
                    },
                    "upper_bounds": {
                        "generated_surfaces": 24668,
                        "theoretical_output_tokens": 6315008,
                    },
                    "gate": {"passed": True},
                },
            )
            self.write_json(
                runs_root.parent
                / "penman"
                / "m-grapes-token-lifting.json",
                {
                    "schema_version": 1,
                    "audit": "penman-bytelevel-token-lifting",
                    "source": {
                        "records": 1471,
                        "sha256": "d3f2be3224570ad433c951a2950d2271a94db4c1f4ce3f9749c9a4ae18d8b103",
                    },
                    "tokenizer": {
                        "model": "example/m",
                        "requested_revision": "abc",
                    },
                    "coverage": {
                        "denominator": 1471,
                        "exact_byte_roundtrip": 1471,
                        "accepted_with_eos": 1471,
                    },
                    "token_lengths": {
                        "median": 80,
                        "p95": 240,
                        "maximum": 500,
                    },
                    "implementation": {"config_sha256": "config-hash"},
                    "gate": {"passed": True},
                },
            )
            method_gate_names = (
                "dev400-method-pilot-gate.json",
                "dev400-layer-ablation-gate.json",
                "dev400-fine-grained-ablation-gate.json",
                "dev400-beam-selection-gate.json",
                "dev1132-publication-matrix-gate.json",
                "test1132-publication-matrix-gate.json",
                "tax-publication-matrix-gate.json",
            )
            for name in method_gate_names:
                self.write_json(run_root / name, {"passed": True})
            matrix_records = {"dev": 1132, "test": 1132, "tax": 117}
            for split, records in matrix_records.items():
                stem = (
                    "tax-u-bf-wf-greedy"
                    if split == "tax"
                    else f"{split}{records}-u-bf-wf-greedy"
                )
                predictions = run_root / f"{stem}.jsonl"
                predictions.write_text(f"{split}\n", encoding="utf-8")
                manifest = Path(f"{predictions}.manifest.json")
                self.write_json(manifest, {"status": "complete", "split": split})
                audit_name = (
                    "tax-certified-release-audit.json"
                    if split == "tax"
                    else f"{split}{records}-certified-release-audit.json"
                )
                self.write_json(
                    run_root / audit_name,
                    {
                        "schema_version": 2,
                        "audit": "sbn-certified-release",
                        "claim_boundary": (
                            "policy-relative structural/well-formedness release "
                            "only; not task-semantic correctness"
                        ),
                        "gate": {
                            "passed": True,
                            "performance_threshold_used": False,
                        },
                        "implementation": {
                            "audit_sha256": "a" * 64,
                            "release_sha256": "b" * 64,
                        },
                        "input": {
                            "configurations": ["BF", "WF"],
                            "completion_budget_factor": 1.0,
                            "predictions_sha256": hashlib.sha256(
                                predictions.read_bytes()
                            ).hexdigest(),
                            "manifest_sha256": hashlib.sha256(
                                manifest.read_bytes()
                            ).hexdigest(),
                        },
                        "population": {
                            "examples": records,
                            "rows": records * 2,
                            "unique_identities": records * 2,
                        },
                        "release": {
                            "releasable_rows": records * 2,
                            "refused_rows": 0,
                            "failures": [],
                            "reason_counts": {},
                            "guarantee_counts": {
                                "finite-horizon-structural-policy-membership": records,
                                "finite-horizon-structural-and-referee-policy-membership": records,
                            },
                        },
                    },
                )
            for split in ("dev", "test", "tax"):
                self.write_json(
                    run_root / f"{split}-u-invalid-intervention-gate-v2.json",
                    {
                        "passed": True,
                        "diagnostics": {"selected_records": 20},
                    },
                )
            suite = collect_lora_suite(
                config, runs_root, config_sha256="config-hash"
            )
            strict_suite = collect_lora_suite(
                config,
                runs_root,
                config_sha256="config-hash",
                require_complete_methods=True,
            )
            self.assertEqual(suite, strict_suite)
            self.assertEqual(7, suite["schema_version"])
            self.assertEqual("full", suite["scope"])
            selected_model = next(
                row for row in suite["models"] if row["key"] == "m"
            )
            self.assertEqual(
                {
                    "dev400",
                    "dev400-fine",
                    "dev400-beam4",
                    "dev1132",
                    "test1132",
                    "tax117",
                },
                {
                    row["evaluation"]
                    for row in selected_model["method_baselines"]
                },
            )
            release_audit_path = (
                run_root / "dev1132-certified-release-audit.json"
            )
            release_audit = json.loads(
                release_audit_path.read_text(encoding="utf-8")
            )
            tampered_release_audit = dict(release_audit)
            tampered_release_audit["release"] = {
                **release_audit["release"],
                "refused_rows": 1,
            }
            self.write_json(release_audit_path, tampered_release_audit)
            with self.assertRaisesRegex(
                ValueError, "certified release audit is not valid"
            ):
                collect_lora_suite(
                    config,
                    runs_root,
                    config_sha256="config-hash",
                    require_complete_methods=True,
                )
            self.write_json(release_audit_path, release_audit)
            suite["baseline_comparisons"] = {
                "selection_use": False,
                "constraint_headroom": [
                    {
                        "model": "m",
                        "split": "dev",
                        "records": 20,
                        "acceptance": {
                            configuration: {"accepted": 20, "rate": 1.0}
                            for configuration in ("S", "R", "G", "B", "W")
                        },
                        "first_blocking_layer": {
                            "S": 0,
                            "R": 0,
                            "G": 0,
                            "B": 0,
                            "W": 0,
                            "accepted": 20,
                        },
                        "truncation": {
                            "hit_max_new_tokens": 0,
                            "and_b_invalid": 0,
                            "and_w_invalid": 0,
                        },
                    }
                ],
                "comparisons": [
                    {
                        "split": "dev",
                        "left": "old",
                        "right": "m",
                        "paired": 20,
                        "metrics": {
                            "sbn_smatch": {
                                "delta_right_minus_left": 0.02,
                                "delta_ci_lower": 0.01,
                                "delta_ci_upper": 0.03,
                            },
                            "wellformed_valid": {
                                "delta_right_minus_left": 0.05,
                                "delta_ci_lower": 0.0,
                                "delta_ci_upper": 0.1,
                                "right_fixed": 2,
                                "right_regressed": 1,
                                "mcnemar_exact_p": 1.0,
                                "mcnemar_exact_p_holm": 1.0,
                            },
                        },
                        "constraint_headroom": {
                            "w_validity_overlap": {
                                "both_valid": 17,
                                "left_only_valid": 1,
                                "right_only_valid": 2,
                                "both_invalid": 0,
                            }
                        },
                    }
                ],
            }
            markdown = lora_suite_to_markdown(suite)

            output_json = runs_root / "suite.json"
            output_markdown = runs_root / "suite.md"
            output_json.write_text(
                json.dumps(suite, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            output_markdown.write_text(markdown, encoding="utf-8")
            verify_lora_suite_outputs(suite, output_json, output_markdown)

            input_hashes = suite["evidence"]["input_sha256"]
            self.assertIn(str(run_root / "train_manifest.json"), input_hashes)
            self.assertEqual("config-hash", suite["evidence"]["config_sha256"])

            baseline_suite = collect_lora_suite(
                config,
                runs_root,
                config_sha256="config-hash",
                scope="baseline",
            )
            self.assertEqual(4, baseline_suite["schema_version"])
            self.assertEqual("baseline", baseline_suite["scope"])
            self.assertIsNone(baseline_suite["method_resource_plan"])
            self.assertIsNone(baseline_suite["cross_representation"])
            self.assertTrue(
                all(not model["comparisons"] for model in baseline_suite["models"])
            )
            self.assertFalse(
                any(
                    "paired-statistics" in path
                    or "dev400-" in path
                    or "selected-method-resource-plan" in path
                    or "certified-release" in path
                    or "grapes-token-lifting" in path
                    for path in baseline_suite["evidence"]["input_sha256"]
                )
            )
            self.write_json(
                run_root / "dev400-u-bf-wf-greedy.summary.json",
                {"later_method_artifact": True},
            )
            self.assertEqual(
                baseline_suite,
                collect_lora_suite(
                    config,
                    runs_root,
                    config_sha256="config-hash",
                    scope="baseline",
                ),
            )

            output_markdown.write_text(markdown + "tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Markdown differs"):
                verify_lora_suite_outputs(suite, output_json, output_markdown)

        self.assertEqual("complete", suite["models"][0]["training"]["status"])
        self.assertEqual(3, len(suite["models"][0]["baselines"]))
        self.assertEqual(16, len(suite["models"][0]["comparisons"]))
        self.assertEqual(12, len(suite["models"][0]["interventions"]))
        self.assertEqual(10, len(suite["method_gates"]))
        self.assertEqual(3, len(suite["certified_releases"]))
        self.assertEqual("m", suite["host_selection"]["selected_key"])
        self.assertIn("development results only", markdown)
        self.assertIn("Development-only host selection", markdown)
        self.assertIn("Test and tax metrics are excluded", markdown)
        self.assertIn("| 1 | `m` | 0.9100 | 0.9900 | 0.9800 | 3.00 | yes |", markdown)
        self.assertIn("Paired baseline quality comparisons", markdown)
        self.assertIn("Unconstrained baseline constraint headroom", markdown)
        self.assertIn("| `m` | dev | 20 | 20/20 | 20/20", markdown)
        self.assertIn("Paired W-validity overlap", markdown)
        self.assertIn("| dev | `old` → `m` | 17 | 1 | 2 | 0 |", markdown)
        self.assertIn("0.0200 [0.0100, 0.0300]", markdown)
        self.assertIn("batch8-unscored-v1", markdown)
        self.assertIn("3.00/3.50", markdown)
        self.assertIn("Frozen resource ceiling", markdown)
        self.assertIn("24,668 returned surfaces", markdown)
        self.assertIn("20/20", markdown)
        self.assertIn("0.1000 [0.0500, 0.1500]", markdown)
        self.assertIn("0.5000/1.0000", markdown)
        self.assertIn("beam4", markdown)
        self.assertIn("| `m` | dev400-beam4 | U | beam4 |", markdown)
        self.assertIn("2/20 / 2.50", markdown)
        self.assertIn("Failure-conditioned mechanism diagnostics", markdown)
        self.assertIn("must not be reported as population", markdown)
        self.assertIn("| `m` | dev | 20 | B |", markdown)
        self.assertIn("All 10 registered", markdown)
        self.assertIn("Certified structural release", markdown)
        self.assertIn(
            "| dev | 1132 | 2264 | 2264 | 100.00% | 100.00% | 1132 | 1132 | 0 |",
            markdown,
        )
        self.assertIn("Certified policy validity is conditional on release", markdown)
        self.assertIn("not task-semantic correctness", markdown)
        self.assertIn("Cross-representation token lifting", markdown)
        self.assertEqual(
            1471,
            suite["cross_representation"]["audit"]["coverage"][
                "accepted_with_eos"
            ],
        )


if __name__ == "__main__":
    unittest.main()
