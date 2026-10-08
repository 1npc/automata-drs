from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from automata_drs.lora_method_gate import (
    evaluate_lora_method_pilot,
    evaluate_lora_publication_matrix,
    verify_frozen_lora_method_pilot,
    verify_frozen_lora_publication_matrix,
)
from automata_drs.paired_analysis import analyze_paired_experiment


PAIRED_PARAMETERS = {
    "paired_unit": "example",
    "baseline_configuration": "U",
    "bootstrap": "paired-percentile",
    "bootstrap_resamples": 10,
    "confidence": 0.95,
    "seed": 7,
    "mcnemar": "two-sided-exact-conditional-binomial",
    "multiplicity": "Holm within each binary metric across comparisons",
    "primary_metrics": ["wellformed_valid", "sbn_smatch"],
    "secondary_metrics": [
        "raw_sbn_valid",
        "sbn_valid",
        "exact_serialization",
        "raw_model_mean_logprob",
        "generation_ms",
        "constraint_ms",
        "decoder_steps",
    ],
}


class LoraMethodGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = {
            "key": "model",
            "repo_id": "org/model",
            "revision": "a" * 40,
            "safetensors_bytes": 1_000_000_000,
        }
        self.config = {
            "method_experiment": {
                "pilot": {
                    "split": "dev",
                    "selection": "first-corpus-order",
                    "records": 2,
                    "configurations": ["U", "BF", "WF"],
                    "strategy": "greedy",
                    "max_new_tokens": 256,
                    "requirements": {
                        "u_surface_equivalent_to_frozen_baseline": True,
                        "bounded_outputs_accepting_under_policy": True,
                        "wf_semantic_wellformedness": True,
                        "all_masked_rows": 0,
                        "truncated_rows": 0,
                        "finite_unmasked_logprob_for_every_output": True,
                        "performance_threshold_used": False,
                    },
                },
                "publication_matrices": [
                    {
                        "split": "test",
                        "records": 2,
                        "configurations": ["U", "BF", "WF"],
                    }
                ],
                "paired_statistics": PAIRED_PARAMETERS,
            }
        }

    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    @staticmethod
    def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    @staticmethod
    def row(identifier: str, configuration: str) -> dict[str, object]:
        return {
            "id": identifier,
            "reference": "male.n.02",
            "raw_prediction": "male.n.02",
            "prediction": "male.n.02",
            "configuration": configuration,
            "decode": {"strategy": "greedy", "num_beams": 1},
            "validity": {
                "raw_sbn": True,
                "sbn": True,
                "clf_referee": True,
            },
            "generation": {
                "hit_max_new_tokens": False,
                "ended_with_eos": True,
                "decoder_steps": 2,
            },
            "score": {
                "raw_model_logprob": -2.0,
                "raw_model_mean_logprob": -1.0,
                "scored_tokens": 2,
            },
            "timing": {
                "generation_ms": 1.0,
                "model_score_ms": 0.5,
                "constraint_ms": 0.1 if configuration != "U" else 0.0,
            },
            "constraint": (
                None
                if configuration == "U"
                else {
                    "calls": 2,
                    "rows": 2,
                    "bounded_rows": 2,
                    "candidates_tested": 2,
                    "candidates_accepted": 2,
                    "allowed_total": 2,
                    "all_masked_rows": 0,
                    "completion_certificates_tested": 2,
                    "completion_certificates_accepted": 2,
                }
            ),
        }

    def make_complete(self, root: Path) -> tuple[Path, Path]:
        run = root / "model-r16-seed42"
        baseline = [self.row(str(index), "U") for index in range(2)]
        baseline_path = run / "dev-u-greedy.jsonl"
        self.write_jsonl(baseline_path, baseline)
        matrix = [
            self.row(str(index), configuration)
            for index in range(2)
            for configuration in ("U", "BF", "WF")
        ]
        matrix_path = run / "dev2-u-bf-wf-greedy.jsonl"
        self.write_jsonl(matrix_path, matrix)
        self.write_json(
            Path(str(matrix_path) + ".manifest.json"),
            {
                "status": "complete",
                "model": "org/model",
                "requested_revision": "a" * 40,
                "limit": 2,
                "max_new_tokens": 256,
                "num_beams": 1,
                "batch_size": 1,
                "configurations": ["U", "BF", "WF"],
                "completion_budget_factor": 1.0,
                "score_model_logprob": True,
                "constraint_lifting": "exact-bytelevel-bpe-score-ranked",
                "schedule": "rotate",
                "expected_examples": 2,
                "expected_rows": 6,
                "examples": 2,
                "rows": 6,
                "output_sha256": hashlib.sha256(
                    matrix_path.read_bytes()
                ).hexdigest(),
                "physical_gpu_index": "1",
                "cuda_device_name": "Test GPU",
                "cuda_device_total_bytes": 24_000_000_000,
            },
        )
        prefix = matrix_path.with_suffix("")
        scores_path = Path(str(prefix) + ".smatch.jsonl")
        score_rows = [
            {
                "id": row["id"],
                "configuration": row["configuration"],
                "decode": row["decode"],
                "semantic_scores": {"sbn_smatch": 1.0},
            }
            for row in matrix
        ]
        self.write_jsonl(scores_path, score_rows)
        self.write_json(
            Path(str(scores_path) + ".manifest.json"),
            {
                "input_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
                "records": 6,
                "groups": [
                    {"configuration": configuration, "strategy": "greedy", "records": 2}
                    for configuration in ("BF", "U", "WF")
                ],
            },
        )
        self.write_json(
            Path(str(prefix) + ".summary.json"),
            {
                "groups": [
                    {
                        "configuration": configuration,
                        "strategy": "greedy",
                        "total": 2,
                        "sbn_smatch_mean": 1.0,
                    }
                    for configuration in ("U", "BF", "WF")
                ],
                "paired_vs_u": [
                    {
                        "configuration": configuration,
                        "strategy": "greedy",
                        "paired": 2,
                        "missing_unconstrained": 0,
                    }
                    for configuration in ("BF", "WF")
                ],
            },
        )
        merged = []
        for value in matrix:
            item = dict(value)
            item["semantic_scores"] = {"sbn_smatch": 1.0}
            merged.append(item)
        paired = analyze_paired_experiment(
            merged,
            configurations=("BF", "WF"),
            bootstrap_resamples=10,
            confidence=0.95,
            seed=7,
        )
        paired.update(
            {
                "population_scope": "fixed-without-u-outcome-selection",
                "gate": {"passed": True},
                "source": {
                    "predictions_sha256": hashlib.sha256(
                        matrix_path.read_bytes()
                    ).hexdigest(),
                    "scores_sha256": hashlib.sha256(
                        scores_path.read_bytes()
                    ).hexdigest(),
                },
            }
        )
        self.write_json(Path(str(prefix) + ".paired-statistics.json"), paired)
        return run, matrix_path

    def test_complete_integrity_evidence_passes_without_performance_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_complete(root)
            report = evaluate_lora_method_pilot(
                self.config, self.model, root
            )
        self.assertTrue(report["passed"])
        self.assertFalse(report["evidence"]["performance_threshold_used"])

    def test_changed_u_surface_fails_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, matrix_path = self.make_complete(root)
            rows = [json.loads(line) for line in matrix_path.read_text().splitlines()]
            next(row for row in rows if row["configuration"] == "U")[
                "raw_prediction"
            ] = "female.n.02"
            self.write_jsonl(matrix_path, rows)
            report = evaluate_lora_method_pilot(
                self.config, self.model, root
            )
        check = next(item for item in report["checks"] if item["name"] == "u_surface_equivalence")
        self.assertFalse(check["passed"])
        self.assertFalse(report["passed"])

    def test_performance_threshold_cannot_be_added_after_results(self):
        self.config["method_experiment"]["pilot"]["requirements"][
            "performance_threshold_used"
        ] = True
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "frozen gate"):
                evaluate_lora_method_pilot(
                    self.config, self.model, Path(directory)
                )

    def test_bounded_labels_without_certificate_activity_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, matrix_path = self.make_complete(root)
            rows = [json.loads(line) for line in matrix_path.read_text().splitlines()]
            row = next(
                row for row in rows if row["configuration"] == "BF"
            )
            row["constraint"]["bounded_rows"] = 0
            row["constraint"]["completion_certificates_tested"] = 0
            row["constraint"]["completion_certificates_accepted"] = 0
            self.write_jsonl(matrix_path, rows)
            report = evaluate_lora_method_pilot(
                self.config, self.model, root
            )
        check = next(
            item
            for item in report["checks"]
            if item["name"] == "bounded_completion_activity"
        )
        self.assertFalse(check["passed"])
        self.assertEqual("0:BF", check["observed"][0]["cell"])
        self.assertFalse(report["passed"])

    def test_missing_physical_gpu_provenance_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, matrix_path = self.make_complete(root)
            manifest_path = Path(str(matrix_path) + ".manifest.json")
            manifest = json.loads(manifest_path.read_text())
            manifest["physical_gpu_index"] = None
            self.write_json(manifest_path, manifest)
            report = evaluate_lora_method_pilot(
                self.config, self.model, root
            )
        check = next(
            item
            for item in report["checks"]
            if item["name"] == "generation_hardware"
        )
        self.assertFalse(check["passed"])
        self.assertFalse(report["passed"])

    def test_tampered_paired_analysis_source_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, matrix_path = self.make_complete(root)
            paired_path = Path(
                str(matrix_path.with_suffix("")) + ".paired-statistics.json"
            )
            paired = json.loads(paired_path.read_text(encoding="utf-8"))
            paired["source"]["predictions_sha256"] = "0" * 64
            self.write_json(paired_path, paired)
            report = evaluate_lora_method_pilot(
                self.config, self.model, root
            )
        check = next(
            item
            for item in report["checks"]
            if item["name"] == "paired_statistics"
        )
        self.assertFalse(check["passed"])
        self.assertIn("prediction_sha256", check["observed"])
        self.assertFalse(report["passed"])

    def test_publication_test_matrix_reuses_the_same_integrity_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, dev_matrix = self.make_complete(root)
            (run / "test-u-greedy.jsonl").write_bytes(
                (run / "dev-u-greedy.jsonl").read_bytes()
            )
            test_matrix = run / "test2-u-bf-wf-greedy.jsonl"
            test_matrix.write_bytes(dev_matrix.read_bytes())
            for suffix in (
                ".jsonl.manifest.json",
                ".smatch.jsonl",
                ".smatch.jsonl.manifest.json",
                ".summary.json",
                ".paired-statistics.json",
            ):
                source = Path(str(dev_matrix.with_suffix("")) + suffix)
                target = Path(str(test_matrix.with_suffix("")) + suffix)
                target.write_bytes(source.read_bytes())
            score_manifest = Path(
                str(test_matrix.with_suffix("")) + ".smatch.jsonl.manifest.json"
            )
            value = json.loads(score_manifest.read_text())
            score_path = Path(str(test_matrix.with_suffix("")) + ".smatch.jsonl")
            value["input_sha256"] = hashlib.sha256(test_matrix.read_bytes()).hexdigest()
            value["output_sha256"] = hashlib.sha256(score_path.read_bytes()).hexdigest()
            self.write_json(score_manifest, value)
            report = evaluate_lora_publication_matrix(
                self.config, self.model, root, "test"
            )
        self.assertTrue(report["passed"])

    def test_publication_tax_matrix_accepts_explicit_full_corpus_limit(self):
        self.config["method_experiment"]["publication_matrices"] = [
            {
                "split": "tax",
                "records": 2,
                "configurations": ["U", "BF", "WF"],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, dev_matrix = self.make_complete(root)
            (run / "tax-u-greedy.jsonl").write_bytes(
                (run / "dev-u-greedy.jsonl").read_bytes()
            )
            tax_matrix = run / "tax-u-bf-wf-greedy.jsonl"
            tax_matrix.write_bytes(dev_matrix.read_bytes())
            for suffix in (
                ".jsonl.manifest.json",
                ".smatch.jsonl",
                ".smatch.jsonl.manifest.json",
                ".summary.json",
                ".paired-statistics.json",
            ):
                source = Path(str(dev_matrix.with_suffix("")) + suffix)
                target = Path(str(tax_matrix.with_suffix("")) + suffix)
                target.write_bytes(source.read_bytes())
            score_path = Path(str(tax_matrix.with_suffix("")) + ".smatch.jsonl")
            score_manifest = Path(str(score_path) + ".manifest.json")
            value = json.loads(score_manifest.read_text())
            value["input_sha256"] = hashlib.sha256(tax_matrix.read_bytes()).hexdigest()
            value["output_sha256"] = hashlib.sha256(score_path.read_bytes()).hexdigest()
            self.write_json(score_manifest, value)
            report = evaluate_lora_publication_matrix(
                self.config, self.model, root, "tax"
            )
        self.assertTrue(report["passed"])

    def test_stored_pilot_gate_detects_evidence_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_complete(root)
            current = evaluate_lora_method_pilot(self.config, self.model, root)
            stored = json.loads(json.dumps(current))
            stored.update(
                {
                    "checked_at_utc": "2026-08-26T00:00:00+00:00",
                    "config_sha256": "c" * 64,
                    "selection_sha256": "s" * 64,
                }
            )
            verify_frozen_lora_method_pilot(
                current,
                stored,
                config_sha256="c" * 64,
                selection_sha256="s" * 64,
            )
            stored["evidence"]["matrix_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "differs from current evidence"):
                verify_frozen_lora_method_pilot(
                    current,
                    stored,
                    config_sha256="c" * 64,
                    selection_sha256="s" * 64,
                )

    def test_v2_gate_accepts_only_hash_bound_batch1_replay_correction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, _ = self.make_complete(root)
            baseline_path = run / "dev-u-greedy.jsonl"
            baseline = [
                json.loads(line) for line in baseline_path.read_text().splitlines()
            ]
            baseline[0]["raw_prediction"] = "different.n.01"
            self.write_jsonl(baseline_path, baseline)
            current = evaluate_lora_method_pilot(self.config, self.model, root)
            failed = [
                check for check in current["checks"] if not check["passed"]
            ]
            self.assertEqual(["u_surface_equivalence"], [row["name"] for row in failed])

            config_sha = "c" * 64
            selection_sha = "s" * 64
            failed_path = root / "failed-v1.json"
            failed_value = json.loads(json.dumps(current))
            failed_value.update(
                {
                    "checked_at_utc": "2026-08-26T00:00:00+00:00",
                    "config_sha256": config_sha,
                    "selection_sha256": selection_sha,
                }
            )
            self.write_json(failed_path, failed_value)
            replay_path = root / "replay.jsonl"
            replay_path.write_text("replay evidence\n", encoding="utf-8")
            replay_manifest = Path(str(replay_path) + ".manifest.json")
            self.write_json(replay_manifest, {"status": "complete"})
            plan_path = root / "plan.json"
            self.write_json(
                plan_path,
                {
                    "schema_version": 1,
                    "kind": "selected-u-control-replay-plan",
                    "status": "frozen",
                    "observed_batch_sensitive_ids": failed[0]["observed"],
                    "replay": {"performance_threshold_used": False},
                },
            )
            audit_path = root / "audit.json"
            audit = {
                "schema_version": 1,
                "kind": "selected-u-control-replay-audit",
                "status": "complete",
                "passed": True,
                "identity_failures": [],
                "surface_failures": [],
                "performance_threshold_used": False,
                "batch_sensitive_ids": failed[0]["observed"],
                "evidence": {
                    "plan_path": str(plan_path),
                    "plan_sha256": hashlib.sha256(
                        plan_path.read_bytes()
                    ).hexdigest(),
                    "pilot_matrix_sha256": current["evidence"]["matrix_sha256"],
                    "replay_path": str(replay_path),
                    "replay_sha256": hashlib.sha256(
                        replay_path.read_bytes()
                    ).hexdigest(),
                    "replay_manifest_sha256": hashlib.sha256(
                        replay_manifest.read_bytes()
                    ).hexdigest(),
                },
            }
            self.write_json(audit_path, audit)
            stored = {
                "schema_version": 2,
                "kind": "selected-lora-method-pilot-gate",
                "checked_at_utc": "2026-08-26T00:00:00+00:00",
                "passed": True,
                "protocol_correction": {
                    "original_gate_passed": False,
                    "only_original_failure": "u_surface_equivalence",
                    "cause": "legacy batch-8 versus registered batch-1 execution-contract sensitivity",
                    "corrected_control": "independent U-only batch-1 exact replay",
                    "performance_threshold_used": False,
                    "task_metrics_used_for_acceptance": False,
                    "batch_sensitive_ids": failed[0]["observed"],
                },
                "retained_v1_checks": [
                    check
                    for check in current["checks"]
                    if check["name"] != "u_surface_equivalence"
                ],
                "evidence": {
                    **current["evidence"],
                    "config_sha256": config_sha,
                    "selection_sha256": selection_sha,
                    "failed_v1_gate_path": str(failed_path),
                    "failed_v1_gate_sha256": hashlib.sha256(
                        failed_path.read_bytes()
                    ).hexdigest(),
                    "replay_plan_path": str(plan_path),
                    "replay_plan_sha256": audit["evidence"]["plan_sha256"],
                    "replay_audit_path": str(audit_path),
                    "replay_audit_sha256": hashlib.sha256(
                        audit_path.read_bytes()
                    ).hexdigest(),
                    "replay_sha256": audit["evidence"]["replay_sha256"],
                    "replay_manifest_sha256": audit["evidence"][
                        "replay_manifest_sha256"
                    ],
                },
            }
            verify_frozen_lora_method_pilot(
                current,
                stored,
                config_sha256=config_sha,
                selection_sha256=selection_sha,
            )
            replay_path.write_text("drift\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "replay evidence drifted"):
                verify_frozen_lora_method_pilot(
                    current,
                    stored,
                    config_sha256=config_sha,
                    selection_sha256=selection_sha,
                )

    def test_stored_publication_gate_detects_evidence_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, dev_matrix = self.make_complete(root)
            (run / "test-u-greedy.jsonl").write_bytes(
                (run / "dev-u-greedy.jsonl").read_bytes()
            )
            test_matrix = run / "test2-u-bf-wf-greedy.jsonl"
            test_matrix.write_bytes(dev_matrix.read_bytes())
            for suffix in (
                ".jsonl.manifest.json",
                ".smatch.jsonl",
                ".smatch.jsonl.manifest.json",
                ".summary.json",
                ".paired-statistics.json",
            ):
                source = Path(str(dev_matrix.with_suffix("")) + suffix)
                target = Path(str(test_matrix.with_suffix("")) + suffix)
                target.write_bytes(source.read_bytes())
            score_path = Path(str(test_matrix.with_suffix("")) + ".smatch.jsonl")
            score_manifest = Path(str(score_path) + ".manifest.json")
            value = json.loads(score_manifest.read_text())
            value["input_sha256"] = hashlib.sha256(test_matrix.read_bytes()).hexdigest()
            value["output_sha256"] = hashlib.sha256(score_path.read_bytes()).hexdigest()
            self.write_json(score_manifest, value)
            current = evaluate_lora_publication_matrix(
                self.config, self.model, root, "test"
            )
            stored = json.loads(json.dumps(current))
            stored.update(
                {
                    "split": "test",
                    "checked_at_utc": "2026-08-26T00:00:00+00:00",
                    "config_sha256": "c" * 64,
                    "selection_sha256": "s" * 64,
                    "pilot_gate_sha256": "p" * 64,
                    "u_control_plan_sha256": "u" * 64,
                }
            )
            verify_frozen_lora_publication_matrix(
                current,
                stored,
                split="test",
                config_sha256="c" * 64,
                selection_sha256="s" * 64,
                pilot_gate_sha256="p" * 64,
                u_control_plan_sha256="u" * 64,
            )
            stored["evidence"]["matrix_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "differs from current evidence"):
                verify_frozen_lora_publication_matrix(
                    current,
                    stored,
                    split="test",
                    config_sha256="c" * 64,
                    selection_sha256="s" * 64,
                    pilot_gate_sha256="p" * 64,
                    u_control_plan_sha256="u" * 64,
                )


if __name__ == "__main__":
    unittest.main()
