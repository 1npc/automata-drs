from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from automata_drs.intervention_gate import (
    evaluate_lora_intervention,
    verify_frozen_lora_intervention,
)
from automata_drs.interventions import build_intervention_plan
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


class InterventionGateTest(unittest.TestCase):
    criterion = (
        "configuration == U and (raw SBN invalid under B or semantic state "
        "invalid under W)"
    )

    def setUp(self) -> None:
        self.model = {
            "key": "model",
            "repo_id": "org/model",
            "revision": "a" * 40,
        }
        self.config = {
            "method_experiment": {
                "conditioned_intervention": {
                    "splits": ["dev", "test", "tax"],
                    "selection_criterion": self.criterion,
                    "configurations": ["U", "B", "W", "BF", "WF"],
                    "strategy": "greedy",
                    "max_new_tokens": 256,
                    "membership_conditioned_on_u_failure": True,
                    "performance_threshold_used": False,
                },
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
    def row(configuration: str) -> dict[str, object]:
        prediction = (
            "male.n.02 Theme +1" if configuration == "U" else "male.n.02"
        )
        valid = configuration != "U"
        return {
            "id": "data/en/dev/standard.sbn:7",
            "input": "A man.",
            "reference": "male.n.02",
            "prediction": prediction,
            "raw_prediction": prediction,
            "configuration": configuration,
            "decode": {"strategy": "greedy", "num_beams": 1},
            "validity": {
                "raw_sbn": valid,
                "sbn": valid,
                "clf_referee": True if valid else None,
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
                "constraint_ms": 0.0 if configuration == "U" else 0.1,
            },
            "constraint": (
                None if configuration == "U" else {"all_masked_rows": 0}
            ),
        }

    def make_baseline_and_plan(
        self, root: Path, *, selected: bool
    ) -> tuple[Path, Path, Path]:
        run = root / "model-r16-seed42"
        baseline = run / "dev-u-greedy.jsonl"
        baseline_row = self.row("U")
        if not selected:
            baseline_row["prediction"] = "male.n.02"
            baseline_row["raw_prediction"] = "male.n.02"
            baseline_row["validity"] = {
                "raw_sbn": True,
                "sbn": True,
                "clf_referee": True,
            }
        self.write_jsonl(baseline, [baseline_row])
        plan = run / "dev-u-invalid-selection.json"
        plan_value = build_intervention_plan(
            [baseline_row],
            input_path=str(baseline),
            input_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        )
        self.write_json(plan, plan_value)
        return run, baseline, plan

    def make_complete(self, root: Path, *, raw_invalid_wf: bool = False) -> None:
        run, _, _ = self.make_baseline_and_plan(root, selected=True)
        output = run / "dev-u-invalid-u-b-w-bf-wf-greedy.jsonl"
        rows = [self.row(name) for name in ("U", "B", "W", "BF", "WF")]
        if raw_invalid_wf:
            rows[-1]["validity"]["raw_clf_referee"] = False
            rows[-1]["validity"]["clf_referee"] = True
        self.write_jsonl(output, rows)
        self.write_json(
            Path(str(output) + ".manifest.json"),
            {
                "model": "org/model",
                "requested_revision": "a" * 40,
                "line_numbers": [7],
                "max_new_tokens": 256,
                "num_beams": 1,
                "batch_size": 1,
                "configurations": ["U", "B", "W", "BF", "WF"],
                "score_model_logprob": True,
                "schedule": "rotate",
            },
        )
        self.write_sidecars(output, rows)

    def write_sidecars(
        self, output: Path, rows: list[dict[str, object]]
    ) -> None:
        prefix = output.with_suffix("")
        scores = Path(str(prefix) + ".smatch.jsonl")
        self.write_jsonl(
            scores,
            [
                {
                    "id": row["id"],
                    "configuration": row["configuration"],
                    "decode": row["decode"],
                    "semantic_scores": {"sbn_smatch": 1.0},
                }
                for row in rows
            ],
        )
        self.write_json(
            Path(str(scores) + ".manifest.json"),
            {
                "input_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(scores.read_bytes()).hexdigest(),
                "records": 5,
                "groups": [
                    {"configuration": name, "strategy": "greedy", "records": 1}
                    for name in ("B", "BF", "U", "W", "WF")
                ],
            },
        )
        self.write_json(
            Path(str(prefix) + ".summary.json"),
            {
                "groups": [
                    {
                        "configuration": name,
                        "strategy": "greedy",
                        "total": 1,
                        "sbn_smatch_mean": 1.0,
                    }
                    for name in ("U", "B", "W", "BF", "WF")
                ],
                "paired_vs_u": [
                    {
                        "configuration": name,
                        "strategy": "greedy",
                        "paired": 1,
                        "missing_unconstrained": 0,
                    }
                    for name in ("B", "W", "BF", "WF")
                ],
            },
        )
        merged = []
        for value in rows:
            item = dict(value)
            item["semantic_scores"] = {"sbn_smatch": 1.0}
            merged.append(item)
        paired = analyze_paired_experiment(
            merged,
            configurations=("B", "W", "BF", "WF"),
            bootstrap_resamples=10,
            confidence=0.95,
            seed=7,
        )
        paired.update(
            {
                "population_scope": "failure-conditioned-on-u-invalidity",
                "gate": {"passed": True},
                "source": {
                    "predictions_sha256": hashlib.sha256(
                        output.read_bytes()
                    ).hexdigest(),
                    "scores_sha256": hashlib.sha256(
                        scores.read_bytes()
                    ).hexdigest(),
                },
            }
        )
        self.write_json(Path(str(prefix) + ".paired-statistics.json"), paired)

    def make_complete_v2(self, root: Path) -> None:
        run, baseline, selection = self.make_baseline_and_plan(root, selected=True)
        baseline_manifest = Path(str(baseline) + ".manifest.json")
        self.write_json(baseline_manifest, {"status": "complete"})
        supplement = run / "dev-u-invalid-b-w-bf-wf-greedy-v2.jsonl"
        constrained = [self.row(name) for name in ("B", "W", "BF", "WF")]
        self.write_jsonl(supplement, constrained)
        supplement_manifest = Path(str(supplement) + ".manifest.json")
        self.write_json(
            supplement_manifest,
            {
                "status": "complete",
                "output_sha256": hashlib.sha256(
                    supplement.read_bytes()
                ).hexdigest(),
                "model": "org/model",
                "requested_revision": "a" * 40,
                "line_numbers": [7],
                "max_new_tokens": 256,
                "num_beams": 1,
                "batch_size": 1,
                "configurations": ["B", "W", "BF", "WF"],
                "score_model_logprob": True,
                "schedule": "rotate",
            },
        )
        output = run / "dev-u-invalid-u-b-w-bf-wf-greedy-v2.jsonl"
        rows = [self.row(name) for name in ("U", "B", "W", "BF", "WF")]
        self.write_jsonl(output, rows)
        correction = Path("configs/qwen_drs_intervention_correction_v2.json")
        output_manifest = Path(str(output) + ".manifest.json")
        self.write_json(
            output_manifest,
            {
                "status": "complete",
                "kind": "derived-lora-conditioned-intervention-v2",
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "rows": 5,
                "expected_rows": 5,
                "model": "org/model",
                "requested_revision": "a" * 40,
                "line_numbers": [7],
                "max_new_tokens": 256,
                "num_beams": 1,
                "batch_size": 1,
                "configurations": ["U", "B", "W", "BF", "WF"],
                "score_model_logprob": True,
                "schedule": "cell-preserving-derived-v2",
                "sources": {
                    "U": {
                        "path": str(baseline),
                        "sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
                        "manifest_path": str(baseline_manifest),
                        "manifest_sha256": hashlib.sha256(
                            baseline_manifest.read_bytes()
                        ).hexdigest(),
                    },
                    "B/W/BF/WF": {
                        "path": str(supplement),
                        "sha256": hashlib.sha256(
                            supplement.read_bytes()
                        ).hexdigest(),
                        "manifest_path": str(supplement_manifest),
                        "manifest_sha256": hashlib.sha256(
                            supplement_manifest.read_bytes()
                        ).hexdigest(),
                    },
                    "selection": {
                        "path": str(selection),
                        "sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
                    },
                    "correction_config": {
                        "path": str(correction),
                        "sha256": hashlib.sha256(correction.read_bytes()).hexdigest(),
                    },
                },
            },
        )
        self.write_sidecars(output, rows)

    def test_complete_conditioned_matrix_passes_with_explicit_bias_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_complete(root)
            report = evaluate_lora_intervention(
                self.config, self.model, root, "dev"
            )
        self.assertTrue(report["passed"])
        self.assertFalse(
            report["diagnostics"]["population_validity_estimate_allowed"]
        )

    def test_corrected_v2_matrix_binds_cell_sources_and_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_complete_v2(root)
            report = evaluate_lora_intervention(
                self.config,
                self.model,
                root,
                "dev",
                artifact_suffix="-v2",
            )
        self.assertTrue(report["passed"], report["checks"])
        self.assertEqual("-v2", report["diagnostics"]["artifact_suffix"])
        self.assertIn("correction_config_sha256", report["evidence"])

    def test_selected_u_must_exactly_reproduce_frozen_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_complete(root)
            output = (
                root
                / "model-r16-seed42"
                / "dev-u-invalid-u-b-w-bf-wf-greedy.jsonl"
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            rows[0]["raw_prediction"] = "different"
            self.write_jsonl(output, rows)
            report = evaluate_lora_intervention(
                self.config, self.model, root, "dev"
            )
        failed = {item["name"] for item in report["checks"] if not item["passed"]}
        self.assertIn("selected_u_surface_equivalence", failed)
        self.assertIn("official_score_matrix", failed)

    def test_bounded_wf_gate_uses_raw_referee_value(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_complete(root, raw_invalid_wf=True)
            report = evaluate_lora_intervention(
                self.config, self.model, root, "dev"
            )
        failed = {item["name"] for item in report["checks"] if not item["passed"]}
        self.assertIn("bounded_outputs", failed)

    def test_zero_failure_population_passes_without_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_baseline_and_plan(root, selected=False)
            report = evaluate_lora_intervention(
                self.config, self.model, root, "dev"
            )
        self.assertTrue(report["passed"])
        self.assertTrue(report["diagnostics"]["generation_skipped"])

    def test_stored_intervention_gate_detects_evidence_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_complete(root)
            current = evaluate_lora_intervention(
                self.config, self.model, root, "dev"
            )
            stored = json.loads(json.dumps(current))
            stored.update(
                {
                    "checked_at_utc": "2026-08-26T00:00:00+00:00",
                    "config_sha256": "c" * 64,
                    "selection_sha256": "s" * 64,
                }
            )
            verify_frozen_lora_intervention(
                current,
                stored,
                config_sha256="c" * 64,
                selection_sha256="s" * 64,
            )
            stored["evidence"]["output_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "differs from current evidence"):
                verify_frozen_lora_intervention(
                    current,
                    stored,
                    config_sha256="c" * 64,
                    selection_sha256="s" * 64,
                )


if __name__ == "__main__":
    unittest.main()
