from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from automata_drs.lora_fine_gate import (
    evaluate_lora_fine_ablation,
    verify_frozen_lora_fine_ablation,
)
from automata_drs.lora_layer_ablation import combine_fine_grained_ablation_rows
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


class LoraFineGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = {
            "key": "model",
            "repo_id": "org/model",
            "revision": "a" * 40,
        }
        self.config = {
            "method_experiment": {
                "pilot": {"records": 1},
                "layer_ablation": {"records": 1},
                "fine_grained_ablation": {
                    "split": "dev",
                    "selection": "same-first-corpus-order-as-pilot",
                    "records": 1,
                    "generated_configurations": ["S", "R", "G"],
                    "source_ladder": ["U", "B", "W", "BF", "WF"],
                    "derived_full_ladder": [
                        "U", "S", "R", "G", "B", "W", "BF", "WF"
                    ],
                    "strategy": "greedy",
                    "max_new_tokens": 256,
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
        return {
            "id": "1",
            "input": "A man.",
            "reference": "male.n.02",
            "prediction": "male.n.02",
            "raw_prediction": "male.n.02",
            "configuration": configuration,
            "decode": {"strategy": "greedy", "num_beams": 1},
            "validity": {"raw_sbn": True, "sbn": True, "clf_referee": True},
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
            "constraint": None if configuration == "U" else {"all_masked_rows": 0},
        }

    def make_complete(
        self, root: Path, *, truncated_s: bool = False
    ) -> Path:
        run = root / "model-r16-seed42"
        layer = run / "dev1-u-b-w-bf-wf-greedy.jsonl"
        supplement = run / "dev1-s-r-g-greedy.jsonl"
        combined = run / "dev1-u-s-r-g-b-w-bf-wf-greedy.jsonl"
        layer_rows = [self.row(name) for name in ("U", "B", "W", "BF", "WF")]
        supplement_rows = [self.row(name) for name in ("S", "R", "G")]
        if truncated_s:
            supplement_rows[0]["generation"] = {
                "hit_max_new_tokens": True,
                "ended_with_eos": False,
                "decoder_steps": 256,
            }
            supplement_rows[0]["raw_prediction"] = "unfinished"
            supplement_rows[0]["prediction"] = "unfinished"
            supplement_rows[0]["validity"] = {
                "raw_sbn": False,
                "sbn": False,
                "clf_referee": None,
            }
        self.write_jsonl(layer, layer_rows)
        self.write_json(Path(str(layer) + ".manifest.json"), {"status": "complete"})
        self.write_jsonl(supplement, supplement_rows)
        self.write_json(
            Path(str(supplement) + ".manifest.json"),
            {
                "status": "complete",
                "model": "org/model",
                "requested_revision": "a" * 40,
                "limit": 1,
                "max_new_tokens": 256,
                "num_beams": 1,
                "batch_size": 1,
                "configurations": ["S", "R", "G"],
                "score_model_logprob": True,
                "schedule": "rotate",
            },
        )
        combined_rows, evidence = combine_fine_grained_ablation_rows(
            layer_rows, supplement_rows, expected_records=1
        )
        self.write_jsonl(combined, combined_rows)
        self.write_json(
            Path(str(combined) + ".manifest.json"),
            {
                "schema_version": 1,
                "status": "complete",
                "kind": "derived-lora-fine-grained-ablation",
                "output_sha256": hashlib.sha256(combined.read_bytes()).hexdigest(),
                "sources": [
                    {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "manifest_sha256": hashlib.sha256(
                            Path(str(path) + ".manifest.json").read_bytes()
                        ).hexdigest(),
                    }
                    for path in (layer, supplement)
                ],
                **evidence,
            },
        )
        prefix = combined.with_suffix("")
        scores = Path(str(prefix) + ".smatch.jsonl")
        score_rows = [
            {
                "id": row["id"],
                "configuration": row["configuration"],
                "decode": row["decode"],
                "semantic_scores": {"sbn_smatch": 1.0},
            }
            for row in combined_rows
        ]
        self.write_jsonl(scores, score_rows)
        self.write_json(
            Path(str(scores) + ".manifest.json"),
            {
                "input_sha256": hashlib.sha256(combined.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(scores.read_bytes()).hexdigest(),
                "records": 8,
                "groups": [
                    {"configuration": name, "strategy": "greedy", "records": 1}
                    for name in ("B", "BF", "G", "R", "S", "U", "W", "WF")
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
                    for name in ("U", "S", "R", "G", "B", "W", "BF", "WF")
                ],
                "paired_vs_u": [
                    {
                        "configuration": name,
                        "strategy": "greedy",
                        "paired": 1,
                        "missing_unconstrained": 0,
                    }
                    for name in ("S", "R", "G", "B", "W", "BF", "WF")
                ],
            },
        )
        merged = []
        for value in combined_rows:
            item = dict(value)
            item["semantic_scores"] = {"sbn_smatch": 1.0}
            merged.append(item)
        paired = analyze_paired_experiment(
            merged,
            configurations=("S", "R", "G", "B", "W", "BF", "WF"),
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
                        combined.read_bytes()
                    ).hexdigest(),
                    "scores_sha256": hashlib.sha256(scores.read_bytes()).hexdigest(),
                },
            }
        )
        self.write_json(Path(str(prefix) + ".paired-statistics.json"), paired)
        return combined

    def test_complete_fine_grained_ablation_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_complete(root)
            report = evaluate_lora_fine_ablation(self.config, self.model, root)
        self.assertTrue(report["passed"])

    def test_unbounded_s_truncation_is_reported_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_complete(root, truncated_s=True)
            report = evaluate_lora_fine_ablation(self.config, self.model, root)
        self.assertTrue(report["passed"])
        self.assertEqual(1, report["diagnostics"]["truncated_by_configuration"]["S"])

    def test_tampered_fine_join_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = self.make_complete(root)
            rows = [json.loads(line) for line in combined.read_text().splitlines()]
            rows[0]["prediction"] = "tampered"
            self.write_jsonl(combined, rows)
            report = evaluate_lora_fine_ablation(self.config, self.model, root)
        failed = {item["name"] for item in report["checks"] if not item["passed"]}
        self.assertIn("cell_preserving_fine_join", failed)
        self.assertIn("derived_manifest", failed)

    def test_stored_fine_gate_detects_evidence_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_complete(root)
            current = evaluate_lora_fine_ablation(self.config, self.model, root)
            stored = json.loads(json.dumps(current))
            stored.update(
                {
                    "checked_at_utc": "2026-08-26T00:00:00+00:00",
                    "config_sha256": "c" * 64,
                    "selection_sha256": "s" * 64,
                    "pilot_gate_sha256": "p" * 64,
                    "layer_gate_sha256": "l" * 64,
                }
            )
            verify_frozen_lora_fine_ablation(
                current,
                stored,
                config_sha256="c" * 64,
                selection_sha256="s" * 64,
                pilot_gate_sha256="p" * 64,
                layer_gate_sha256="l" * 64,
            )
            stored["evidence"]["combined_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "differs from current evidence"):
                verify_frozen_lora_fine_ablation(
                    current,
                    stored,
                    config_sha256="c" * 64,
                    selection_sha256="s" * 64,
                    pilot_gate_sha256="p" * 64,
                    layer_gate_sha256="l" * 64,
                )


if __name__ == "__main__":
    unittest.main()
