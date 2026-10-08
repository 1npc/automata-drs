from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from automata_drs.beam_selection import combine_beam_selection_rows
from automata_drs.lora_beam_gate import (
    evaluate_lora_beam_selection,
    verify_frozen_lora_beam_selection,
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


class LoraBeamGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = {
            "key": "model",
            "repo_id": "org/model",
            "revision": "a" * 40,
            "model_loader": "causal-lm",
        }
        self.study = {
            "frozen_after_granite_results_before_qwen_task_outputs": (
                "2026-08-26T03:58:30+02:00"
            ),
            "motivation_source": "primary sources",
            "split": "dev",
            "selection": "same-first-corpus-order-as-pilot",
            "records": 1,
            "num_beams": 4,
            "max_new_tokens": 256,
            "generated_conditions": ["U4-candidates", "W", "WF"],
            "derived_comparison": ["U", "PostW", "W", "WF"],
            "u_num_return_sequences": 4,
            "online_num_return_sequences": 1,
            "postw_selection": "highest score valid",
            "strategy": "beam",
            "performance_threshold_used": False,
            "report": ["validity"],
            "note": "diagnostic",
        }
        self.config = {
            "method_experiment": {
                "pilot": {"records": 1, "max_new_tokens": 256},
                "beam_selection_study": self.study,
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
    def row(configuration: str, *, rank: int | None = None) -> dict[str, object]:
        row: dict[str, object] = {
            "id": "data/en/dev/standard.sbn:1",
            "input": "A man.",
            "reference": "male.n.02",
            "prediction": "male.n.02",
            "raw_prediction": "male.n.02",
            "configuration": configuration,
            "decode": {"strategy": "beam", "num_beams": 4},
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
                "generation_ms": 2.0,
                "model_score_ms": 0.5,
                "constraint_ms": 0.0,
            },
            "constraint": (
                None if configuration == "U4" else {"all_masked_rows": 0}
            ),
        }
        if rank is not None:
            row["decode"] = {
                "strategy": "beam",
                "num_beams": 4,
                "num_return_sequences": 4,
            }
            row["beam_candidate"] = {
                "rank": rank,
                "sequence_score": -float(rank),
                "validation_ms": 0.1,
            }
        return row

    def make_complete(self, root: Path) -> Path:
        run = root / "model-r16-seed42"
        baseline = run / "dev-u-greedy.jsonl"
        baseline_row = self.row("U")
        baseline_row["decode"] = {"strategy": "greedy", "num_beams": 1}
        self.write_jsonl(baseline, [baseline_row])

        candidates = run / "dev1-u4-beam-candidates.jsonl"
        candidate_rows = [self.row("U4", rank=rank) for rank in range(4)]
        self.write_jsonl(candidates, candidate_rows)
        self.write_json(
            Path(str(candidates) + ".manifest.json"),
            {
                "status": "complete",
                "kind": "unconstrained-beam-candidates",
                "model": "org/model",
                "model_loader": "causal-lm",
                "requested_revision": "a" * 40,
                "output_sha256": hashlib.sha256(candidates.read_bytes()).hexdigest(),
                "records": 1,
                "rows": 4,
                "num_beams": 4,
            },
        )

        online = run / "dev1-w-wf-beam4.jsonl"
        online_rows = [self.row("W"), self.row("WF")]
        self.write_jsonl(online, online_rows)
        self.write_json(
            Path(str(online) + ".manifest.json"),
            {
                "model": "org/model",
                "model_loader": "causal-lm",
                "requested_revision": "a" * 40,
                "configurations": ["W", "WF"],
                "num_beams": 4,
                "limit": 1,
                "max_new_tokens": 256,
            },
        )

        combined_rows = combine_beam_selection_rows(
            candidate_rows,
            online_rows,
            ["data/en/dev/standard.sbn:1"],
            num_beams=4,
        )
        matrix = run / "dev1-u-postw-w-wf-beam4.jsonl"
        self.write_jsonl(matrix, combined_rows)
        self.write_json(
            Path(str(matrix) + ".manifest.json"),
            {
                "status": "complete",
                "kind": "derived-beam-selection-matrix",
                "candidate_sha256": hashlib.sha256(
                    candidates.read_bytes()
                ).hexdigest(),
                "online_sha256": hashlib.sha256(online.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
                "study_contract": self.study,
                "records": 1,
                "rows": 4,
                "configurations": ["U", "PostW", "W", "WF"],
                "num_beams": 4,
            },
        )

        prefix = matrix.with_suffix("")
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
                "input_sha256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(scores.read_bytes()).hexdigest(),
                "records": 4,
            },
        )
        self.write_json(
            Path(str(prefix) + ".summary.json"),
            {
                "records": 4,
                "groups": [
                    {
                        "configuration": name,
                        "strategy": "beam4",
                        "total": 1,
                    }
                    for name in ("U", "PostW", "W", "WF")
                ],
            },
        )
        merged = []
        for row in combined_rows:
            value = dict(row)
            value["semantic_scores"] = {"sbn_smatch": 1.0}
            merged.append(value)
        paired = analyze_paired_experiment(
            merged,
            configurations=("PostW", "W", "WF"),
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
                        matrix.read_bytes()
                    ).hexdigest(),
                    "scores_sha256": hashlib.sha256(scores.read_bytes()).hexdigest(),
                },
            }
        )
        self.write_json(Path(str(prefix) + ".paired-statistics.json"), paired)
        return matrix

    def test_complete_beam_selection_study_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_complete(root)
            report = evaluate_lora_beam_selection(self.config, self.model, root)
        self.assertTrue(report["passed"])
        self.assertEqual(0, report["diagnostics"]["postw_fallbacks"])

    def test_tampered_derived_matrix_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = self.make_complete(root)
            rows = [json.loads(line) for line in matrix.read_text().splitlines()]
            rows[0]["prediction"] = "tampered"
            self.write_jsonl(matrix, rows)
            report = evaluate_lora_beam_selection(self.config, self.model, root)
        failed = {item["name"] for item in report["checks"] if not item["passed"]}
        self.assertIn("exact_derived_matrix", failed)
        self.assertIn("derived_manifest", failed)
        self.assertIn("official_scores", failed)

    def test_stored_beam_gate_detects_evidence_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_complete(root)
            current = evaluate_lora_beam_selection(self.config, self.model, root)
            stored = json.loads(json.dumps(current))
            stored.update(
                {
                    "checked_at_utc": "2026-08-26T00:00:00+00:00",
                    "config_sha256": "c" * 64,
                    "selection_sha256": "s" * 64,
                    "publication_gate_sha256": "p" * 64,
                }
            )
            verify_frozen_lora_beam_selection(
                current,
                stored,
                config_sha256="c" * 64,
                selection_sha256="s" * 64,
                publication_gate_sha256="p" * 64,
            )
            stored["diagnostics"]["postw_fallbacks"] = 1
            with self.assertRaisesRegex(ValueError, "differs from current evidence"):
                verify_frozen_lora_beam_selection(
                    current,
                    stored,
                    config_sha256="c" * 64,
                    selection_sha256="s" * 64,
                    publication_gate_sha256="p" * 64,
                )


if __name__ == "__main__":
    unittest.main()
