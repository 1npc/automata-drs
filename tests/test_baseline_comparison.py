from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from automata_drs.baseline_comparison import (
    analyze_lora_baseline_comparisons,
    summarize_constraint_headroom,
)


class BaselineComparisonTest(unittest.TestCase):
    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    @staticmethod
    def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def make_split(
        self,
        root: Path,
        *,
        key: str,
        repo_id: str,
        split: str,
        corpus_label: str,
        valid_count: int,
        smatch: float,
    ) -> None:
        run = root / "runs" / f"{key}-r16-seed42"
        prediction = run / f"{split}-u-greedy.jsonl"
        predictions = []
        scores = []
        for index in range(4):
            valid = index < valid_count
            target = f"concept{index}.n.01"
            surface = target if valid else "bad"
            row = {
                "id": f"{corpus_label}:{index + 1}",
                "input": f"Text {index}.",
                "reference": target,
                "prediction": surface,
                "configuration": "U",
                "decode": {"strategy": "greedy", "num_beams": 1},
                "validity": {
                    "raw_sbn": valid,
                    "sbn": valid,
                    "clf_referee": valid,
                },
                "generation": {
                    "hit_max_new_tokens": False,
                    "canonical_tokenization": True,
                },
            }
            predictions.append(row)
            scores.append(
                {
                    "id": row["id"],
                    "configuration": "U",
                    "decode": row["decode"],
                    "semantic_scores": {"sbn_smatch": smatch},
                }
            )
        self.write_jsonl(prediction, predictions)
        self.write_json(
            Path(str(prediction) + ".manifest.json"),
            {
                "model": repo_id,
                "requested_revision": "a" * 40,
                "path": corpus_label,
                "start": 0,
                "limit": -1,
                "line_numbers": None,
                "configurations": ["U"],
                "num_beams": 1,
                "batch_size": 8,
                "score_model_logprob": False,
                "schedule": "rotate",
            },
        )
        prefix = prediction.with_suffix("")
        score_path = Path(str(prefix) + ".smatch.jsonl")
        self.write_jsonl(score_path, scores)
        self.write_json(
            Path(str(score_path) + ".manifest.json"),
            {
                "input_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
                "records": 4,
                "groups": [
                    {"configuration": "U", "strategy": "greedy", "records": 4}
                ],
            },
        )
        Path(str(prefix) + ".summary.md").write_text("summary\n", encoding="utf-8")
        self.write_json(
            Path(str(prefix) + ".summary.json"),
            {
                "groups": [
                    {
                        "configuration": "U",
                        "strategy": "greedy",
                        "total": 4,
                        "raw_sbn_valid": valid_count,
                        "referee_scored": 4,
                        "referee_valid": valid_count,
                        "exact_serialization": valid_count,
                        "truncated": 0,
                        "canonical_tokenization_scored": 4,
                        "canonical_tokenization": 4,
                        "sbn_smatch_mean": smatch,
                    }
                ]
            },
        )
        self.write_json(Path(str(prefix) + ".failures.json"), {})

    def test_all_model_pairs_use_shared_rows_and_do_not_select(self):
        config = {
            "models": [
                {
                    "key": "left",
                    "repo_id": "org/left",
                    "revision": "a" * 40,
                    "priority": 1,
                    "baseline_generation_contract": "batch8-unscored-v1",
                },
                {
                    "key": "right",
                    "repo_id": "org/right",
                    "revision": "a" * 40,
                    "priority": 2,
                    "baseline_generation_contract": "batch8-unscored-v1",
                },
            ]
        }
        labels = {
            "dev": "data/en/dev/standard.sbn",
            "test": "data/en/test/standard.sbn",
            "tax": "data/en/test/tax_challenge.sbn",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in labels.values():
                corpus = root / label
                corpus.parent.mkdir(parents=True, exist_ok=True)
                corpus.write_text(
                    "".join(
                        f"Text {index}.\tconcept{index}.n.01\n"
                        for index in range(4)
                    ),
                    encoding="utf-8",
                )
            for split, label in labels.items():
                self.make_split(
                    root,
                    key="left",
                    repo_id="org/left",
                    split=split,
                    corpus_label=label,
                    valid_count=1,
                    smatch=0.5,
                )
                self.make_split(
                    root,
                    key="right",
                    repo_id="org/right",
                    split=split,
                    corpus_label=label,
                    valid_count=3,
                    smatch=0.75,
                )
            first = analyze_lora_baseline_comparisons(
                config,
                root / "runs",
                root,
                bootstrap_resamples=100,
                seed=7,
            )
            second = analyze_lora_baseline_comparisons(
                config,
                root / "runs",
                root,
                bootstrap_resamples=100,
                seed=7,
            )
        self.assertEqual(first, second)
        self.assertEqual(2, first["schema_version"])
        self.assertFalse(first["selection_use"])
        self.assertEqual(
            ["S", "R", "G", "B", "W"],
            first["parameters"]["constraint_replay"]["ladder"],
        )
        self.assertEqual(3, len(first["comparisons"]))
        self.assertEqual(6, len(first["constraint_headroom"]))
        dev = first["comparisons"][0]
        self.assertEqual(("left", "right"), (dev["left"], dev["right"]))
        self.assertEqual(
            0.25,
            dev["metrics"]["sbn_smatch"]["delta_right_minus_left"],
        )
        validity = dev["metrics"]["wellformed_valid"]
        self.assertEqual(2, validity["right_fixed"])
        self.assertEqual(0, validity["right_regressed"])
        self.assertEqual(
            validity["mcnemar_exact_p"],
            validity["mcnemar_exact_p_holm"],
        )
        overlap = dev["constraint_headroom"]["w_validity_overlap"]
        self.assertEqual(
            {
                "both_valid": 1,
                "left_only_valid": 0,
                "right_only_valid": 2,
                "both_invalid": 1,
            },
            overlap,
        )
        left_dev = first["constraint_headroom"][0]
        self.assertEqual(("left", "dev", 4), (
            left_dev["model"], left_dev["split"], left_dev["records"]
        ))
        self.assertEqual(1, left_dev["acceptance"]["W"]["accepted"])
        self.assertEqual(3, left_dev["first_blocking_layer"]["S"])
        self.assertEqual(4, left_dev["surface"]["prediction_fallback"])
        self.assertEqual(0, left_dev["surface"]["raw_postprocessed_different"])

    def test_constraint_headroom_locates_each_incremental_layer_on_raw_surface(self):
        surfaces = (
            "bad",
            "person.n.01 Agent -2",
            "person.n.01 Theme +1 entity.n.01 Theme -1",
            "think.v.01 Proposition >1",
            "person.n.01 CONTINUATION <0 person.n.01",
            "person.n.01",
        )
        rows = []
        for index, raw_prediction in enumerate(surfaces):
            b_valid = index >= 4
            w_valid = index == 5
            rows.append(
                {
                    "id": f"dev:{index + 1}",
                    # Deliberately different: replay must use the raw surface.
                    "prediction": "person.n.01",
                    "raw_prediction": raw_prediction,
                    "validity": {
                        "raw_sbn": b_valid,
                        "raw_clf_referee": w_valid,
                    },
                    "generation": {
                        "hit_max_new_tokens": index == 0,
                        "canonical_tokenization": True,
                    },
                }
            )
        report = summarize_constraint_headroom(rows, model="m", split="dev")
        self.assertEqual(
            {"S": 5, "R": 4, "G": 3, "B": 2, "W": 1},
            {
                configuration: report["acceptance"][configuration]["accepted"]
                for configuration in ("S", "R", "G", "B", "W")
            },
        )
        self.assertEqual(
            {"S": 1, "R": 1, "G": 1, "B": 1, "W": 1, "accepted": 1},
            report["first_blocking_layer"],
        )
        self.assertEqual(6, report["surface"]["explicit_raw_prediction"])
        self.assertEqual(5, report["surface"]["raw_postprocessed_different"])
        self.assertEqual(
            {
                "hit_max_new_tokens": 1,
                "and_b_invalid": 1,
                "and_w_invalid": 1,
            },
            report["truncation"],
        )
        self.assertEqual(
            5,
            sum(report["w_failure_family"].values()),
        )
        self.assertEqual(5, sum(report["w_failure_event"].values()))

    def test_constraint_headroom_fails_on_stale_stored_validity(self):
        row = {
            "id": "dev:1",
            "prediction": "person.n.01",
            "validity": {"raw_sbn": False, "raw_clf_referee": False},
            "generation": {
                "hit_max_new_tokens": False,
                "canonical_tokenization": True,
            },
        }
        with self.assertRaisesRegex(ValueError, "stored/raw B validity mismatch"):
            summarize_constraint_headroom([row], model="m", split="dev")

    def test_constraint_headroom_rejects_unsafe_legacy_referee_fallback(self):
        row = {
            "id": "dev:1",
            "prediction": "person.n.01",
            "raw_prediction": "person.n.01\ntrailing text",
            "validity": {"raw_sbn": False, "clf_referee": True},
            "generation": {
                "hit_max_new_tokens": False,
                "canonical_tokenization": True,
            },
        }
        with self.assertRaisesRegex(
            ValueError, "fallback requires identical raw/postprocessed"
        ):
            summarize_constraint_headroom([row], model="m", split="dev")


if __name__ == "__main__":
    unittest.main()
