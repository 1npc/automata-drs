from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from automata_drs.lora_run_state import (
    inspect_lora_baseline,
    verify_complete_lora_baseline_outputs,
)


class LoraRunStateTest(unittest.TestCase):
    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def setUp(self) -> None:
        self.config = {
            "shared_recipe": {"max_length": 1024},
            "models": [
                {
                    "key": "model",
                    "repo_id": "org/model",
                    "revision": "a" * 40,
                    "baseline_generation_contract": "batch8-unscored-v1",
                    "priority": 1,
                    "smoke_records": 1,
                }
            ],
        }

    def make_corpora(self, root: Path) -> None:
        for relative in (
            "data/en/dev/standard.sbn",
            "data/en/test/standard.sbn",
            "data/en/test/tax_challenge.sbn",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Text.\t+\n", encoding="utf-8")

    def audit(self, path: Path) -> None:
        self.write_json(
            path,
            {
                "requested_revision": "a" * 40,
                "target_roundtrip_failure_count": 0,
                "lengths": {"total_max": 100},
            },
        )

    def training(self, run: Path) -> None:
        self.audit(Path(str(run) + ".tokenizer-audit.json"))
        self.write_json(
            run / "train_manifest.json",
            {
                "status": "complete",
                "model": "org/model",
                "requested_revision": "a" * 40,
                "recipe": {"max_steps": -1},
            },
        )
        self.write_json(
            run / "train_metrics.json",
            {"train": {"train_loss": 1.0}, "eval": {"eval_loss": 0.5}},
        )
        self.write_json(run / "adapter" / "adapter_config.json", {"r": 16})
        (run / "adapter" / "adapter_model.safetensors").write_bytes(b"weights")

    def prediction(self, path: Path, count: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name.startswith("test-"):
            corpus = "data/en/test/standard.sbn"
        elif path.name.startswith("tax-"):
            corpus = "data/en/test/tax_challenge.sbn"
        else:
            corpus = "data/en/dev/standard.sbn"
        full_split = "dev20" not in path.name
        with path.open("w", encoding="utf-8") as sink:
            for index in range(count):
                sink.write(
                    json.dumps(
                        {
                            "id": (
                                f"{corpus}:{index + 1}"
                                if full_split
                                else str(index)
                            ),
                            **(
                                {
                                    "input": "Text.",
                                    "reference": "+",
                                    "prediction": "+",
                                    "validity": {
                                        "raw_sbn": True,
                                        "sbn": True,
                                        "clf_referee": True,
                                    },
                                    "generation": {
                                        "hit_max_new_tokens": False,
                                        "canonical_tokenization": True,
                                    },
                                }
                                if full_split
                                else {}
                            ),
                            "configuration": "U",
                            "decode": {"strategy": "greedy", "num_beams": 1},
                        }
                    )
                    + "\n"
                )
        self.write_json(
            Path(str(path) + ".manifest.json"),
            {
                "model": "org/model",
                "requested_revision": "a" * 40,
                "path": corpus,
                "start": 0,
                "limit": -1 if full_split else 20,
                "line_numbers": None,
                "configurations": ["U"],
                "num_beams": 1,
                "batch_size": 8 if full_split else 1,
                "score_model_logprob": False,
                "schedule": "rotate",
            },
        )

    def scores(self, prediction: Path, count: int) -> None:
        prefix = prediction.with_suffix("")
        score_path = Path(str(prefix) + ".smatch.jsonl")
        predictions = [
            json.loads(line) for line in prediction.read_text().splitlines()
        ]
        score_path.write_text(
            "".join(
                json.dumps(
                    {
                        "id": row["id"],
                        "configuration": row["configuration"],
                        "decode": row["decode"],
                        "semantic_scores": {"sbn_smatch": 1.0},
                    }
                )
                + "\n"
                for row in predictions[:count]
            ),
            encoding="utf-8",
        )
        self.write_json(
            Path(str(score_path) + ".manifest.json"),
            {
                "input_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
                "records": count,
                "groups": [
                    {"configuration": "U", "strategy": "greedy", "records": count}
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
                        "total": count,
                        "raw_sbn_valid": count,
                        "referee_scored": count,
                        "referee_valid": count,
                        "exact_serialization": count,
                        "truncated": 0,
                        "canonical_tokenization_scored": count,
                        "canonical_tokenization": count,
                        "sbn_smatch_mean": 1.0,
                    }
                ]
            },
        )
        self.write_json(Path(str(prefix) + ".failures.json"), {})

    def complete_smoke(self, root: Path) -> None:
        run = root / "runs" / "model-smoke"
        self.training(run)
        manifest = self.read_json(run / "train_manifest.json")
        manifest["recipe"]["max_steps"] = 8
        self.write_json(run / "train_manifest.json", manifest)
        rows = []
        for configuration in ("U", "B", "W", "BF", "WF"):
            rows.append(
                {
                    "id": "0",
                    "configuration": configuration,
                    "validity": {
                        "raw_sbn": configuration in {"BF", "WF"},
                        "clf_referee": configuration == "WF",
                    },
                    "generation": {
                        "hit_max_new_tokens": False,
                        "canonical_tokenization": True,
                    },
                    "constraint": {"all_masked_rows": 0}
                    if configuration != "U"
                    else None,
                }
            )
        path = run / "dev20-u-b-w-bf-wf-greedy.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        self.write_json(
            Path(str(path) + ".manifest.json"),
            {"requested_revision": "a" * 40},
        )
        self.write_json(run / "smoke_gate.json", self._computed_smoke(root))

    def _computed_smoke(self, root: Path) -> dict[str, object]:
        from automata_drs.lora_gate import evaluate_lora_smoke_gate

        return evaluate_lora_smoke_gate(
            self.config["models"][0],
            root / "runs" / "model-smoke",
            max_length=1024,
        )

    def inspect(self, root: Path) -> dict[str, object]:
        return inspect_lora_baseline(self.config, "model", root / "runs", root)

    def test_absent_run_starts_with_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_corpora(root)
            report = self.inspect(root)
        self.assertEqual("run_smoke", report["next_action"])

    def test_complete_smoke_without_gate_is_gate_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_corpora(root)
            run = root / "runs" / "model-smoke"
            self.training(run)
            manifest = self.read_json(run / "train_manifest.json")
            manifest["recipe"]["max_steps"] = 8
            self.write_json(run / "train_manifest.json", manifest)
            rows = []
            for configuration in ("U", "B", "W", "BF", "WF"):
                rows.append(
                    {
                        "id": "0",
                        "configuration": configuration,
                        "validity": {
                            "raw_sbn": configuration in {"BF", "WF"},
                            "clf_referee": configuration == "WF",
                        },
                        "generation": {
                            "hit_max_new_tokens": False,
                            "canonical_tokenization": True,
                        },
                        "constraint": {"all_masked_rows": 0}
                        if configuration != "U"
                        else None,
                    }
                )
            path = run / "dev20-u-b-w-bf-wf-greedy.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            self.write_json(
                Path(str(path) + ".manifest.json"),
                {"requested_revision": "a" * 40},
            )
            report = self.inspect(root)
        self.assertEqual("ready_for_gate", report["smoke"]["state"])
        self.assertEqual("write_smoke_gate", report["next_action"])

    @staticmethod
    def read_json(path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    def test_partial_prediction_is_invalid_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_corpora(root)
            run = root / "runs" / "model-r16-seed42"
            self.training(run)
            prediction = run / "dev20-u-sequential-greedy.jsonl"
            prediction.write_text(
                json.dumps(
                    {
                        "id": "0",
                        "configuration": "U",
                        "decode": {"strategy": "greedy", "num_beams": 1},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.write_json(
                Path(str(prediction) + ".manifest.json"),
                {
                    "model": "org/model",
                    "requested_revision": "a" * 40,
                    "configurations": ["U"],
                    "num_beams": 1,
                },
            )
            report = self.inspect(root)
        self.assertEqual("invalid", report["full"]["state"])
        self.assertFalse(report["resumable"])
        self.assertIn("expected 20 predictions", report["full"]["reasons"][0])

    def test_completed_generation_resumes_only_missing_score_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_corpora(root)
            self.complete_smoke(root)
            run = root / "runs" / "model-r16-seed42"
            self.training(run)
            self.prediction(run / "dev20-u-sequential-greedy.jsonl", 20)
            for split in ("dev", "test", "tax"):
                self.prediction(run / f"{split}-u-greedy.jsonl", 1)
            self.scores(run / "dev-u-greedy.jsonl", 1)
            report = self.inspect(root)
        self.assertEqual("scoring_partial", report["full"]["state"])
        self.assertEqual(["test", "tax"], report["full"]["missing_scores"])
        self.assertEqual("finalize_missing_scores", report["next_action"])

    def test_full_split_rows_are_bound_to_the_named_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_corpora(root)
            self.complete_smoke(root)
            run = root / "runs" / "model-r16-seed42"
            self.training(run)
            self.prediction(run / "dev20-u-sequential-greedy.jsonl", 20)
            for split in ("dev", "test", "tax"):
                prediction = run / f"{split}-u-greedy.jsonl"
                self.prediction(prediction, 1)
                self.scores(prediction, 1)
            test_prediction = run / "test-u-greedy.jsonl"
            rows = [
                json.loads(line) for line in test_prediction.read_text().splitlines()
            ]
            rows[0]["input"] = "wrong split"
            test_prediction.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = self.inspect(root)
        self.assertEqual("invalid", report["full"]["state"])
        self.assertIn(
            "differs from data/en/test/standard.sbn",
            report["full"]["reasons"][0],
        )

    def test_score_rows_must_match_prediction_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_corpora(root)
            self.complete_smoke(root)
            run = root / "runs" / "model-r16-seed42"
            self.training(run)
            self.prediction(run / "dev20-u-sequential-greedy.jsonl", 20)
            for split in ("dev", "test", "tax"):
                prediction = run / f"{split}-u-greedy.jsonl"
                self.prediction(prediction, 1)
                self.scores(prediction, 1)
            score_path = run / "tax-u-greedy.smatch.jsonl"
            score_rows = [
                json.loads(line) for line in score_path.read_text().splitlines()
            ]
            score_rows[0]["id"] = "data/en/dev/standard.sbn:1"
            score_path.write_text(
                "".join(json.dumps(row) + "\n" for row in score_rows),
                encoding="utf-8",
            )
            score_manifest = Path(str(score_path) + ".manifest.json")
            value = self.read_json(score_manifest)
            value["output_sha256"] = hashlib.sha256(
                score_path.read_bytes()
            ).hexdigest()
            self.write_json(score_manifest, value)
            report = self.inspect(root)
        self.assertEqual("invalid", report["full"]["state"])
        self.assertIn("score/prediction identity", report["full"]["reasons"][0])

    def test_complete_baseline_verifier_binds_all_splits_and_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_corpora(root)
            run = root / "runs" / "model-r16-seed42"
            for split in ("dev", "test", "tax"):
                prediction = run / f"{split}-u-greedy.jsonl"
                self.prediction(prediction, 1)
                self.scores(prediction, 1)
            report = verify_complete_lora_baseline_outputs(
                self.config, root / "runs", root
            )
            self.assertTrue(report["passed"])
            self.assertEqual(
                {"dev", "test", "tax"},
                set(report["models"]["model"]),
            )

            summary = run / "test-u-greedy.summary.json"
            value = self.read_json(summary)
            value["groups"][0]["sbn_smatch_mean"] = 0.5
            self.write_json(summary, value)
            with self.assertRaisesRegex(
                ValueError, "summary differs from prediction/score rows"
            ):
                verify_complete_lora_baseline_outputs(
                    self.config, root / "runs", root
                )

    def test_complete_baseline_verifier_accepts_registered_legacy_manifest(self):
        self.config["models"][0][
            "baseline_generation_contract"
        ] = "legacy-sequential-unscored-v0"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_corpora(root)
            run = root / "runs" / "model-r16-seed42"
            for split in ("dev", "test", "tax"):
                prediction = run / f"{split}-u-greedy.jsonl"
                self.prediction(prediction, 1)
                manifest_path = Path(str(prediction) + ".manifest.json")
                manifest = self.read_json(manifest_path)
                manifest.pop("batch_size")
                manifest.pop("score_model_logprob")
                self.write_json(manifest_path, manifest)
                self.scores(prediction, 1)
            report = verify_complete_lora_baseline_outputs(
                self.config, root / "runs", root
            )
        self.assertTrue(report["passed"])

    def test_complete_baseline_verifier_honors_split_specific_contracts(self):
        self.config["models"][0]["baseline_generation_split_contracts"] = {
            "dev": "batch8-unscored-v1",
            "test": "legacy-sequential-scored-v1",
            "tax": "legacy-sequential-scored-v1",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_corpora(root)
            run = root / "runs" / "model-r16-seed42"
            for split in ("dev", "test", "tax"):
                prediction = run / f"{split}-u-greedy.jsonl"
                self.prediction(prediction, 1)
                if split != "dev":
                    manifest_path = Path(str(prediction) + ".manifest.json")
                    manifest = self.read_json(manifest_path)
                    manifest["batch_size"] = 1
                    manifest["score_model_logprob"] = True
                    self.write_json(manifest_path, manifest)
                self.scores(prediction, 1)
            report = verify_complete_lora_baseline_outputs(
                self.config, root / "runs", root
            )
        self.assertTrue(report["passed"])


if __name__ == "__main__":
    unittest.main()
