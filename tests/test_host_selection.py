from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from automata_drs.host_selection import (
    select_lora_host,
    verify_frozen_lora_host_selection,
)


class HostSelectionTest(unittest.TestCase):
    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    @staticmethod
    def config() -> dict[str, object]:
        return {
            "shared_recipe": {
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": "all-linear",
                "learning_rate": 0.0001,
                "warmup_ratio": 0.03,
                "epochs": 3,
                "per_device_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "max_length": 1024,
                "precision": "bf16",
                "seed": 42,
            },
            "host_selection": {
                "split": "dev",
                "required_records": 2,
                "require_all_models_complete": True,
                "primary_metric": "sbn_smatch_mean",
                "direction": "max",
                "tie_breakers": [
                    "referee_valid_rate_total",
                    "raw_sbn_valid_rate",
                    "parameters_billions:min",
                    "priority:min",
                ],
                "test_metrics_used": False,
                "referee_validity_surface": "raw_generation",
            },
            "models": [
                {
                    "key": "small",
                    "repo_id": "org/small",
                    "revision": "a" * 40,
                    "parameters_billions": 2.0,
                    "priority": 1,
                    "baseline_generation_contract": "batch8-unscored-v1",
                    "model_loader": "causal-lm",
                    "training_manifest_contract": "model-loader-v1",
                    "training_target_modules_manifest": "all-linear",
                    "tokenizer_audit_contract": "run-root-resolved-revision-v1",
                },
                {
                    "key": "large",
                    "repo_id": "org/large",
                    "revision": "b" * 40,
                    "parameters_billions": 4.0,
                    "priority": 2,
                    "baseline_generation_contract": "batch8-unscored-v1",
                    "model_loader": "causal-lm",
                    "training_manifest_contract": "model-loader-v1",
                    "training_target_modules_manifest": "all-linear",
                    "tokenizer_audit_contract": "run-root-resolved-revision-v1",
                },
            ],
        }

    def make_candidate(
        self,
        root: Path,
        row: dict[str, object],
        *,
        smatch: float,
        referee: float = 1.0,
        raw: float = 1.0,
        raw_referee: float | None = None,
    ) -> None:
        corpus = root / "data/en/dev/standard.sbn"
        if not corpus.exists():
            corpus.parent.mkdir(parents=True, exist_ok=True)
            corpus.write_text(
                "Text zero.\tdog.n.01\nText one.\tcat.n.01\n",
                encoding="utf-8",
            )
        train_corpus = root / "data/en/train/gold.sbn"
        if not train_corpus.exists():
            train_corpus.parent.mkdir(parents=True, exist_ok=True)
            train_corpus.write_text(
                "Train zero.\tdog.n.01\nTrain one.\tcat.n.01\n",
                encoding="utf-8",
            )
        run = root / f"{row['key']}-r16-seed42"
        self.write_json(
            Path(str(run) + ".tokenizer-audit.json"),
            {
                "model": row["repo_id"],
                "requested_revision": row["revision"],
                "resolved_revision": row["revision"],
                "records": 2,
                "target_roundtrip_failure_count": 0,
                "lengths": {"total_max": 100},
            },
        )
        self.write_json(
            run / "adapter" / "adapter_config.json",
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "inference_mode": True,
                "bias": "none",
                "use_dora": False,
                "use_rslora": False,
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "base_model_name_or_path": row["repo_id"],
                "target_modules": ["q_proj", "v_proj"],
            },
        )
        (run / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
        self.write_json(
            run / "train_manifest.json",
            {
                "status": "complete",
                "model": row["repo_id"],
                "requested_revision": row["revision"],
                "resolved_revision": row["revision"],
                "model_loader": row["model_loader"],
                "train_path": "data/en/train/gold.sbn",
                "eval_path": "data/en/dev/standard.sbn",
                "train": {"records": 2},
                "eval": {"records": 2},
                "parameters": {
                    "total": 2_000_000_000,
                    "trainable": 20_000_000,
                },
                "recipe": {
                    "max_length": 1024,
                    "epochs": 3.0,
                    "max_steps": -1,
                    "batch_size": 4,
                    "gradient_accumulation": 4,
                    "effective_batch_size": 16,
                    "learning_rate": 0.0001,
                    "planned_optimizer_steps": 3,
                    "warmup_steps": 1,
                    "lora_rank": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "target_modules": row["training_target_modules_manifest"],
                    "precision": "bf16",
                    "seed": 42,
                },
            },
        )
        self.write_json(run / "train_metrics.json", {"eval": {"eval_loss": 0.1}})
        self.write_json(
            run / "dev-u-greedy.jsonl.manifest.json",
            {
                "model": row["repo_id"],
                "requested_revision": row["revision"],
                "path": "data/en/dev/standard.sbn",
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
        inputs = ("Text zero.", "Text one.")
        references = ("dog.n.01", "cat.n.01")
        with (run / "dev-u-greedy.jsonl").open("w", encoding="utf-8") as sink:
            for index in range(2):
                sink.write(
                    json.dumps(
                        {
                            "id": f"data/en/dev/standard.sbn:{index + 1}",
                            "input": inputs[index],
                            "reference": references[index],
                            "prediction": references[index],
                            "raw_prediction": references[index],
                            "configuration": "U",
                            "decode": {"strategy": "greedy", "num_beams": 1},
                            "validity": {
                                "raw_sbn": index < round(raw * 2),
                                "clf_referee": index < round(referee * 2),
                                **(
                                    {}
                                    if raw_referee is None
                                    else {
                                        "raw_clf_referee": index
                                        < round(raw_referee * 2)
                                    }
                                ),
                            },
                        }
                    )
                    + "\n"
                )
        scores_path = run / "dev-u-greedy.smatch.jsonl"
        scores_path.write_text(
            "".join(
                json.dumps(
                    {
                        "id": f"data/en/dev/standard.sbn:{index + 1}",
                        "configuration": "U",
                        "decode": {"strategy": "greedy", "num_beams": 1},
                        "semantic_scores": {"sbn_smatch": smatch},
                    }
                )
                + "\n"
                for index in range(2)
            ),
            encoding="utf-8",
        )
        prediction_sha = hashlib.sha256(
            (run / "dev-u-greedy.jsonl").read_bytes()
        ).hexdigest()
        score_sha = hashlib.sha256(scores_path.read_bytes()).hexdigest()
        self.write_json(
            run / "dev-u-greedy.smatch.jsonl.manifest.json",
            {
                "input_sha256": prediction_sha,
                "output_sha256": score_sha,
                "records": 2,
                "groups": [
                    {"configuration": "U", "strategy": "greedy", "records": 2}
                ],
            },
        )
        self.write_json(
            run / "dev-u-greedy.summary.json",
            {
                "groups": [
                    {
                        "configuration": "U",
                        "strategy": "greedy",
                        "total": 2,
                        "sbn_smatch_mean": smatch,
                        "referee_valid_rate_total": referee,
                        **(
                            {}
                            if raw_referee is None
                            else {
                                "raw_referee_valid_rate_total": (
                                    min(round(raw * 2), round(raw_referee * 2))
                                    / 2
                                )
                            }
                        ),
                        "raw_sbn_valid_rate": raw,
                        "truncated": 0,
                    }
                ]
            },
        )

    def test_primary_metric_is_development_smatch(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            small, large = config["models"]
            self.make_candidate(root, small, smatch=0.90, referee=1.0)
            self.make_candidate(root, large, smatch=0.91, referee=1.0)
            result = select_lora_host(config, root, repository_root=root)
        self.assertEqual("large", result["selected_key"])
        self.assertEqual(["large", "small"], result["ranking"])

    def test_ties_prefer_referee_then_smaller_model(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            small, large = config["models"]
            self.make_candidate(root, small, smatch=0.90, referee=0.5)
            self.make_candidate(root, large, smatch=0.90, referee=1.0)
            self.assertEqual(
                "large",
                select_lora_host(config, root, repository_root=root)[
                    "selected_key"
                ],
            )
            self.make_candidate(root, small, smatch=0.90, referee=1.0)
            self.assertEqual(
                "small",
                select_lora_host(config, root, repository_root=root)[
                    "selected_key"
                ],
            )

    def test_referee_tie_breaker_uses_raw_generation_before_postprocessing(self):
        config = self.config()
        small, large = config["models"]
        small["parameters_billions"] = 4.0
        large["parameters_billions"] = 2.0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_candidate(
                root,
                small,
                smatch=0.90,
                referee=1.0,
                raw_referee=1.0,
            )
            self.make_candidate(
                root,
                large,
                smatch=0.90,
                referee=1.0,
                raw_referee=0.0,
            )
            result = select_lora_host(config, root, repository_root=root)
        self.assertEqual("small", result["selected_key"])
        candidates = {row["key"]: row for row in result["candidates"]}
        self.assertEqual(
            0.0,
            candidates["large"]["metrics"]["referee_valid_rate_total"],
        )
        self.assertEqual(
            1.0,
            candidates["large"]["metrics"][
                "postprocessed_referee_valid_rate_total"
            ],
        )

    def test_legacy_referee_fallback_rejects_raw_postprocessed_divergence(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in config["models"]:
                self.make_candidate(root, row, smatch=0.9)
            prediction = root / "small-r16-seed42" / "dev-u-greedy.jsonl"
            rows = [
                json.loads(line) for line in prediction.read_text().splitlines()
            ]
            rows[0]["raw_prediction"] = "dog.n.01\ntrailing text"
            prediction.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            score_manifest_path = Path(
                str(root / "small-r16-seed42" / "dev-u-greedy.smatch.jsonl")
                + ".manifest.json"
            )
            score_manifest = json.loads(score_manifest_path.read_text())
            score_manifest["input_sha256"] = hashlib.sha256(
                prediction.read_bytes()
            ).hexdigest()
            self.write_json(score_manifest_path, score_manifest)
            with self.assertRaisesRegex(
                ValueError, "fallback requires identical raw/postprocessed"
            ):
                select_lora_host(config, root, repository_root=root)

    def test_missing_candidate_fails_closed(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_candidate(root, config["models"][0], smatch=0.9)
            with self.assertRaisesRegex(ValueError, "selection is not ready"):
                select_lora_host(config, root, repository_root=root)

    def test_referee_tie_breaker_surface_is_explicit_and_fail_closed(self):
        config = self.config()
        del config["host_selection"]["referee_validity_surface"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in config["models"]:
                self.make_candidate(root, row, smatch=0.9)
            with self.assertRaisesRegex(ValueError, "must use raw generation"):
                select_lora_host(config, root, repository_root=root)

    def test_revision_mismatch_fails_closed(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in config["models"]:
                self.make_candidate(root, row, smatch=0.9)
            manifest = root / "small-r16-seed42" / "train_manifest.json"
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["requested_revision"] = "wrong"
            self.write_json(manifest, value)
            with self.assertRaisesRegex(ValueError, "revision does not match"):
                select_lora_host(config, root, repository_root=root)

    def test_split_specific_development_contract_must_match_selector(self):
        config = self.config()
        config["models"][0]["baseline_generation_split_contracts"] = {
            "dev": "legacy-sequential-unscored-v0",
            "test": "batch8-unscored-v1",
            "tax": "batch8-unscored-v1",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in config["models"]:
                self.make_candidate(root, row, smatch=0.9)
            with self.assertRaisesRegex(
                ValueError, "split-specific contract"
            ):
                select_lora_host(config, root, repository_root=root)

    def test_shared_lora_recipe_and_training_population_are_revalidated(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in config["models"]:
                self.make_candidate(root, row, smatch=0.9)
            manifest = root / "small-r16-seed42" / "train_manifest.json"
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["recipe"]["learning_rate"] = 0.001
            self.write_json(manifest, value)
            with self.assertRaisesRegex(ValueError, "training recipe differs"):
                select_lora_host(config, root, repository_root=root)

            self.make_candidate(root, config["models"][0], smatch=0.9)
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["train"]["records"] = 1
            self.write_json(manifest, value)
            with self.assertRaisesRegex(ValueError, "corpus path/count"):
                select_lora_host(config, root, repository_root=root)

            self.make_candidate(root, config["models"][0], smatch=0.9)
            adapter = root / "small-r16-seed42" / "adapter" / "adapter_config.json"
            value = json.loads(adapter.read_text(encoding="utf-8"))
            value["r"] = 8
            self.write_json(adapter, value)
            with self.assertRaisesRegex(ValueError, "saved adapter differs"):
                select_lora_host(config, root, repository_root=root)

    def test_architecture_specific_language_target_manifest_is_exact(self):
        config = self.config()
        large = config["models"][1]
        large["model_loader"] = "multimodal-text-only"
        large["training_target_modules_manifest"] = {
            "scope": "language-linear",
            "count": 248,
            "prefix": "model.language_model.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in config["models"]:
                self.make_candidate(root, row, smatch=0.9)
            result = select_lora_host(config, root, repository_root=root)
            candidates = {row["key"]: row for row in result["candidates"]}
            self.assertEqual(
                large["training_target_modules_manifest"],
                candidates["large"]["evidence"][
                    "training_target_modules_manifest"
                ],
            )

            manifest = root / "large-r16-seed42" / "train_manifest.json"
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["recipe"]["target_modules"]["count"] = 247
            self.write_json(manifest, value)
            with self.assertRaisesRegex(ValueError, "training recipe differs"):
                select_lora_host(config, root, repository_root=root)

    def test_unregistered_target_module_manifest_fails_closed(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in config["models"]:
                self.make_candidate(root, row, smatch=0.9)
            config["models"][0].pop("training_target_modules_manifest")
            with self.assertRaisesRegex(
                ValueError, "no registered training target-module manifest"
            ):
                select_lora_host(config, root, repository_root=root)

    def test_tokenizer_audit_coverage_and_roundtrip_are_revalidated(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in config["models"]:
                self.make_candidate(root, row, smatch=0.9)
            audit = Path(
                str(root / "small-r16-seed42") + ".tokenizer-audit.json"
            )
            value = json.loads(audit.read_text(encoding="utf-8"))
            value["target_roundtrip_failure_count"] = 1
            self.write_json(audit, value)
            with self.assertRaisesRegex(ValueError, "tokenizer audit"):
                select_lora_host(config, root, repository_root=root)

            self.make_candidate(root, config["models"][0], smatch=0.9)
            value = json.loads(audit.read_text(encoding="utf-8"))
            value["records"] = 1
            self.write_json(audit, value)
            with self.assertRaisesRegex(ValueError, "tokenizer audit"):
                select_lora_host(config, root, repository_root=root)

    def test_dev_rows_and_generation_manifest_are_bound_to_dev_corpus(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in config["models"]:
                self.make_candidate(root, row, smatch=0.9)
            prediction = root / "small-r16-seed42" / "dev-u-greedy.jsonl"
            rows = [json.loads(line) for line in prediction.read_text().splitlines()]
            rows[0]["input"] = "test-set text masquerading as dev"
            prediction.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "differs from data/en/dev"):
                select_lora_host(config, root, repository_root=root)

            self.make_candidate(root, config["models"][0], smatch=0.9)
            manifest = (
                root
                / "small-r16-seed42"
                / "dev-u-greedy.jsonl.manifest.json"
            )
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["path"] = "data/en/test/standard.sbn"
            self.write_json(manifest, value)
            with self.assertRaisesRegex(ValueError, "generation manifest"):
                select_lora_host(config, root, repository_root=root)

    def test_development_summary_is_recomputed_from_score_rows(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in config["models"]:
                self.make_candidate(root, row, smatch=0.9)
            summary = root / "small-r16-seed42" / "dev-u-greedy.summary.json"
            value = json.loads(summary.read_text(encoding="utf-8"))
            value["groups"][0]["sbn_smatch_mean"] = 0.99
            self.write_json(summary, value)
            with self.assertRaisesRegex(ValueError, "summary differs"):
                select_lora_host(config, root, repository_root=root)

    def test_registered_legacy_sequential_manifest_is_narrowly_accepted(self):
        config = self.config()
        config["models"][0][
            "baseline_generation_contract"
        ] = "legacy-sequential-unscored-v0"
        config["models"][0][
            "training_manifest_contract"
        ] = "legacy-no-model-loader-v0"
        config["models"][0][
            "tokenizer_audit_contract"
        ] = "legacy-train-name-no-resolved-revision-v0"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in config["models"]:
                self.make_candidate(root, row, smatch=0.9)
            run = root / "small-r16-seed42"
            manifest_path = run / "dev-u-greedy.jsonl.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("batch_size")
            manifest.pop("score_model_logprob")
            self.write_json(manifest_path, manifest)
            train_manifest_path = run / "train_manifest.json"
            train_manifest = json.loads(
                train_manifest_path.read_text(encoding="utf-8")
            )
            train_manifest.pop("model_loader")
            self.write_json(train_manifest_path, train_manifest)
            modern_audit = Path(str(run) + ".tokenizer-audit.json")
            legacy_audit = root / "small-train-tokenizer-audit.json"
            audit = json.loads(modern_audit.read_text(encoding="utf-8"))
            audit["resolved_revision"] = None
            self.write_json(legacy_audit, audit)
            modern_audit.unlink()
            result = select_lora_host(config, root, repository_root=root)
        self.assertEqual("small", result["selected_key"])
        self.assertEqual(
            "legacy-sequential-unscored-v0",
            result["candidates"][0]["evidence"][
                "baseline_generation_contract"
            ],
        )

    def test_frozen_selection_revalidates_every_evidence_hash(self):
        config = self.config()
        config_bytes = json.dumps(config, sort_keys=True).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, row in enumerate(config["models"]):
                self.make_candidate(root, row, smatch=0.91 - index * 0.01)
            selection = select_lora_host(config, root, repository_root=root)
            selection.update(
                {
                    "selected_at_utc": "2026-08-26T00:00:00+00:00",
                    "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                }
            )
            self.assertEqual(
                "small",
                verify_frozen_lora_host_selection(
                    config,
                    config_bytes,
                    selection,
                    root,
                    repository_root=root,
                ),
            )
            (root / "small-r16-seed42" / "adapter" / "adapter_model.safetensors").write_bytes(
                b"tampered"
            )
            with self.assertRaisesRegex(ValueError, "differs from current evidence"):
                verify_frozen_lora_host_selection(
                    config,
                    config_bytes,
                    selection,
                    root,
                    repository_root=root,
                )

    def test_frozen_selection_rejects_changed_config(self):
        config = self.config()
        config_bytes = json.dumps(config, sort_keys=True).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in config["models"]:
                self.make_candidate(root, row, smatch=0.9)
            selection = select_lora_host(config, root, repository_root=root)
            selection.update(
                {
                    "selected_at_utc": "2026-08-26T00:00:00+00:00",
                    "config_sha256": "0" * 64,
                }
            )
            with self.assertRaisesRegex(ValueError, "config hash"):
                verify_frozen_lora_host_selection(
                    config,
                    config_bytes,
                    selection,
                    root,
                    repository_root=root,
                )


if __name__ == "__main__":
    unittest.main()
