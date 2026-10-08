from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "check_amr_full_training.py"
PYTHON = REPOSITORY_ROOT / ".venv" / "bin" / "python"


class AMRFullTrainingGateTest(unittest.TestCase):
    @staticmethod
    def write(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def test_complete_training_is_hash_bound_and_verifiable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            self.write(
                config,
                {
                    "host": {
                        "key": "m",
                        "repo_id": "model",
                        "model_loader": "causal-lm",
                        "revision": "rev",
                    },
                    "data": {
                        "train": {"records": 10, "sha256": "a" * 64},
                        "development": {"records": 2, "sha256": "b" * 64},
                    },
                    "training": {
                        "learning_rate": 1e-4,
                        "warmup_ratio": 0.03,
                        "lora_rank": 16,
                        "lora_alpha": 32,
                        "lora_dropout": 0.05,
                        "seed": 42,
                    },
                    "resource_contract": {"cpu_math_threads": 2},
                },
            )
            registry = root / "registry.json"
            self.write(
                registry,
                {
                    "models": [
                        {
                            "key": "m",
                            "training_target_modules_manifest": ["q_proj"],
                        }
                    ]
                },
            )
            plan = root / "plan.json"
            self.write(
                plan,
                {
                    "audit": "amr-full-run-resource-plan",
                    "inputs": {
                        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()
                    },
                    "recipe": {
                        "max_length": 128,
                        "epochs": 3,
                        "per_device_batch_size": 1,
                        "gradient_accumulation_steps": 16,
                        "effective_batch_size": 16,
                        "planned_optimizer_steps": 3,
                    },
                    "resource_evidence": {"required_gpu_margin_bytes": 4 * 1024**3},
                    "gate": {"passed": True, "performance_threshold_used": False},
                },
            )
            run_root = root / "run"
            self.write(
                run_root / "train_manifest.json",
                {
                    "status": "complete",
                    "task": "amr",
                    "model": "model",
                    "model_loader": "causal-lm",
                    "requested_revision": "rev",
                    "resolved_revision": "rev",
                    "train_source_sha256": "a" * 64,
                    "eval_source_sha256": "b" * 64,
                    "train_record_numbers": None,
                    "train": {"records": 10, "max_sequence_tokens": 100},
                    "eval": {"records": 2, "max_sequence_tokens": 90},
                    "recipe": {
                        "max_length": 128,
                        "epochs": 3,
                        "max_steps": -1,
                        "batch_size": 1,
                        "gradient_accumulation": 16,
                        "effective_batch_size": 16,
                        "planned_optimizer_steps": 3,
                        "learning_rate": 1e-4,
                        "warmup_ratio": 0.03,
                        "lora_rank": 16,
                        "lora_alpha": 32,
                        "lora_dropout": 0.05,
                        "seed": 42,
                        "precision": "bf16",
                        "target_modules": ["q_proj"],
                        "torch_threads": 2,
                        "dataloader_num_workers": 0,
                        "dataset_storage": "compact-uint32-input-ids; masks-and-labels-materialized-per-item",
                    },
                },
            )
            self.write(
                run_root / "train_metrics.json",
                {
                    "train": {"train_loss": 0.2},
                    "eval": {"eval_loss": 0.3},
                    "wall_seconds": 10.0,
                    "cuda_memory": {
                        "peak_allocated_bytes": 20 * 1024**3,
                        "peak_reserved_bytes": 22 * 1024**3,
                        "device_total_bytes": 46 * 1024**3,
                    },
                },
            )
            adapter = run_root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            output = root / "gate.json"
            command = [
                str(PYTHON), str(SCRIPT), str(run_root),
                "--resource-plan", str(plan), "--config", str(config),
                "--registry", str(registry),
            ]
            created = subprocess.run(
                [*command, "--output", str(output)], cwd=REPOSITORY_ROOT,
                text=True, capture_output=True,
            )
            self.assertEqual(0, created.returncode, created.stderr)
            verified = subprocess.run(
                [*command, "--verify-existing", str(output)], cwd=REPOSITORY_ROOT,
                text=True, capture_output=True,
            )
            self.assertEqual(0, verified.returncode, verified.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(11, sum(report["gate"]["checks"].values()))


if __name__ == "__main__":
    unittest.main()
