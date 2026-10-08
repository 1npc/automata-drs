from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "plan_amr_full_run.py"
PYTHON = REPOSITORY_ROOT / ".venv" / "bin" / "python"


class AMRFullRunPlanTest(unittest.TestCase):
    @staticmethod
    def write(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def test_promotes_only_the_measured_smoke_batch_and_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            self.write(
                config,
                {
                    "status": "prepared-not-launched",
                    "data": {"train": {"records": 55_635}},
                    "training": {
                        "epochs": 3,
                        "per_device_batch_size": 4,
                        "gradient_accumulation_steps": 4,
                    },
                    "resource_contract": {"cpu_math_threads": 2},
                },
            )
            config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
            preflight = root / "preflight.json"
            self.write(
                preflight,
                {
                    "inputs": {"config_sha256": config_sha},
                    "recommendation": {
                        "max_length": 2048,
                        "max_input_tokens": 256,
                        "max_new_tokens": 1536,
                    },
                    "gate": {"passed": True},
                },
            )
            run_root = root / "smoke"
            manifest = run_root / "train_manifest.json"
            metrics = run_root / "train_metrics.json"
            self.write(
                manifest,
                {
                    "status": "complete",
                    "recipe": {
                        "max_length": 2048,
                        "max_steps": 8,
                        "batch_size": 1,
                        "gradient_accumulation": 16,
                        "torch_threads": 2,
                        "dataloader_num_workers": 0,
                    },
                },
            )
            self.write(
                metrics,
                {
                    "wall_seconds": 80.0,
                    "cuda_memory": {
                        "peak_reserved_bytes": 30 * 1024**3,
                        "device_total_bytes": 46 * 1024**3,
                    },
                },
            )
            smoke = root / "smoke.json"
            self.write(
                smoke,
                {
                    "inputs": {
                        "preflight_gate_sha256": hashlib.sha256(
                            preflight.read_bytes()
                        ).hexdigest(),
                        "train_manifest_sha256": hashlib.sha256(
                            manifest.read_bytes()
                        ).hexdigest(),
                        "train_metrics_sha256": hashlib.sha256(
                            metrics.read_bytes()
                        ).hexdigest(),
                    },
                    "gate": {"passed": True},
                },
            )
            output = root / "plan.json"
            command = [
                str(PYTHON),
                str(SCRIPT),
                "--config",
                str(config),
                "--preflight-gate",
                str(preflight),
                "--smoke-gate",
                str(smoke),
                "--smoke-run-root",
                str(run_root),
            ]
            created = subprocess.run(
                [*command, "--output", str(output)],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, created.returncode, created.stderr)
            verified = subprocess.run(
                [*command, "--verify-existing", str(output)],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, verified.returncode, verified.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(1, report["recipe"]["per_device_batch_size"])
        self.assertEqual(16, report["recipe"]["gradient_accumulation_steps"])
        self.assertEqual(10_434, report["recipe"]["planned_optimizer_steps"])
        self.assertTrue(
            report["resource_evidence"]["wall_time_estimate_is_scheduling_only"]
        )


if __name__ == "__main__":
    unittest.main()
