from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from automata_drs.method_resource import build_method_resource_plan


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class MethodResourcePlanTest(unittest.TestCase):
    def config(self) -> dict[str, object]:
        return {
            "models": [
                {
                    "key": "model",
                    "repo_id": "org/model",
                    "safetensors_bytes": 9_000_000_000,
                }
            ],
            "method_experiment": {
                "execution_schedule": [
                    "pilot-batch-equivalence",
                    "pilot-dev",
                    "publication-dev",
                    "layer-supplement",
                    "fine-supplement",
                    "conditioned-dev-interventions-worst-case",
                    "beam-candidates-and-online",
                    "heldout-test-publication",
                    "heldout-tax-publication",
                    "conditioned-test-interventions-worst-case",
                    "conditioned-tax-interventions-worst-case",
                ],
                "pilot": {
                    "records": 400,
                    "configurations": ["U", "BF", "WF"],
                    "max_new_tokens": 256,
                },
                "layer_ablation": {
                    "records": 400,
                    "generated_configurations": ["B", "W"],
                    "max_new_tokens": 256,
                },
                "fine_grained_ablation": {
                    "records": 400,
                    "generated_configurations": ["S", "R", "G"],
                    "max_new_tokens": 256,
                },
                "beam_selection_study": {
                    "records": 400,
                    "generated_conditions": ["U4-candidates", "W", "WF"],
                    "u_num_return_sequences": 4,
                    "max_new_tokens": 256,
                },
                "conditioned_intervention": {
                    "splits": ["dev", "test", "tax"],
                    "configurations": ["U", "B", "W", "BF", "WF"],
                    "max_new_tokens": 256,
                },
                "publication_matrices": [
                    {"split": "dev", "records": 1132, "configurations": ["U", "BF", "WF"]},
                    {"split": "test", "records": 1132, "configurations": ["U", "BF", "WF"]},
                    {"split": "tax", "records": 117, "configurations": ["U", "BF", "WF"]},
                ],
            },
        }

    def test_current_design_has_a_finite_conservative_ceiling(self):
        plan = build_method_resource_plan(
            self.config(),
            {"status": "selected", "selected_key": "model"},
            config_sha256="c" * 64,
            selection_sha256="s" * 64,
        )
        self.assertTrue(plan["gate"]["passed"])
        self.assertEqual(24_668, plan["upper_bounds"]["generated_surfaces"])
        self.assertEqual(
            6_315_008, plan["upper_bounds"]["theoretical_output_tokens"]
        )
        self.assertEqual(1, plan["runtime_limits"]["concurrent_model_processes"])
        self.assertTrue(plan["runtime_limits"]["private_scratch_required"])
        stages = [row["stage"] for row in plan["stages"]]
        heldout = stages.index("heldout-test-publication")
        self.assertLess(stages.index("fine-supplement"), heldout)
        self.assertLess(stages.index("beam-candidates-and-online"), heldout)
        self.assertLess(
            stages.index("conditioned-dev-interventions-worst-case"),
            heldout,
        )

    def test_accidental_population_expansion_fails_before_a_run(self):
        config = self.config()
        config["method_experiment"]["publication_matrices"][0]["records"] = 5000
        with self.assertRaisesRegex(ValueError, "resource ceiling"):
            build_method_resource_plan(
                config,
                {"status": "selected", "selected_key": "model"},
                config_sha256="c" * 64,
                selection_sha256="s" * 64,
            )

    def test_accidental_execution_reordering_fails_before_a_run(self):
        config = self.config()
        schedule = config["method_experiment"]["execution_schedule"]
        schedule[6], schedule[7] = schedule[7], schedule[6]
        with self.assertRaisesRegex(ValueError, "execution schedule"):
            build_method_resource_plan(
                config,
                {"status": "selected", "selected_key": "model"},
                config_sha256="c" * 64,
                selection_sha256="s" * 64,
            )

    def test_cli_writes_exclusively_and_verifies_hash_bound_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            selection = root / "selection.json"
            output = root / "plan.json"
            config.write_text(json.dumps(self.config()) + "\n", encoding="utf-8")
            selection.write_text(
                json.dumps({"status": "selected", "selected_key": "model"}) + "\n",
                encoding="utf-8",
            )
            base = [
                sys.executable,
                "scripts/plan_lora_method_resources.py",
                "--config",
                str(config),
                "--selection",
                str(selection),
            ]
            first = subprocess.run(
                [*base, "--output", str(output)],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, first.returncode, first.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                hashlib.sha256(config.read_bytes()).hexdigest(),
                report["evidence"]["config_sha256"],
            )
            verify = subprocess.run(
                [*base, "--verify-existing", str(output)],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, verify.returncode, verify.stderr)
            second = subprocess.run(
                [*base, "--output", str(output)],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(0, second.returncode)


if __name__ == "__main__":
    unittest.main()
