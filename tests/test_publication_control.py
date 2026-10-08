from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from automata_drs.publication_control import (
    KIND,
    planned_outputs,
    verify_publication_control_plan,
)


class PublicationControlTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[dict, Path, Path]:
        config = {
            "method_experiment": {
                "publication_matrices": [
                    {"split": "dev", "records": 3},
                    {"split": "test", "records": 3},
                    {"split": "tax", "records": 2},
                ]
            }
        }
        pilot = root / "model-r16-seed42" / "dev3-method-pilot-gate.json"
        pilot.parent.mkdir(parents=True)
        pilot.write_text("{}\n", encoding="utf-8")
        plan = root / "model-r16-seed42" / "publication-u-control-plan.json"
        value = {
            "schema_version": 1,
            "kind": KIND,
            "status": "frozen",
            "created_at_utc": "2026-08-26T00:00:00+00:00",
            "selected_key": "model",
            "control_contract": {
                "host_selection_u_role": "quality-only legacy batch-8 screen",
                "method_u_role": "within-matrix batch-1 greedy control",
                "method_u_logits_processor": None,
                "cross_contract_surface_identity_required": False,
                "pilot_batch1_exact_replay_required": True,
                "paired_population_required": True,
                "task_metric_threshold_used": False,
            },
            "planned_outputs": planned_outputs(config, "model", root),
            "outputs_absent_when_frozen": True,
            "evidence": {
                "config_sha256": "c" * 64,
                "selection_sha256": "s" * 64,
                "pilot_gate_path": str(pilot),
                "pilot_gate_sha256": hashlib.sha256(
                    pilot.read_bytes()
                ).hexdigest(),
            },
        }
        plan.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return config, pilot, plan

    def test_hash_bound_within_matrix_control_plan_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, pilot, plan = self.fixture(root)
            value = verify_publication_control_plan(
                plan,
                config=config,
                config_sha256="c" * 64,
                selection_sha256="s" * 64,
                model_key="model",
                runs_root=root,
                pilot_gate_path=pilot,
            )
            self.assertFalse(
                value["control_contract"][
                    "cross_contract_surface_identity_required"
                ]
            )
            self.assertEqual(3, len(value["planned_outputs"]))

    def test_control_or_pilot_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, pilot, plan = self.fixture(root)
            value = json.loads(plan.read_text(encoding="utf-8"))
            value["control_contract"]["task_metric_threshold_used"] = True
            plan.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contract changed"):
                verify_publication_control_plan(
                    plan,
                    config=config,
                    config_sha256="c" * 64,
                    selection_sha256="s" * 64,
                    model_key="model",
                    runs_root=root,
                    pilot_gate_path=pilot,
                )

