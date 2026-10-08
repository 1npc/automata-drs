from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_drs_horizon_planner_repair.py"
CONFIG = ROOT / "configs/wcd_drs_horizon_planner_repair_v1.json"
RUNNER = ROOT / "scripts/run_colossus_drs_horizon_planner_repair_v1.sh"
SPEC = importlib.util.spec_from_file_location(
    "audit_drs_horizon_planner_repair", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DRSPlannerRepairAuditTest(unittest.TestCase):
    def test_contract_freezes_development_scope_and_full_population(self):
        contract = MODULE.load_contract(CONFIG)
        self.assertEqual("registered-development-audit", contract["status"])
        self.assertEqual(128, contract["source"]["records"])
        self.assertEqual(48, contract["source"]["max_new_tokens"])
        self.assertFalse(contract["repair"]["trusted_core_changed"])
        self.assertFalse(contract["repair"]["state_transition_changed"])
        self.assertFalse(contract["repair"]["release_policy_changed"])
        self.assertFalse(contract["repair"]["new_semantic_rules_added"])
        self.assertFalse(contract["audit"]["performance_conditioned_selection"])
        self.assertFalse(contract["resource_contract"]["model_weights_loaded"])
        self.assertFalse(contract["resource_contract"]["gpu_used"])
        self.assertEqual(1, contract["resource_contract"]["cpu_math_threads"])

        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn(hashlib.sha256(CONFIG.read_bytes()).hexdigest(), source)
        self.assertIn("OMP_NUM_THREADS=1", source)
        self.assertIn("TOKENIZERS_PARALLELISM=false", source)
        self.assertIn("nice -n 15", source)
        self.assertIn("ulimit -f 1024", source)
        self.assertIn("--verify-existing", source)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", source)
        self.assertNotIn("nvidia-smi", source)
        self.assertNotIn("rm -rf", source)
        completed = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual("", completed.stderr)
        self.assertEqual(0, completed.returncode)

    def test_summary_requires_both_known_rows_to_be_repaired(self):
        known = {"a", "b"}
        outcomes = [
            {
                "id": identifier,
                "auditable": True,
                "source_path_replay_accepting": True,
                "planner_accepts_entire_path": True,
                "false_negative_events": [],
                "other_rejection_events": [],
            }
            for identifier in ("a", "b", "c")
        ]
        summary = MODULE.summarize_outcomes(outcomes, known)
        self.assertEqual(3, summary["planner_accepts_entire_path_rows"])
        self.assertEqual(["a", "b"], summary["known_false_negative_rows_repaired"])
        self.assertEqual(0, summary["constructive_planner_false_negative_events"])


if __name__ == "__main__":
    unittest.main()
