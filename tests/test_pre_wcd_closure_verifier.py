from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "verify_pre_wcd_experiment_closure.py"
SPEC = importlib.util.spec_from_file_location("verify_pre_wcd_closure", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PreWCDClosureVerifierTest(unittest.TestCase):
    def test_verifies_gate_and_every_hash_bound_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence.json"
            evidence.write_text('{"passed": true}\n', encoding="utf-8")
            digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
            closure = root / "closure.json"
            closure.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "audit": "pre-wcd-experiment-closure",
                        "status": "closed",
                        "primary_gates_passed": True,
                        "drs": {
                            "primary_gates": [
                                {
                                    "path": "evidence.json",
                                    "sha256": digest,
                                    "passed": True,
                                }
                            ]
                        },
                        "gate": {
                            "passed": True,
                            "all_primary_drs_gates_passed": True,
                            "all_primary_amr_gates_passed": True,
                            "all_amr_alignments_optimal": True,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            closure_digest = hashlib.sha256(closure.read_bytes()).hexdigest()
            previous_root = MODULE.REPOSITORY_ROOT
            MODULE.REPOSITORY_ROOT = root
            try:
                result = MODULE.verify_closure(closure, closure_digest)
                self.assertTrue(result["gate_passed"])
                self.assertEqual(1, result["dependencies_verified"])
                evidence.write_text("changed\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "dependency hash changed"):
                    MODULE.verify_closure(closure, closure_digest)
            finally:
                MODULE.REPOSITORY_ROOT = previous_root


if __name__ == "__main__":
    unittest.main()
