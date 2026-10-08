from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
DRS = ROOT / "scripts" / "generate_causal_lm_baseline.py"
AMR = ROOT / "scripts" / "generate_causal_lm_amr_pf.py"
CONFIG = ROOT / "configs" / "wcd_experiment_v1.json"


class WCDGenerationScriptsTest(unittest.TestCase):
    def test_drs_and_amr_use_the_shared_wcd_adapters(self):
        drs = DRS.read_text(encoding="utf-8")
        amr = AMR.read_text(encoding="utf-8")
        self.assertIn("WitnessCarryingSubwordSBNLogitsProcessor", drs)
        self.assertIn('"BWCD", "WWCD"', drs)
        self.assertIn("ByteLevelWitnessCarryingLogitsProcessor", amr)
        self.assertIn('choices=("PF", "WCD")', amr)
        self.assertIn('default="PF"', amr)

    def test_registration_is_small_cross_representation_and_hash_bound(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual("WCD", config["method"]["name"])
        self.assertEqual(["BWCD", "WWCD"], config["drs"]["conditions"])
        self.assertEqual("WCD", config["amr"]["condition"])
        predecessor = ROOT / config["frozen_predecessors"]["amr_config"]
        self.assertEqual(
            hashlib.sha256(predecessor.read_bytes()).hexdigest(),
            config["frozen_predecessors"]["amr_config_sha256"],
        )
        self.assertEqual(1, config["resources"]["gpus_per_experiment"])
        self.assertLessEqual(config["resources"]["cpu_math_threads"], 2)

    def test_scripts_and_registration_parse(self):
        for script in (DRS, AMR):
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(script)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
