from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY_ROOT / "scripts/run_colossus_amr_pf_feasibility.sh"


class AMRPFFeasibilityRunnerTest(unittest.TestCase):
    def test_runner_is_serial_cpu_only_bounded_and_post_smoke(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn(".amr-pf-feasibility.lock", source)
        self.assertIn("nice -n 15", source)
        self.assertIn("OMP_NUM_THREADS=1", source)
        self.assertIn("TOKENIZERS_PARALLELISM=false", source)
        self.assertIn("--max-cache-entries 20000", source)
        self.assertIn("automata_scratch_mktemp amr-pf", source)
        self.assertIn("automata_scratch_cleanup", source)
        self.assertIn("qwen35-4b-base-r16-seed42-smoke", source)
        self.assertIn('--tokenizer-source "${tokenizer_source}"', source)
        self.assertIn("scripts/check_amr_preflight.py", source)
        self.assertIn("scripts/check_amr_smoke.py", source)
        self.assertIn("scripts/plan_amr_full_run.py", source)
        self.assertIn("--require-zero-false-rejections", source)
        self.assertIn("--expect-corpus-sha256", source)
        self.assertIn("--expect-config-sha256", source)
        self.assertIn("--contract-file", source)
        self.assertIn("--verify-existing", source)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=", source)
        self.assertNotIn("nvidia-smi", source)
        self.assertNotIn("rm -rf", source)
        config_path = REPOSITORY_ROOT / "configs/amr_transfer_experiment.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertIn(hashlib.sha256(config_path.read_bytes()).hexdigest(), source)
        self.assertEqual(
            config["data"]["development"]["sha256"],
            "8088d7f8a53f66c6164ea6a7ceb6421f2650b451adb26a6fe661e911f7764ed6",
        )

    def test_runner_has_valid_bash_syntax(self):
        completed = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual("", completed.stderr)
        self.assertEqual(0, completed.returncode)


if __name__ == "__main__":
    unittest.main()
