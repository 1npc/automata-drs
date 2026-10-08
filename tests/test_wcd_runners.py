from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNERS = (
    ROOT / "scripts" / "run_colossus_external_amr_wcd4.sh",
    ROOT / "scripts" / "run_colossus_drs_wcd_smoke.sh",
    ROOT / "scripts" / "run_colossus_drs_wcd_full_dev.sh",
)


class WCDRunnerTest(unittest.TestCase):
    def test_runners_are_small_locked_and_resource_bounded(self):
        sources = [path.read_text(encoding="utf-8") for path in RUNNERS]
        for source in sources:
            self.assertIn("automata_gpu_claim", source)
            self.assertIn("OMP_NUM_THREADS=2", source)
            self.assertIn("OPENBLAS_NUM_THREADS=2", source)
            self.assertIn("TOKENIZERS_PARALLELISM=false", source)
            self.assertIn("check_wcd_smoke.py", source)
            self.assertIn("refusing overwrite", source)
            self.assertIn("pgrep -f", source)
            self.assertIn("flock -n 9", source)
            self.assertNotIn("rm -rf", source)
        self.assertIn("--limit 4", sources[0])
        self.assertIn("--method WCD", sources[0])
        self.assertIn("smatchpp-v1.json", sources[0])
        self.assertIn("--score-report", sources[0])
        self.assertIn("--limit 8", sources[1])
        self.assertIn("--configurations BWCD WWCD", sources[1])
        self.assertIn("HF_HUB_OFFLINE=1", sources[1])
        self.assertNotIn("automata_scratch_mktemp", sources[1])
        self.assertIn("--limit 1132", sources[2])
        self.assertIn("--sbn-scores", sources[2])
        self.assertIn("wcd_drs_natural_dev_v1.json", sources[2])
        self.assertIn("HF_HUB_OFFLINE=1", sources[2])

    def test_runners_have_valid_bash_syntax(self):
        for runner in RUNNERS:
            result = subprocess.run(
                ["bash", "-n", str(runner)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
