from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY_ROOT / "scripts" / "run_colossus_amr_full_suite.sh"


class AMRFullRunnerTest(unittest.TestCase):
    def test_runner_is_serial_resource_gated_and_heldout_last(self):
        source = RUNNER.read_text(encoding="utf-8")
        for evidence in (
            "check_amr_preflight.py",
            "check_amr_smoke.py",
            "plan_amr_full_run.py",
            "check_amr_full_training.py",
            "check_amr_matrix.py",
            "audit_penman_certified_release.py",
            "analyze_amr_beam.py",
            "summarize_amr_suite.py",
            "build_manuscript_result_package.py",
            "check_lora_suite_v3_promotion.py",
        ):
            self.assertIn(evidence, source)
        self.assertIn(".selected-amr-full-suite.lock", source)
        self.assertIn("automata_gpu_claim", source)
        self.assertIn("OMP_NUM_THREADS=1", source)
        self.assertIn("OMP_NUM_THREADS=2", source)
        self.assertIn("output_scores", (
            REPOSITORY_ROOT / "scripts" / "generate_amr_beam_candidates.py"
        ).read_text(encoding="utf-8"))
        self.assertLess(source.index("pgrep -f"), source.index("automata_gpu_claim"))
        self.assertLess(source.index("pilot_predictions="), source.index("dev_predictions="))
        self.assertLess(source.index("dev_predictions="), source.index("beam_candidates="))
        self.assertLess(source.index("beam_report="), source.index("test_predictions="))
        self.assertIn('"${dev_gate}" "${beam_report}" "${beam_release}"', source)
        self.assertLess(source.index("test_predictions="), source.index("suite_json="))
        self.assertLess(source.index("suite_json="), source.index("paper_json="))
        self.assertIn("recent-model-and-method-results-v3.json", source)
        self.assertIn("recent-model-and-method-results-v3.promotion.json", source)
        self.assertLess(
            source.index("check_lora_suite_v3_promotion.py"),
            source.index("automata_gpu_claim"),
        )
        syntax = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, syntax.returncode, syntax.stderr)


if __name__ == "__main__":
    unittest.main()
