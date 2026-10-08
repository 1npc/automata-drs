from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "generate_causal_lm_amr_pf.py"
FROZEN_GENERATOR = REPOSITORY_ROOT / "scripts" / "generate_causal_lm_amr.py"


class AMRPFGenerationScriptTest(unittest.TestCase):
    def test_pf_has_a_separate_post_primary_entrypoint(self):
        source = SCRIPT.read_text(encoding="utf-8")
        frozen = FROZEN_GENERATOR.read_text(encoding="utf-8")
        self.assertIn("ByteLevelFiniteHorizonConstraintLogitsProcessor", source)
        self.assertIn('choices=("PF", "WCD")', source)
        self.assertIn('default="PF"', source)
        self.assertIn('"configurations": [args.method]', source)
        self.assertIn("construct_penman_completion_suffix", source)
        self.assertNotIn('choices=("U", "P", "PF")', frozen)

    def test_script_syntax(self):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(SCRIPT)],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
