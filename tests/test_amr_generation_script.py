from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

import penman


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "generate_causal_lm_amr.py"


class AMRGenerationScriptTest(unittest.TestCase):
    def test_independent_validator_distinguishes_raw_prose(self):
        spec = importlib.util.spec_from_file_location("amr_generator_test", SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual((True, None), module._independent_penman_valid("(a / alpha)", penman))
        valid, reason = module._independent_penman_valid("Here is an AMR", penman)
        self.assertFalse(valid)
        self.assertIsInstance(reason, str)

    def test_generator_has_only_registered_first_stage_conditions(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('choices=("U", "P")', source)
        self.assertIn("ByteLevelConstraintLogitsProcessor", source)
        self.assertIn("PENMANConstraintState.initial()", source)
        self.assertNotIn("SubwordSBNLogitsProcessor", source)
        self.assertNotIn('"PF"', source)
        self.assertNotIn("truncation=True", source)
        self.assertIn("exceeding max-input-tokens", source)
        self.assertIn('default="frozen-plain"', source)
        self.assertIn("args.adapter is not None", source)
        self.assertIn("viamr-qwen3-graph-prefill requires", source)

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
