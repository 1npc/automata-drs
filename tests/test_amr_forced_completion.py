from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "audit_amr_forced_completion.py"


class AMRForcedCompletionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("amr_forced_completion", SCRIPT)
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_closes_a_live_truncated_penman_prefix(self):
        completed, suffix = self.module.complete_penman_prefix(
            "(a / alpha :ARG0 (b / beta"
        )
        self.assertEqual("))", suffix)
        self.assertEqual("(a / alpha :ARG0 (b / beta))", completed)

    def test_refuses_a_dead_prefix(self):
        completed, suffix = self.module.complete_penman_prefix("not penman")
        self.assertIsNone(completed)
        self.assertIsNone(suffix)

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
