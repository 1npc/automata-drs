from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "build_budget_cliff_figure.py"
SPEC = importlib.util.spec_from_file_location("build_budget_cliff_figure", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BudgetCliffFigureTest(unittest.TestCase):
    def test_tax9_trace_reconstructs_the_stored_certificate(self):
        evidence = MODULE.build_evidence(
            MODULE.DEFAULT_INPUT,
            MODULE.DEFAULT_RECORD_ID,
            max_new_tokens=256,
        )
        divergence = evidence["divergence"]
        self.assertEqual(206, divergence["step"])
        self.assertEqual("D", chr(divergence["unconstrained_byte"]))
        self.assertTrue(divergence["unconstrained_locally_live"])
        self.assertIsNone(divergence["unconstrained_witness_tokens"])
        self.assertEqual("t", chr(divergence["bounded_byte"]))
        self.assertEqual(42, divergence["bounded_witness_tokens"])
        self.assertEqual(8, divergence["bounded_slack_tokens"])

        trace = evidence["trace"]
        self.assertEqual(256, len(trace))
        self.assertEqual(3156, sum(point["witness_tokens"] for point in trace))
        self.assertEqual(29484, sum(point["slack_tokens"] for point in trace))
        self.assertEqual(0, min(point["slack_tokens"] for point in trace))
        self.assertTrue(evidence["unconstrained"]["locally_extendable_at_cap"])
        self.assertFalse(evidence["unconstrained"]["can_end_at_cap"])
        self.assertTrue(evidence["bounded"]["can_end_before_eos"])

    def test_checked_in_trace_and_svg_are_deterministic(self):
        evidence = MODULE.build_evidence(
            MODULE.DEFAULT_INPUT,
            MODULE.DEFAULT_RECORD_ID,
            max_new_tokens=256,
        )
        expected_data = MODULE.DEFAULT_DATA.read_text(encoding="utf-8")
        self.assertEqual(
            expected_data,
            MODULE.json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        )
        self.assertEqual(
            MODULE.DEFAULT_SVG.read_text(encoding="utf-8"),
            MODULE.render_svg(evidence, tail_start=192),
        )


if __name__ == "__main__":
    unittest.main()
