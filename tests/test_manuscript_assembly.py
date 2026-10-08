from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "assemble_manuscript.py"
MANUSCRIPT = REPOSITORY_ROOT / "docs" / "manuscript.md"
SPEC = importlib.util.spec_from_file_location("assemble_manuscript", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ManuscriptAssemblyTest(unittest.TestCase):
    def test_checked_in_manuscript_is_exact_deterministic_assembly(self):
        expected = MODULE.assemble(REPOSITORY_ROOT)
        self.assertEqual(expected, MANUSCRIPT.read_text(encoding="utf-8"))

    def test_assembly_is_reviewable_and_keeps_claim_boundaries(self):
        text = MANUSCRIPT.read_text(encoding="utf-8")
        headings = [
            "## Abstract",
            "## 1 Introduction",
            "## 2 Related Work",
            "## 3 Method",
            "## 4 Experiments",
            "## 5 Conclusion",
        ]
        positions = [text.index(heading) for heading in headings]
        self.assertEqual(sorted(positions), positions)
        self.assertEqual(1, sum(line.startswith("# ") for line in text.splitlines()))
        self.assertGreaterEqual(text.count("RESULT PENDING"), 34)
        for forbidden in (
            "<!-- Role:",
            "<!-- Paragraph role:",
            "Pre-writing answers",
            "mini-outline",
            "reverse outline",
            "Claim–evidence map",
            "self-review",
        ):
            self.assertNotIn(forbidden.lower(), text.lower())
        for required in (
            "selective, policy-relative contract",
            "Negative decisions are incomplete",
            "Certified validity and availability are reported separately",
            "rather than deterministic search or an error-free semantic parser",
            "Figure 1. Native-token reachability-constrained decoding",
        ):
            self.assertIn(required, " ".join(text.split()))


if __name__ == "__main__":
    unittest.main()
