from __future__ import annotations

from pathlib import Path
import unittest


FORMALIZATION = (
    Path(__file__).resolve().parents[1] / "docs" / "formalization.md"
)


class FormalizationContractTest(unittest.TestCase):
    def test_generic_soundness_claim_keeps_coverage_and_semantics_separate(self):
        text = " ".join(
            FORMALIZATION.read_text(encoding="utf-8").split()
        )
        for phrase in (
            "A1 — deterministic fragment composition",
            "A2 — absorbing rejection",
            "A3 — sound finalization",
            "A4 — EOS isolation",
            "A5 — exact token lifting",
            "Theorem 1 (generic accepting-EOS soundness)",
            "Theorem 2 (conditional finite-horizon termination and membership)",
            "Proposition 3 (selective release soundness)",
            "Lemma 4 (exact-byte trace preservation)",
            "not 100% release coverage",
            "does not compute `Live_C` exactly",
            "says nothing about",
        ):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
