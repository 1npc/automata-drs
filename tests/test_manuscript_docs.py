from __future__ import annotations

from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INTRODUCTION = REPOSITORY_ROOT / "docs" / "manuscript-scaffold.md"
METHOD = REPOSITORY_ROOT / "docs" / "manuscript-method.md"
EXPERIMENTS = REPOSITORY_ROOT / "docs" / "manuscript-experiments.md"
RELATED_WORK = REPOSITORY_ROOT / "docs" / "manuscript-related-work.md"
ABSTRACT = REPOSITORY_ROOT / "docs" / "manuscript-abstract.md"
CONCLUSION = REPOSITORY_ROOT / "docs" / "manuscript-conclusion.md"
MANUSCRIPT = REPOSITORY_ROOT / "docs" / "manuscript.md"
PAPER_REVIEW = REPOSITORY_ROOT / "docs" / "paper-review-audit.md"
README = REPOSITORY_ROOT / "README.md"


class ManuscriptDocsTest(unittest.TestCase):
    def test_drafts_have_no_hidden_control_characters(self):
        for path in (
            INTRODUCTION,
            METHOD,
            EXPERIMENTS,
            RELATED_WORK,
            ABSTRACT,
            CONCLUSION,
            MANUSCRIPT,
            PAPER_REVIEW,
        ):
            payload = path.read_bytes()
            unexpected = [
                (index, value)
                for index, value in enumerate(payload)
                if value < 32 and value not in {9, 10, 13}
            ]
            self.assertEqual([], unexpected, str(path))

    def test_claim_boundaries_and_navigation_remain_explicit(self):
        introduction = INTRODUCTION.read_text(encoding="utf-8")
        method = METHOD.read_text(encoding="utf-8")
        experiments = EXPERIMENTS.read_text(encoding="utf-8")
        related_work = RELATED_WORK.read_text(encoding="utf-8")
        abstract = ABSTRACT.read_text(encoding="utf-8")
        conclusion = CONCLUSION.read_text(encoding="utf-8")
        paper_review = PAPER_REVIEW.read_text(encoding="utf-8")
        method_words = " ".join(method.split())
        experiment_words = " ".join(experiments.split())
        readme = README.read_text(encoding="utf-8")
        self.assertGreaterEqual(introduction.count("RESULT PENDING"), 2)
        self.assertIn("Forbidden claim", introduction)
        for phrase in (
            "S/R/G/B/W",
            "zero-all-masked",
            "Negative decisions are incomplete",
            "does not inherit the BF/WF availability guarantee",
        ):
            self.assertIn(phrase, method_words)
        self.assertGreaterEqual(experiments.count("RESULT PENDING"), 20)
        for phrase in (
            "Certified validity and availability are reported separately",
            "An abstention preserves the selective structural guarantee but lowers coverage",
            "invalid raw graph contributes zero matching and predicted triples",
            "not guaranteed availability for every attempt",
        ):
            self.assertIn(phrase, experiment_words)
        self.assertIn("docs/manuscript-scaffold.md", readme)
        self.assertIn("docs/manuscript-method.md", readme)
        self.assertIn("docs/manuscript-experiments.md", readme)
        self.assertIn("docs/manuscript-related-work.md", readme)
        self.assertIn("docs/manuscript-abstract.md", readme)
        self.assertIn("docs/manuscript-conclusion.md", readme)
        self.assertIn("docs/manuscript.md", readme)
        for phrase in (
            "hard finite-token reachability is not new",
            "failure to construct one is not a proof of unreachability",
            "they neither implement globally conditioned sampling nor claim unbiasedness",
            "online graph-aware DRS constraints are not themselves novel",
            "RESULT PENDING",
        ):
            self.assertIn(phrase, " ".join(related_work.split()))
        self.assertGreaterEqual(abstract.count("RESULT PENDING"), 8)
        for forbidden in ("deterministic decoding", "error-free applications"):
            self.assertNotIn(forbidden, abstract.lower())
        for phrase in (
            "selective, policy-relative contract",
            "every released output satisfies the registered structural policy",
            "without claiming that failed certificate search proves unreachability",
        ):
            self.assertIn(phrase, " ".join(abstract.split()))
        self.assertGreaterEqual(conclusion.count("RESULT PENDING"), 4)
        for phrase in (
            "rather than deterministic search or an error-free semantic parser",
            "does not make a structurally valid DRS the correct interpretation",
            "failure of the hand-built planner does not prove",
            "without a finite-horizon PF planner",
        ):
            self.assertIn(phrase, " ".join(conclusion.split()))
        review_words = " ".join(paper_review.split())
        for phrase in (
            "complete local regression now passes 442/442 tests",
            "seeded/deduplicated semantic scoring",
            "registered AMR task suite are not complete",
            "return a `CertifiedSurface` or raise",
            "task execution pending",
        ):
            self.assertIn(phrase, review_words)
        self.assertNotIn("no final manuscript exists", review_words)
        self.assertNotIn("No AMR generation experiment", review_words)


if __name__ == "__main__":
    unittest.main()
