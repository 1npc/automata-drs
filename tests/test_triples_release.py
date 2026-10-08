from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

from automata_drs import (
    CANONICAL_TRIPLE_RELEASE_GUARANTEE,
    CanonicalTripleBatchValidationResult,
    CertifiedReleaseError,
    FinalSurfaceReleasePolicy,
    assess_canonical_triple_certified_release,
    assess_final_state_certified_release,
    release_canonical_triple_surface,
)


class CanonicalTripleCertifiedReleaseTest(unittest.TestCase):
    @staticmethod
    def row() -> dict[str, object]:
        return {
            "configuration": "KG",
            "raw_prediction": "NODE a Person\nNODE b Book\nEDGE a likes b",
            "validity": {"raw_canonical_triples": True},
            "generation": {
                "ended_with_eos": True,
                "hit_max_new_tokens": False,
            },
            "constraint": {
                "calls": 10,
                "rows": 10,
                "allowed_min": 1,
                "all_masked_rows": 0,
            },
        }

    def test_releases_only_after_fresh_membership_and_execution_checks(self):
        decision = assess_canonical_triple_certified_release(self.row())
        self.assertTrue(decision.releasable, decision.reasons)
        self.assertEqual(CANONICAL_TRIPLE_RELEASE_GUARANTEE, decision.guarantee)
        released = release_canonical_triple_surface(self.row())
        self.assertEqual(self.row()["raw_prediction"], released.text)
        self.assertEqual("KG", released.configuration)

    def test_refuses_invalid_stale_or_incomplete_evidence(self):
        invalid = deepcopy(self.row())
        invalid["raw_prediction"] = "NODE b Book\nNODE a Person"
        self.assertIn(
            "raw-surface-rejected-by-canonical-triple-policy",
            assess_canonical_triple_certified_release(invalid).reasons,
        )

        stale = deepcopy(self.row())
        stale["validity"]["raw_canonical_triples"] = False
        self.assertIn(
            "stored-canonical-triple-validity-is-not-true",
            assess_canonical_triple_certified_release(stale).reasons,
        )

        unfinished = deepcopy(self.row())
        unfinished["generation"]["ended_with_eos"] = False
        with self.assertRaises(CertifiedReleaseError):
            release_canonical_triple_surface(unfinished)

        masked = deepcopy(self.row())
        masked["constraint"]["all_masked_rows"] = 1
        masked["constraint"]["allowed_min"] = 0
        self.assertFalse(
            assess_canonical_triple_certified_release(masked).releasable
        )

    def test_generic_release_gate_fails_closed_when_validator_raises(self):
        def broken_validator(_: str) -> bool:
            raise RuntimeError("validator unavailable")

        policy = FinalSurfaceReleasePolicy(
            configuration="KG",
            guarantee=CANONICAL_TRIPLE_RELEASE_GUARANTEE,
            validate_surface=broken_validator,
            surface_rejection_reason="invalid-surface",
            validity_requirements=(("raw_canonical_triples", "stale-validity"),),
        )
        decision = assess_final_state_certified_release(
            self.row(), policy=policy
        )
        self.assertFalse(decision.releasable)
        self.assertEqual(
            ("surface-validator-raised-an-exception",), decision.reasons
        )

    def test_release_requires_the_independent_batch_oracle(self):
        disagreement = CanonicalTripleBatchValidationResult(
            False, reason="independent rejection"
        )
        with patch(
            "automata_drs.release.validate_canonical_triples_batch",
            return_value=disagreement,
        ):
            decision = assess_canonical_triple_certified_release(self.row())
        self.assertFalse(decision.releasable)
        self.assertIn(
            "raw-surface-rejected-by-canonical-triple-policy",
            decision.reasons,
        )


if __name__ == "__main__":
    unittest.main()
