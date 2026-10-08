from __future__ import annotations

from copy import deepcopy
import unittest

from automata_drs.release import (
    CertifiedReleaseError,
    PENMAN_RELEASE_GUARANTEE,
    assess_penman_certified_release,
    has_complete_selective_release_accounting,
    release_penman_surface,
)


class PENMANCertifiedReleaseTest(unittest.TestCase):
    @staticmethod
    def row() -> dict[str, object]:
        return {
            "configuration": "P",
            "raw_prediction": "(a / alpha :ARG0 (b / beta))",
            "validity": {
                "raw_local_penman": True,
                "raw_official_penman": True,
            },
            "generation": {
                "ended_with_eos": True,
                "hit_max_new_tokens": False,
            },
            "constraint": {
                "calls": 10,
                "rows": 10,
                "allowed_min": 1,
                "allowed_total": 10,
                "all_masked_rows": 0,
            },
        }

    def test_releases_freshly_revalidated_online_penman_output(self):
        decision = assess_penman_certified_release(
            self.row(), independent_penman_valid=True
        )
        self.assertTrue(decision.releasable, decision.reasons)
        self.assertEqual(
            "final-local-and-independent-penman-membership",
            decision.guarantee,
        )
        released = release_penman_surface(
            self.row(), independent_penman_valid=True
        )
        self.assertEqual(self.row()["raw_prediction"], released.text)
        self.assertEqual(decision.guarantee, released.guarantee)

    def test_refuses_unconstrained_or_invalid_surface(self):
        unconstrained = self.row()
        unconstrained["configuration"] = "U"
        self.assertFalse(
            assess_penman_certified_release(
                unconstrained, independent_penman_valid=True
            ).releasable
        )
        with self.assertRaises(CertifiedReleaseError):
            release_penman_surface(
                unconstrained, independent_penman_valid=True
            )
        invalid = self.row()
        invalid["raw_prediction"] = "(a / alpha"
        self.assertIn(
            "raw-surface-rejected-by-local-penman-policy",
            assess_penman_certified_release(
                invalid, independent_penman_valid=False
            ).reasons,
        )

    def test_refuses_truncation_all_masked_or_missing_independent_check(self):
        truncated = deepcopy(self.row())
        truncated["generation"]["ended_with_eos"] = False
        truncated["generation"]["hit_max_new_tokens"] = True
        self.assertFalse(
            assess_penman_certified_release(
                truncated, independent_penman_valid=True
            ).releasable
        )
        all_masked = deepcopy(self.row())
        all_masked["constraint"]["all_masked_rows"] = 1
        all_masked["constraint"]["allowed_min"] = 0
        self.assertFalse(
            assess_penman_certified_release(
                all_masked, independent_penman_valid=True
            ).releasable
        )
        self.assertIn(
            "raw-surface-rejected-or-not-checked-by-independent-penman",
            assess_penman_certified_release(
                self.row(), independent_penman_valid=None
            ).reasons,
        )

    def test_selective_accounting_certifies_releases_without_requiring_coverage(self):
        release = {
            "releasable_rows": 1,
            "refused_rows": 1,
            "guarantee_counts": {PENMAN_RELEASE_GUARANTEE: 1},
            "reason_counts": {"generation-did-not-end-with-eos": 1},
            "refusals": [
                {
                    "id": "example-2",
                    "reasons": ["generation-did-not-end-with-eos"],
                }
            ],
        }
        self.assertTrue(
            has_complete_selective_release_accounting(release, attempts=2)
        )
        release["guarantee_counts"] = {PENMAN_RELEASE_GUARANTEE: 2}
        self.assertFalse(
            has_complete_selective_release_accounting(release, attempts=2)
        )


if __name__ == "__main__":
    unittest.main()
