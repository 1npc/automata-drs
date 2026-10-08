from __future__ import annotations

from copy import deepcopy
import unittest

from automata_drs.release import (
    CertifiedReleaseError,
    assess_sbn_certified_release,
    release_sbn_surface,
)


class CertifiedReleaseTest(unittest.TestCase):
    @staticmethod
    def row(configuration: str = "BF") -> dict[str, object]:
        return {
            "configuration": configuration,
            "raw_prediction": (
                "person.n.01 EQU speaker win.v.01 Agent -1 Time +1 "
                "time.n.08 TPR now"
            ),
            "validity": {
                "raw_sbn": True,
                "raw_clf_referee": True,
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
                "bounded_rows": 10,
                "completion_certificates_tested": 10,
                "completion_certificates_accepted": 10,
            },
        }

    def test_releases_recomputed_full_horizon_structural_output(self):
        decision = assess_sbn_certified_release(self.row())
        self.assertTrue(decision.releasable)
        self.assertEqual(
            "finite-horizon-structural-policy-membership", decision.guarantee
        )
        decision.require()
        released = release_sbn_surface(self.row())
        self.assertEqual(self.row()["raw_prediction"], released.text)
        self.assertEqual("BF", released.configuration)
        self.assertEqual(decision.guarantee, released.guarantee)

    def test_refuses_unconstrained_invalid_or_unfinished_outputs(self):
        unconstrained = self.row("U")
        self.assertFalse(assess_sbn_certified_release(unconstrained).releasable)

        invalid = self.row()
        invalid["raw_prediction"] = "not SBN"
        self.assertIn(
            "raw-surface-rejected-by-policy",
            assess_sbn_certified_release(invalid).reasons,
        )

        unfinished = self.row()
        unfinished["generation"]["ended_with_eos"] = False
        decision = assess_sbn_certified_release(unfinished)
        self.assertFalse(decision.releasable)
        with self.assertRaises(CertifiedReleaseError):
            decision.require()
        with self.assertRaises(CertifiedReleaseError):
            release_sbn_surface(unfinished)

    def test_refuses_all_masked_or_missing_certificate_activity(self):
        all_masked = self.row()
        all_masked["constraint"]["all_masked_rows"] = 1
        all_masked["constraint"]["allowed_min"] = 0
        self.assertIn(
            "processor-exposed-an-all-masked-row",
            assess_sbn_certified_release(all_masked).reasons,
        )

        inactive = self.row()
        inactive["constraint"]["bounded_rows"] = 9
        self.assertIn(
            "certificate-was-not-active-on-every-row",
            assess_sbn_certified_release(inactive).reasons,
        )

        shortened = self.row()
        self.assertIn(
            "completion-budget-factor-is-not-full-horizon",
            assess_sbn_certified_release(
                shortened, completion_budget_factor=0.75
            ).reasons,
        )

    def test_wf_requires_raw_semantic_acceptance(self):
        row = self.row("WF")
        accepted = assess_sbn_certified_release(row)
        self.assertTrue(accepted.releasable, accepted.reasons)

        missing = deepcopy(row)
        missing["validity"]["raw_clf_referee"] = None
        self.assertIn(
            "stored-raw-referee-validity-is-not-true",
            assess_sbn_certified_release(missing).reasons,
        )


if __name__ == "__main__":
    unittest.main()
