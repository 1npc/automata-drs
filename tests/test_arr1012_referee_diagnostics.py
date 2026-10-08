import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from arr1012_review_referee_diagnostics import paired_contributions


class PairedImpactTest(unittest.TestCase):
    def test_full_denominator_and_lost_are_retained(self):
        apd = [(0.8, 0.8, True, True), (0.7, 0.7, True, True),
               (0.0, 0.0, False, False), (0.0, 0.0, False, False)]
        base = [(0.6, 0.6, True, True), (0.0, 0.0, False, False),
                (0.4, 0.4, True, True), (0.0, 0.0, False, False)]
        result = paired_contributions(apd, base)["native"]
        self.assertEqual(result["counts"], {"common_valid": 1, "rescue": 1,
                                           "lost": 1, "neither_valid": 1})
        self.assertAlmostEqual(result["delta"]["mean_points"], 12.5)
        self.assertAlmostEqual(result["common_valid"]["mean_points"], 5)
        self.assertAlmostEqual(result["rescue"]["mean_points"], 17.5)
        self.assertAlmostEqual(result["lost"]["mean_points"], -10)

    def test_rejected_original_changes_policy_not_output(self):
        apd = [(0.8, 0.0, True, False), (0.7, 0.7, True, True)]
        base = [(0.6, 0.6, True, True), (0.5, 0.5, True, True)]
        result = paired_contributions(apd, base)
        self.assertAlmostEqual(result["native"]["delta"]["mean_points"], 20)
        self.assertAlmostEqual(result["compatibility_zeroed"]["delta"]["mean_points"], -20)
        self.assertEqual(result["compatibility_zeroed"]["counts"]["lost"], 1)
        self.assertEqual(apd[0], (0.8, 0.0, True, False))

    def test_identical_methods_have_exact_zero_interval(self):
        rows = [(0.7, 0.7, True, True), (0.0, 0.0, False, False)]
        result = paired_contributions(rows, rows)
        for policy in result.values():
            self.assertEqual(policy["delta"]["ci95_points"], [0.0, 0.0])

    def test_invalid_nonzero_or_changed_valid_score_rejected(self):
        good = [(0.7, 0.7, True, True)]
        for bad in ([(0.7, 0.0, False, False)], [(0.7, 0.0, True, True)],
                    [(0.0, 0.0, False, True)], [(0.7, 0.7, 0.5, True)]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                paired_contributions(bad, good)

    def test_empty_or_nonmatched_grid_rejected(self):
        with self.assertRaises(ValueError):
            paired_contributions([], [])
        with self.assertRaises(ValueError):
            paired_contributions([(0.5, 0.5, True, True)], [])


if __name__ == "__main__":
    unittest.main()
