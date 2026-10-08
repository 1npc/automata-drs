from __future__ import annotations

import unittest

from automata_drs.scoring import (
    merge_semantic_scores,
    parse_sbn_smatch_output,
    validate_semantic_pair_consistency,
)


class ScoringAdapterTest(unittest.TestCase):
    def test_parse_preserves_missing_items_as_zero(self):
        scores = parse_sbn_smatch_output(
            "0:1.0\ngenerated sbn 1 error: bad\n2:0.25\naverage f1 smatch score: 0.416\n",
            3,
        )
        self.assertEqual([1.0, 0.0, 0.25], scores)

    def test_merge_uses_id_configuration_and_strategy(self):
        record = {
            "id": "1",
            "configuration": "B",
            "decode": {"strategy": "beam", "num_beams": 4},
        }
        score = {
            **record,
            "semantic_scores": {"sbn_smatch": 0.75},
        }
        merged = merge_semantic_scores([record], [score])
        self.assertEqual(0.75, merged[0]["semantic_scores"]["sbn_smatch"])

    def test_invalid_indices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            parse_sbn_smatch_output("3:1.0", 3)

    def test_semantic_pair_consistency_rejects_scorer_noise(self):
        rows = [
            {
                "reference": "dog.n.01",
                "prediction": "dog.n.01",
                "semantic_scores": {"sbn_smatch": score},
            }
            for score in (1.0, 0.75)
        ]
        with self.assertRaisesRegex(ValueError, "inconsistent semantic scores"):
            validate_semantic_pair_consistency(rows)

        rows[1]["semantic_scores"]["sbn_smatch"] = 1.0
        diagnostics = validate_semantic_pair_consistency(rows)
        self.assertEqual(1, diagnostics["unique_semantic_pairs"])
        self.assertEqual(1, diagnostics["reused_score_records"])


if __name__ == "__main__":
    unittest.main()
