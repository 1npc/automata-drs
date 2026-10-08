from __future__ import annotations

import unittest

from automata_drs.beam_selection import (
    combine_beam_selection_rows,
    derive_postw_rows,
    validate_beam_candidate_resume_prefix,
)


class BeamSelectionTest(unittest.TestCase):
    @staticmethod
    def candidate(
        identifier: str, rank: int, *, score: float, w_valid: bool
    ) -> dict[str, object]:
        return {
            "id": identifier,
            "input": f"source-{identifier}",
            "reference": f"reference-{identifier}",
            "prediction": f"prediction-{identifier}-{rank}",
            "raw_prediction": f"prediction-{identifier}-{rank}",
            "configuration": "U4",
            "decode": {
                "strategy": "beam",
                "num_beams": 4,
                "num_return_sequences": 4,
            },
            "beam_candidate": {
                "rank": rank,
                "sequence_score": score,
                "validation_ms": rank + 0.5,
            },
            "validity": {
                "raw_sbn": w_valid,
                "clf_referee": True if w_valid else None,
            },
            "timing": {"generation_ms": 10.0, "constraint_ms": 0.0},
        }

    def test_selects_highest_score_w_valid_and_records_fallback(self):
        rows = [
            self.candidate("a", rank, score=-rank, w_valid=rank in {1, 3})
            for rank in range(4)
        ] + [
            self.candidate("b", rank, score=-rank, w_valid=False)
            for rank in range(4)
        ]
        derived = derive_postw_rows(rows, ["a", "b"])

        self.assertEqual(
            [(row["id"], row["configuration"]) for row in derived],
            [("a", "U"), ("a", "PostW"), ("b", "U"), ("b", "PostW")],
        )
        self.assertEqual(
            1, derived[1]["beam_selection"]["selected_candidate_rank"]
        )
        self.assertFalse(derived[1]["beam_selection"]["fallback_to_u_top1"])
        self.assertEqual(
            [1, 3], derived[1]["beam_selection"]["w_valid_candidate_ranks"]
        )
        self.assertEqual(8.0, derived[1]["timing"]["constraint_ms"])
        self.assertEqual(
            0, derived[3]["beam_selection"]["selected_candidate_rank"]
        )
        self.assertTrue(derived[3]["beam_selection"]["fallback_to_u_top1"])
        self.assertEqual(0, rows[0]["beam_candidate"]["rank"])

    def test_rejects_incomplete_or_misordered_candidate_grid(self):
        rows = [
            self.candidate("a", rank, score=-rank, w_valid=True)
            for rank in range(4)
        ]
        with self.assertRaisesRegex(ValueError, "expected 8"):
            derive_postw_rows(rows, ["a", "b"])

        changed = [dict(row) for row in rows]
        changed[2] = dict(changed[2], id="b")
        with self.assertRaisesRegex(ValueError, "schedule mismatch"):
            derive_postw_rows(changed, ["a"])

    def test_rejects_nonmonotonic_or_nonfinite_scores(self):
        rows = [
            self.candidate("a", rank, score=-rank, w_valid=True)
            for rank in range(4)
        ]
        rows[2]["beam_candidate"]["sequence_score"] = 1.0
        with self.assertRaisesRegex(ValueError, "non-increasing"):
            derive_postw_rows(rows, ["a"])

        rows = [
            self.candidate("a", rank, score=-rank, w_valid=True)
            for rank in range(4)
        ]
        rows[1]["beam_candidate"]["sequence_score"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            derive_postw_rows(rows, ["a"])

    def test_resume_requires_whole_exact_candidate_groups(self):
        rows = [
            self.candidate("a", rank, score=-rank, w_valid=True)
            for rank in range(4)
        ]
        self.assertEqual(
            1, validate_beam_candidate_resume_prefix(rows, ["a", "b"])
        )
        with self.assertRaisesRegex(ValueError, "inside a candidate group"):
            validate_beam_candidate_resume_prefix(rows[:-1], ["a"])

    def test_combines_rotated_online_rows_in_fixed_comparison_order(self):
        candidates = [
            self.candidate("a", rank, score=-rank, w_valid=rank == 1)
            for rank in range(4)
        ]
        online = []
        for configuration in ("WF", "W"):
            row = self.candidate("a", 0, score=0.0, w_valid=True)
            row["configuration"] = configuration
            row["decode"] = {"strategy": "beam", "num_beams": 4}
            online.append(row)
        combined = combine_beam_selection_rows(candidates, online, ["a"])
        self.assertEqual(
            ["U", "PostW", "W", "WF"],
            [row["configuration"] for row in combined],
        )


if __name__ == "__main__":
    unittest.main()
