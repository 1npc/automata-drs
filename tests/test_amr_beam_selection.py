from __future__ import annotations

import unittest

from automata_drs.amr_beam_selection import (
    BEAM_RANKING_METHOD,
    combine_amr_beam_rows,
    derive_postp_rows,
    validate_amr_beam_candidate_resume_prefix,
)


class AMRBeamSelectionTest(unittest.TestCase):
    @staticmethod
    def candidate(identifier: str, rank: int, *, valid: bool) -> dict[str, object]:
        scored = rank in {0, 2}
        return {
            "task": "amr",
            "id": identifier,
            "input": f"source-{identifier}",
            "reference": f"({identifier} / graph)",
            "prediction": f"({identifier} / candidate-{rank})",
            "raw_prediction": f"({identifier} / candidate-{rank})",
            "configuration": "U4",
            "decode": {
                "strategy": "beam",
                "num_beams": 4,
                "num_return_sequences": 4,
                "output_scores": False,
                "length_penalty": 1.0,
            },
            "beam_candidate": {
                "rank": rank,
                "ranking_method": BEAM_RANKING_METHOD,
                "sequence_score": None,
                "validation_ms": rank + 0.5,
            },
            "validity": {
                "raw_local_penman": valid,
                "raw_official_penman": valid,
            },
            "timing": {
                "generation_ms": 10.0,
                "model_score_ms": 1.0 if scored else 0.0,
                "constraint_ms": 0.0,
            },
            "score": {
                "raw_model_logprob": -10.0 - rank if scored else None,
                "raw_model_mean_logprob": -1.0 - rank if scored else None,
                "scored_tokens": 10 if scored else None,
            },
            "constraint": None,
        }

    def test_selects_first_valid_rank_and_scores_only_derived_rows(self):
        rows = [
            self.candidate("a", rank, valid=rank in {2, 3}) for rank in range(4)
        ]
        derived = derive_postp_rows(rows, ["a"])
        self.assertEqual(["U", "PostP"], [row["configuration"] for row in derived])
        selection = derived[1]["beam_selection"]
        self.assertEqual(2, selection["selected_candidate_rank"])
        self.assertEqual([2, 3], selection["raw_penman_valid_candidate_ranks"])
        self.assertFalse(selection["fallback_to_u_top1"])
        self.assertEqual(8.0, derived[1]["timing"]["constraint_ms"])
        self.assertIsNone(selection["candidate_raw_model_mean_logprobs"][1])

    def test_falls_back_to_scored_top1(self):
        rows = [self.candidate("a", rank, valid=False) for rank in range(4)]
        derived = derive_postp_rows(rows, ["a"])
        self.assertTrue(derived[1]["beam_selection"]["fallback_to_u_top1"])
        self.assertEqual(0, derived[1]["beam_selection"]["selected_candidate_rank"])
        self.assertEqual(derived[0]["raw_prediction"], derived[1]["raw_prediction"])

    def test_rejects_step_scores_partial_group_and_unscored_selection(self):
        rows = [self.candidate("a", rank, valid=rank == 2) for rank in range(4)]
        rows[0]["decode"]["output_scores"] = True
        with self.assertRaisesRegex(ValueError, "decode metadata"):
            derive_postp_rows(rows, ["a"])

        rows = [self.candidate("a", rank, valid=rank == 1) for rank in range(4)]
        with self.assertRaisesRegex(ValueError, "was not model-scored"):
            derive_postp_rows(rows, ["a"])
        with self.assertRaisesRegex(ValueError, "inside a beam group"):
            validate_amr_beam_candidate_resume_prefix(rows[:-1], ["a"])

    def test_combines_exact_online_p_rows(self):
        candidates = [
            self.candidate("a", rank, valid=rank == 2) for rank in range(4)
        ]
        online = self.candidate("a", 0, valid=True)
        online["configuration"] = "P"
        online["decode"] = {"strategy": "beam", "num_beams": 4}
        combined = combine_amr_beam_rows(candidates, [online], ["a"])
        self.assertEqual(
            ["U", "PostP", "P"], [row["configuration"] for row in combined]
        )


if __name__ == "__main__":
    unittest.main()
