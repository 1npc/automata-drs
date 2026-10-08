import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from arr1012_review_full_audit import condition_key, require_complete_cells, score_vector


def row_score():
    row = {"id": "input-1", "configuration": "WWCD", "decode": {},
           "generation": {"decoder_failure": None}}
    score = {"id": "input-1", "configuration": "WWCD", "raw_W_valid": True,
             "f1": .8, "f1_upper": .8, "exact": True}
    return row, score


class FullAuditTest(unittest.TestCase):
    def test_rejected_raw_surface_is_zeroed_without_repair(self):
        row, score = row_score()
        self.assertEqual(score_vector(row, score, True, False), (.8, 0, True, False))
        self.assertEqual(score["f1"], .8)
        self.assertEqual(score_vector(row, score, True, True), (.8, .8, True, True))

    def test_parseable_failure_is_not_a_delivered_graph(self):
        row, score = row_score()
        row["generation"]["decoder_failure"] = {"reason": "all-masked"}
        score.update(raw_W_valid=False, f1=0, f1_upper=0)
        self.assertEqual(score_vector(row, score, True, True), (0, 0, False, False))
        score.update(f1=.8, f1_upper=.8)
        with self.assertRaises(ValueError):
            score_vector(row, score, True, True)

    def test_nonexact_mismatched_or_inconsistent_scoring_fails(self):
        row, good = row_score()
        for field, value in (("exact", False), ("f1_upper", .9), ("id", "different"),
                             ("raw_W_valid", False), ("f1", float("nan"))):
            bad = copy.deepcopy(good)
            bad[field] = value
            with self.assertRaises(ValueError):
                score_vector(row, bad, True, True)

    def test_denial_cells_do_not_collapse_into_natural_conditions(self):
        row, _ = row_score()
        self.assertEqual(condition_key(row, {"max_new_tokens": 68}), "H68:deny0:WWCD")
        row["review_operations"] = {"denial_rate": .5}
        self.assertEqual(condition_key(row, {"max_new_tokens": 68}), "H68:deny0.5:WWCD")
        with self.assertRaises(ValueError):
            condition_key(row, {})

    def test_full_identity_grid_and_frozen_values_are_required(self):
        mapping = {str(i): (.8, .8, True, True) for i in range(1132)}
        cells = {"one": mapping, "two": dict(mapping)}
        frozen = {"summaries": {key: {"raw_valid": 1132, "macro_f1": .8} for key in cells}}
        self.assertEqual(len(require_complete_cells(cells, frozen)), 1132)
        for changed in ({"one": mapping}, {"one": mapping, "two": dict(reversed(list(mapping.items())))},
                        {"one": mapping, "two": {k: v for k, v in mapping.items() if k != "0"}}):
            with self.assertRaises(ValueError):
                require_complete_cells(changed, frozen)
        frozen["summaries"]["one"]["macro_f1"] = .9
        with self.assertRaises(ValueError):
            require_complete_cells(cells, frozen)


if __name__ == "__main__":
    unittest.main()
