from __future__ import annotations

import unittest

from automata_drs.sbn import check_sbn_state_wellformedness, validate_sbn


class SBNWellFormednessTest(unittest.TestCase):
    def check(self, text: str):
        result = validate_sbn(text)
        self.assertTrue(result.valid, result.reason)
        return check_sbn_state_wellformedness(result.state)

    def test_nested_reference_is_accessible(self):
        result = self.check(
            "person.n.01 NEGATION <1 smile.v.01 Theme -1"
        )
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(frozenset({(0, 1)}), result.accessibility_subordination)
        self.assertEqual(frozenset({(0, 1)}), result.subordination)

    def test_reference_out_of_negation_creates_a_loop(self):
        result = self.check(
            "entity.n.01 be.v.01 Co-Theme +1 NEGATION <1 money.n.01"
        )
        self.assertFalse(result.valid)
        self.assertEqual("subordination-loop", result.family)
        self.assertEqual(frozenset({(0, 1), (1, 0), (0, 0), (1, 1)}), result.subordination)

    def test_proposition_connects_its_content_box(self):
        result = self.check(
            "think.v.01 Proposition >1 CONTINUATION <0 person.n.01"
        )
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(frozenset({(0, 1)}), result.direct_subordination)

    def test_unattached_continuation_is_disconnected(self):
        result = self.check(
            "person.n.01 CONTINUATION <0 person.n.01"
        )
        self.assertFalse(result.valid)
        self.assertEqual("box-connectedness", result.family)
        self.assertEqual(frozenset({1}), result.disconnected_boxes)

    def test_conjunction_aliases_contexts_before_scope_checking(self):
        result = self.check(
            "NEGATION <1 NEGATION <1 love.v.01 CONJUNCTION <2 person.n.01"
        )
        self.assertTrue(result.valid, result.reason)
        self.assertEqual((0, 1, 2, 1), result.box_aliases)


if __name__ == "__main__":
    unittest.main()
