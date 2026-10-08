from __future__ import annotations

import unittest

from automata_drs.sbn import sbn_state_to_clf, validate_sbn


class SBNCLFProjectionTest(unittest.TestCase):
    def project(self, text: str):
        result = validate_sbn(text)
        self.assertTrue(result.valid, result.reason)
        return sbn_state_to_clf(result.state)

    def test_simple_concepts_hooks_and_constants(self):
        projection = self.project(
            'male.n.02 Name "Alfred Nobel" see.v.01 Agent -1 '
            "Time +1 time.n.08 TPR now"
        )
        self.assertIn(("b0", "REF", "x0"), projection.clauses)
        self.assertIn(("b0", "male", '"n.02"', "x0"), projection.clauses)
        self.assertIn(("b0", "Name", "x0", '"alfred~nobel"'), projection.clauses)
        self.assertIn(("b0", "Agent", "x1", "x0"), projection.clauses)
        self.assertIn(("b0", "Time", "x1", "x2"), projection.clauses)
        self.assertIn(("b0", "TPR", "x2", '"now"'), projection.clauses)

    def test_nested_connectors_and_proposition(self):
        projection = self.project(
            "think.v.01 Proposition >1 NEGATION <1 person.n.01"
        )
        self.assertIn(("b0", "Proposition", "x0", "b1"), projection.clauses)
        self.assertIn(("b0", "NOT", "b1"), projection.clauses)

    def test_conjunction_continues_connected_drs(self):
        projection = self.project(
            "NEGATION <1 NEGATION <1 love.v.01 CONJUNCTION <2 person.n.01"
        )
        self.assertEqual((0, 1, 2, 1), projection.box_aliases)
        self.assertIn(("b0", "NOT", "b1"), projection.clauses)
        self.assertIn(("b1", "NOT", "b2"), projection.clauses)
        self.assertIn(("b1", "person", '"n.01"', "x1"), projection.clauses)
        self.assertFalse(any(clause[1] == "CONJUNCTION" for clause in projection.clauses))

    def test_other_separators_remain_discourse_relations(self):
        projection = self.project(
            "event.n.01 CONTINUATION <1 event.n.01 CONTRAST <1 event.n.01"
        )
        self.assertIn(("b0", "CONTINUATION", "b1"), projection.clauses)
        self.assertIn(("b1", "CONTRAST", "b2"), projection.clauses)


if __name__ == "__main__":
    unittest.main()
