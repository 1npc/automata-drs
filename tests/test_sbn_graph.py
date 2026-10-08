from __future__ import annotations

import unittest

from automata_drs.sbn import SBNGraph, validate_sbn


class SBNGraphTest(unittest.TestCase):
    def test_projection_resolves_all_relative_references(self):
        result = validate_sbn(
            'male.n.02 Name "Tom" think.v.01 Agent -1 Proposition >1 '
            "CONTINUATION <0 person.n.01 EQU speaker"
        )
        self.assertTrue(result.valid, result.reason)

        graph = SBNGraph.from_state(result.state)
        self.assertEqual(2, graph.box_count)
        self.assertEqual(
            ((0, "male.n.02", 0), (1, "think.v.01", 0), (2, "person.n.01", 1)),
            tuple((node.index, node.label, node.box) for node in graph.concepts),
        )
        self.assertEqual(
            ((1, "Agent", 0),),
            tuple((edge.source, edge.label, edge.target) for edge in graph.concept_edges),
        )
        self.assertEqual(
            ((0, "Name", '"Tom"'), (2, "EQU", "speaker")),
            tuple((edge.source, edge.label, edge.target) for edge in graph.constant_edges),
        )
        self.assertEqual((), graph.box_edges)
        self.assertEqual(
            ((1, 1),),
            tuple((edge.source, edge.target_box) for edge in graph.proposition_edges),
        )

    def test_projection_rejects_an_unfinished_state(self):
        result = validate_sbn("person.n.01 Theme +1")
        self.assertFalse(result.valid)
        with self.assertRaisesRegex(ValueError, "accepting"):
            SBNGraph.from_state(result.state)


if __name__ == "__main__":
    unittest.main()
