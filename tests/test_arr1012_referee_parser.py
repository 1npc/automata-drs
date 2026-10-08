"""Regression tests for independent batch-audit tokenization, not decoding."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from arr1012_review_referee import parse_raw


class IndependentAuditParserTest(unittest.TestCase):
    def test_apostrophe_is_literal_in_gold_concept(self):
        graph = parse_raw("person.n.01 's.n.01 User -1")
        self.assertEqual([c.label for c in graph.concepts], ["person.n.01", "'s.n.01"])
        self.assertEqual((graph.concept_edges[0].source, graph.concept_edges[0].target), (1, 0))

    def test_apostrophe_inside_double_quoted_constant(self):
        graph = parse_raw('person.n.01 Name "Alice\'s book"')
        self.assertEqual(graph.constant_edges[0].target, '"Alice\'s book"')

    def test_double_quoted_name_and_forward_reference_are_preserved(self):
        graph = parse_raw('person.n.01 Name "Alice" Theme +1 entity.n.01')
        self.assertEqual(graph.constant_edges[0].target, '"Alice"')
        self.assertEqual((graph.concept_edges[0].source, graph.concept_edges[0].target), (0, 1))

    def test_unclosed_double_quote_is_still_rejected(self):
        with self.assertRaises(ValueError):
            parse_raw('person.n.01 Name "Alice')


if __name__ == "__main__":
    unittest.main()
