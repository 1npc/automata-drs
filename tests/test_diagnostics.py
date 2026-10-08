from __future__ import annotations

import unittest

from automata_drs import ConstraintFamily, SBNPolicy, diagnose_sbn


class PrefixDiagnosticTest(unittest.TestCase):
    def test_valid_sequence_has_no_failure_event(self):
        result = diagnose_sbn("male.n.02")
        self.assertTrue(result.valid)
        self.assertIsNone(result.family)
        self.assertIsNone(result.event)

    def test_unknown_lexeme_is_rejected_at_the_first_irreparable_character(self):
        text = "male.n.02 EQU? now"
        result = diagnose_sbn(text)
        self.assertFalse(result.valid)
        self.assertEqual(ConstraintFamily.LEXICAL_OR_SEQUENCE, result.family)
        self.assertEqual("character", result.event)
        self.assertEqual("?", result.offending_text)
        self.assertEqual(text.index("?") + 1, result.char_offset)

    def test_unresolved_reference_is_an_eos_failure(self):
        text = "male.n.02 Theme +1"
        result = diagnose_sbn(text)
        self.assertFalse(result.valid)
        self.assertEqual(ConstraintFamily.REFERENCE_EXISTENCE, result.family)
        self.assertEqual("eos", result.event)
        self.assertEqual("<eos>", result.offending_text)

    def test_irreparable_cycle_is_rejected_before_lexeme_boundary(self):
        text = "person.n.01 Theme +1 book.n.01 Theme -1"
        result = diagnose_sbn(text)
        self.assertFalse(result.valid)
        self.assertEqual(ConstraintFamily.GRAPH_ACYCLICITY, result.family)
        self.assertEqual("1", result.offending_text)

    def test_unicode_byte_offset_is_reported_for_byt5(self):
        text = "café.n.01 EQU? now"
        result = diagnose_sbn(text)
        self.assertEqual("?", result.offending_text)
        self.assertEqual(result.char_offset + 1, result.byte_offset)

    def test_semantic_scope_loop_is_its_own_constraint_family(self):
        text = "entity.n.01 be.v.01 Co-Theme +1 NEGATION <1 money.n.01"
        result = diagnose_sbn(text, SBNPolicy.semantic_referee())
        self.assertFalse(result.valid)
        self.assertEqual(ConstraintFamily.SEMANTIC_WELLFORMEDNESS, result.family)
        self.assertEqual("character", result.event)
        self.assertEqual("m", result.offending_text)


if __name__ == "__main__":
    unittest.main()
