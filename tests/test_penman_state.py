from __future__ import annotations

import unittest

from automata_drs import (
    PENMANConstraintState,
    SurfaceVocabulary,
    TokenConstraintController,
    validate_penman,
)


class PENMANConstraintStateTest(unittest.TestCase):
    def test_official_nested_example(self):
        text = (
            "(d / drive-01 :ARG0 (h / he) "
            ":manner (c / care-04 :polarity -))"
        )
        result = validate_penman(text)
        self.assertTrue(result.valid, result.reason)
        self.assertEqual("d", result.state.root_variable)
        self.assertEqual(frozenset({"d", "h", "c"}), result.state.declared_variables)
        self.assertIn(("d", "ARG0", "h"), result.state.node_edges)
        self.assertIn(("c", "polarity", "-"), result.state.constant_edges)

    def test_backward_reference_is_reentrancy(self):
        result = validate_penman(
            "(w / want-01 :ARG0 (b / boy) "
            ":ARG1 (g / go-02 :ARG0 b))"
        )
        self.assertTrue(result.valid, result.reason)
        self.assertIn(("g", "ARG0", "b"), result.state.node_edges)
        self.assertNotIn(("g", "ARG0", "b"), result.state.constant_edges)

    def test_forward_reference_is_resolved_after_declaration(self):
        result = validate_penman(
            "(w / want-01 :ARG1 x :ARG0 (x / boy))"
        )
        self.assertTrue(result.valid, result.reason)
        self.assertIn(("w", "ARG1", "x"), result.state.node_edges)

    def test_amr_cycle_is_not_rejected(self):
        result = validate_penman(
            "(p / procedure :purpose (e / ensure-01 :ARG0 p))"
        )
        self.assertTrue(result.valid, result.reason)
        self.assertIn(("e", "ARG0", "p"), result.state.node_edges)

    def test_quoted_constant_and_escape(self):
        result = validate_penman(
            r'(p / person :name (n / name :op1 "A\"da"))'
        )
        self.assertTrue(result.valid, result.reason)
        self.assertIn(("n", "op1", r'"A\"da"'), result.state.constant_edges)

    def test_duplicate_variable_declaration_is_rejected(self):
        result = validate_penman("(a / alpha :ARG0 (a / another))")
        self.assertFalse(result.valid)
        self.assertIn("duplicate", result.reason or "")

    def test_incomplete_prefix_stays_live_but_cannot_end(self):
        state = PENMANConstraintState.initial().advance_text("(w / want-01 :ARG0")
        self.assertTrue(state.is_extendable)
        self.assertFalse(state.can_end())

    def test_bad_delimiters_and_trailing_text_are_rejected(self):
        self.assertFalse(validate_penman("w / want-01").valid)
        self.assertFalse(validate_penman("(w / want-01))").valid)
        self.assertFalse(validate_penman("(w / want-01) x").valid)
        self.assertFalse(validate_penman("(w / want-01 : x)").valid)

    def test_record_whitespace_after_root_is_accepted(self):
        result = validate_penman("(w / want-01)\n\t ")
        self.assertTrue(result.valid, result.reason)

    def test_lifted_token_may_close_root_and_end_with_newline(self):
        vocabulary = SurfaceVocabulary(
            fragments={0: "(w / want-01)\n"},
            eos_token_ids=frozenset({1}),
        )
        controller = TokenConstraintController[PENMANConstraintState](vocabulary)
        closed = controller.transition(PENMANConstraintState.initial(), 0)
        self.assertIsNotNone(closed)
        self.assertIsNotNone(controller.transition(closed, 1))  # type: ignore[arg-type]

    def test_generic_token_controller_reuses_penman_state(self):
        vocabulary = SurfaceVocabulary(
            fragments={0: "(w / want", 1: "-01 :ARG0 (b / boy", 2: "))", 3: "]"},
            eos_token_ids=frozenset({4}),
        )
        controller = TokenConstraintController[PENMANConstraintState](vocabulary)
        state = PENMANConstraintState.initial()
        for token_id in (0, 1, 2):
            successor = controller.transition(state, token_id)
            self.assertIsNotNone(successor)
            state = successor  # type: ignore[assignment]
        self.assertIsNotNone(controller.transition(state, 4))
        self.assertIsNone(controller.transition(PENMANConstraintState.initial(), 3))


if __name__ == "__main__":
    unittest.main()
