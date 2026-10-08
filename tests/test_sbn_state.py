from __future__ import annotations

import unittest

from automata_drs.sbn import (
    SBNConstraintState,
    SBNPolicy,
    validate_sbn,
    validate_sbn_lexemes,
)


class SBNConstraintStateTest(unittest.TestCase):
    def assert_valid(self, text: str) -> None:
        result = validate_sbn(text)
        self.assertTrue(result.valid, result.reason)

    def assert_invalid(self, text: str, reason_fragment: str) -> None:
        result = validate_sbn(text)
        self.assertFalse(result.valid)
        self.assertIn(reason_fragment, result.reason or "")

    def test_simple_sequence(self):
        self.assert_valid(
            "male.n.02 Name \"Tom\" see.v.01 Experiencer -1 "
            "Time +1 time.n.08 TPR now"
        )

    def test_quoted_constant_may_contain_spaces(self):
        self.assert_valid('male.n.02 Name "Alfred Nobel"')

    def test_name_target_signature(self):
        self.assert_valid('male.n.02 Name "Tom"')
        self.assert_valid("male.n.02 Name ?")
        self.assert_invalid("male.n.02 Name -1", "invalid target prefix")
        self.assert_invalid("male.n.02 Name Tom", "invalid target prefix")

    def test_arbitrary_fragments_match_whole_string(self):
        fragments = ["male.n", '.02 Name "Al', 'fred Nobel"', " see.v.01 Agent", " -1"]
        state = SBNConstraintState.initial()
        for fragment in fragments:
            state = state.advance_text(fragment)
            self.assertTrue(state.is_extendable, state.dead_reason)
        self.assertTrue(state.can_end())

    def test_lexeme_fast_path_agrees_with_character_path(self):
        examples = [
            'male.n.02 Name "Alfred Nobel"',
            "NEGATION <1 person.n.01 Theme +1 entity.n.01",
            "person.n.01 Theme +1",
            "person.n.01 Quantity? 1",
        ]
        for text in examples:
            with self.subTest(text=text):
                slow = validate_sbn(text)
                fast = validate_sbn_lexemes(text)
                self.assertEqual(slow.valid, fast.valid)

    def test_separator_can_start_a_sequence(self):
        self.assert_valid("NEGATION <1 person.n.01")

    def test_nested_empty_context_prefixes(self):
        self.assert_valid("NEGATION <1 NEGATION <1 person.n.01")

    def test_context_registers_match_reference_box_arithmetic(self):
        text = (
            "person.n.01 CONTINUATION <0 male.n.02 "
            "CONTINUATION <3 entity.n.01 Theme -1"
        )
        result = validate_sbn(text)
        self.assertTrue(result.valid, result.reason)
        self.assertEqual((0, 1, 2), result.state.concept_boxes)
        # `<0>` creates box 1 without an edge. On the next separator, `<3>`
        # resolves from new box 2 to -1 and the reference converter clamps it
        # to root box 0.
        self.assertEqual(
            ((0, 2, "CONTINUATION"),),
            result.state.box_edges,
        )
        self.assertIn((2, 1, "Theme"), result.state.node_edges)

    def test_proposition_and_separator_box_edges_are_distinct(self):
        text = (
            "think.v.01 Proposition >1 CONTINUATION <1 person.n.01"
        )
        result = validate_sbn(text)
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(((0, 1),), result.state.proposition_targets)
        self.assertEqual(
            ((0, 1, "CONTINUATION"),), result.state.box_edges
        )

    def test_state_retains_lossless_typed_graph_labels(self):
        text = (
            'male.n.02 Name "Tom" see.v.01 Agent -1 Time +1 '
            "time.n.08 TPR now"
        )
        result = validate_sbn(text)
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(
            ("male.n.02", "see.v.01", "time.n.08"),
            result.state.concept_labels,
        )
        self.assertEqual((0, 0, 0), result.state.concept_boxes)
        self.assertEqual(
            ((0, "Name", '"Tom"'), (2, "TPR", "now")),
            result.state.constant_edges,
        )
        self.assertEqual(
            frozenset({(1, 0, "Agent"), (1, 2, "Time")}),
            result.state.node_edges,
        )

    def test_forward_reference_becomes_valid(self):
        prefix = SBNConstraintState.initial().advance_text("buy.v.01 Theme +1")
        self.assertFalse(prefix.can_end())
        completed = prefix.advance_text(" book.n.02")
        self.assertTrue(completed.can_end(), completed.finish().dead_reason)

    def test_negative_reference_before_start_is_rejected(self):
        self.assert_invalid("person.n.01 Agent -2", "before the first concept")

    def test_irreparable_reference_prefix_dies_before_a_boundary(self):
        state = SBNConstraintState.initial().advance_text("person.n.01 Agent -2")
        self.assertTrue(state.is_dead)
        self.assertIn("before the first concept", state.dead_reason or "")

    def test_unresolved_forward_reference_is_rejected_at_eos(self):
        self.assert_invalid("person.n.01 Theme +1", "unresolved forward concept")

    def test_strict_policy_rejects_self_and_two_node_cycles(self):
        for text in (
            "person.n.01 Theme +0",
            "person.n.01 Theme +1 entity.n.01 Theme -1",
        ):
            result = validate_sbn(text)
            self.assertFalse(result.valid)
            self.assertIn("cyclic", result.reason or "")

    def test_strict_policy_rejects_cycle_at_the_boundary(self):
        state = SBNConstraintState.initial().advance_text(
            "person.n.01 Theme +1 entity.n.01 Theme -1"
        )
        self.assertTrue(state.is_dead)
        self.assertIn("cyclic", state.dead_reason or "")

    def test_cyclic_negative_prefix_is_kept_only_if_more_digits_can_repair_it(self):
        concepts = [f"entity_{index}.n.01" for index in range(16)]
        text = " ".join(
            [*concepts[:15], "Theme +1", concepts[15], "Theme -1"]
        )
        state = SBNConstraintState.initial().advance_text(text)
        self.assertTrue(state.is_extendable, state.dead_reason)
        self.assertTrue(state.advance_text("0").can_end())
        rejected_boundary = state.advance_text(" ")
        self.assertTrue(rejected_boundary.is_dead)
        self.assertIn("cyclic", rejected_boundary.dead_reason or "")

    def test_reference_compatible_policy_allows_cycles(self):
        policy = SBNPolicy.reference_parser_compatible()
        for text in (
            "person.n.01 Theme +0",
            "person.n.01 Theme +1 entity.n.01 Theme -1",
        ):
            result = validate_sbn(text, policy)
            self.assertTrue(result.valid, result.reason)

    def test_acyclicity_covers_box_concept_and_proposition_edges(self):
        text = (
            "person.n.01 CONTINUATION <1 "
            "think.v.01 Proposition <1"
        )
        self.assert_invalid(text, "SBN graph cyclic")
        permissive = validate_sbn(text, SBNPolicy.reference_parser_compatible())
        self.assertTrue(permissive.valid, permissive.reason)

    def test_forward_proposition_cycle_is_rejected_after_target_materializes(self):
        # The Proposition edge is registered before box 1 or concept 1 exists.
        # Adding the final backward concept edge closes
        # concept0 -> box1 -> concept1 -> concept0 and must still be caught.
        text = (
            "think.v.01 Proposition >1 CONTINUATION <1 "
            "person.n.01 Theme -1"
        )
        self.assertTrue(
            validate_sbn(text, SBNPolicy.for_configuration("R")).valid
        )
        result = validate_sbn(text, SBNPolicy.for_configuration("G"))
        self.assertFalse(result.valid)
        self.assertIn("cyclic", result.reason or "")

    def test_noncanonical_leading_zero_reference_is_rejected(self):
        # `+0` is already an irreparable self-loop under the strict DAG
        # policy; appending `1` cannot turn it into a canonical reference.
        self.assert_invalid("person.n.01 Theme +01", "cyclic")

    def test_unknown_label_is_rejected(self):
        self.assert_invalid("person.n.01 Quantity? 1", "cannot complete")

    def test_missing_attachment_target_is_rejected(self):
        self.assert_invalid("person.n.01 Agent", "attachment-target")

    def test_proposition_forward_context_obligation(self):
        prefix = SBNConstraintState.initial().advance_text(
            "think.v.01 Proposition >1"
        )
        self.assertFalse(prefix.can_end())
        self.assertTrue(prefix.advance_text(" CONTINUATION <0 person.n.01").can_end())

    def test_proposition_cannot_target_before_first_context(self):
        self.assert_invalid("think.v.01 Proposition <1", "before the first context")

    def test_irreparable_proposition_self_box_prefix_dies_immediately(self):
        state = SBNConstraintState.initial().advance_text(
            "think.v.01 Proposition >0"
        )
        self.assertTrue(state.is_dead)
        self.assertIn("SBN graph cyclic", state.dead_reason or "")

    def test_irreparable_partial_proposition_sign_dies_before_all_masked(self):
        # The prefix is lexically compatible with <0 or <1, but both existing
        # boxes would close a typed cycle and every larger backward target is
        # out of range.  The bare sign must therefore die immediately instead
        # of reaching a token-level state with no successor.
        text = (
            "person.n.01 EQU speaker be.v.02 Theme -1 Time +1 Co-Theme +2 "
            "time.n.08 EQU now person.n.01 Role +1 commissar.n.01 "
            "CONTINUATION <1 male.n.02 say.v.01 Proposition <"
        )
        state = SBNConstraintState.initial(
            SBNPolicy.for_configuration("B")
        ).advance_text(text)
        self.assertTrue(state.is_dead)
        self.assertIn("no viable target", state.dead_reason or "")

        # A forward connector still has a concrete future-box completion.
        forward = SBNConstraintState.initial(
            SBNPolicy.for_configuration("B")
        ).advance_text("think.v.01 Proposition >")
        self.assertTrue(forward.is_extendable, forward.dead_reason)

    def test_non_proposition_role_cannot_target_context(self):
        self.assert_invalid("person.n.01 Agent >1", "cannot complete")

    def test_proposition_cannot_target_a_concept_or_constant(self):
        self.assert_invalid("think.v.01 Proposition +1 person.n.01", "invalid target")
        self.assert_invalid("think.v.01 Proposition now", "invalid target")

    def test_semantic_referee_policy_rejects_scope_loop_online(self):
        policy = SBNPolicy.semantic_referee()
        text = "entity.n.01 be.v.01 Co-Theme +1 NEGATION <1 money.n.01"
        result = validate_sbn(text, policy)
        self.assertFalse(result.valid)
        self.assertIn("DRS well-formedness", result.reason or "")
        # Once `m` starts a concept at the obligated future index, no concept
        # label can repair the box-subordination loop.
        prefix = SBNConstraintState.initial(policy).advance_text(
            "entity.n.01 be.v.01 Co-Theme +1 NEGATION <1 m"
        )
        self.assertTrue(prefix.is_dead)

    def test_semantic_referee_policy_checks_connectedness_only_at_eos(self):
        policy = SBNPolicy.semantic_referee()
        prefix = SBNConstraintState.initial(policy).advance_text(
            "person.n.01 CONTINUATION <0 person.n.01"
        )
        self.assertTrue(prefix.is_extendable, prefix.dead_reason)
        finished = prefix.finish()
        self.assertTrue(finished.is_dead)
        self.assertIn("disconnected", finished.dead_reason or "")

    def test_semantic_referee_policy_accepts_accessible_nested_reference(self):
        result = validate_sbn(
            "person.n.01 NEGATION <1 smile.v.01 Theme -1",
            SBNPolicy.semantic_referee(),
        )
        self.assertTrue(result.valid, result.reason)

    def test_executable_ablation_policies_add_constraints_monotonically(self):
        self.assertTrue(
            validate_sbn(
                "person.n.01 Agent -2", SBNPolicy.for_configuration("S")
            ).valid
        )
        self.assertFalse(
            validate_sbn(
                "person.n.01 Agent -2", SBNPolicy.for_configuration("R")
            ).valid
        )

        cycle = "person.n.01 Theme +1 entity.n.01 Theme -1"
        self.assertTrue(validate_sbn(cycle, SBNPolicy.for_configuration("R")).valid)
        self.assertFalse(validate_sbn(cycle, SBNPolicy.for_configuration("G")).valid)

        proposition = "think.v.01 Proposition >1"
        self.assertTrue(
            validate_sbn(proposition, SBNPolicy.for_configuration("G")).valid
        )
        self.assertFalse(
            validate_sbn(proposition, SBNPolicy.for_configuration("B")).valid
        )

        disconnected = "person.n.01 CONTINUATION <0 person.n.01"
        self.assertTrue(
            validate_sbn(disconnected, SBNPolicy.for_configuration("B")).valid
        )
        self.assertFalse(
            validate_sbn(disconnected, SBNPolicy.for_configuration("W")).valid
        )

    def test_canonical_whitespace(self):
        self.assert_invalid(" person.n.01", "whitespace")
        self.assert_invalid("person.n.01  Agent -1", "whitespace")
        self.assert_invalid("person.n.01 ", "whitespace")

    def test_constant_families_observed_in_gold(self):
        for target in [
            "speaker",
            "?",
            "+",
            "38.5",
            "20:00",
            "3/4",
            "3-0",
            "'198X'",
            "A",
            "i",
        ]:
            with self.subTest(target=target):
                self.assert_valid(f"entity.n.01 EQU {target}")


if __name__ == "__main__":
    unittest.main()
