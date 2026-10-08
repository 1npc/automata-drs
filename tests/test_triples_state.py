from __future__ import annotations

import unittest

from automata_drs import (
    ByteLevelTokenConstraintController,
    ByteLevelVocabulary,
    CanonicalTripleConstraintState,
    SurfaceVocabulary,
    TokenConstraintController,
    validate_canonical_triples,
)


VALID_GRAPH = "NODE a Person\nNODE b Book\nEDGE a likes b"


class CanonicalTripleConstraintStateTest(unittest.TestCase):
    def test_accepts_canonical_connected_graph(self):
        result = validate_canonical_triples(VALID_GRAPH)
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(
            (("a", "Person"), ("b", "Book")), result.state.nodes
        )
        self.assertEqual((("a", "likes", "b"),), result.state.edges)

    def test_enforces_declarations_uniqueness_and_canonical_order(self):
        cases = {
            "NODE b Book\nNODE a Person": "unique and increasing",
            "NODE a Person\nEDGE a likes z": "undeclared canonical edge target",
            (
                "NODE a Person\nNODE b Book\n"
                "EDGE a likes b\nEDGE a likes b"
            ): "unique and increasing",
            "EDGE a likes a": "record keyword",
        }
        for surface, reason in cases.items():
            with self.subTest(surface=surface):
                result = validate_canonical_triples(surface)
                self.assertFalse(result.valid)
                self.assertIn(reason, result.reason or "")

        impossible_start = CanonicalTripleConstraintState.initial().advance_text("E")
        self.assertFalse(impossible_start.is_extendable)

        unicode_digit = CanonicalTripleConstraintState.initial().advance_text(
            "NODE a١"
        )
        self.assertFalse(unicode_digit.is_extendable)

    def test_irreparable_order_prefixes_are_rejected_online(self):
        prefix = CanonicalTripleConstraintState.initial().advance_text(
            "NODE a Person\nNODE b Book\nEDGE a likes b\n"
        )
        self.assertTrue(prefix.is_extendable, prefix.dead_reason)
        lower_relation = prefix.advance_text("EDGE a d")
        self.assertFalse(lower_relation.is_extendable)
        self.assertIn("relation prefix", lower_relation.dead_reason or "")

        duplicate_target = prefix.advance_text("EDGE a likes b")
        self.assertFalse(duplicate_target.is_extendable)
        self.assertIn("target prefix", duplicate_target.dead_reason or "")

        ordered = prefix.advance_text("EDGE b likes a")
        self.assertTrue(ordered.can_end(), ordered.finish().dead_reason)

    def test_connectivity_is_an_eos_obligation(self):
        prefix = CanonicalTripleConstraintState.initial().advance_text(
            "NODE a Person\nNODE b Book"
        )
        self.assertTrue(prefix.is_extendable)
        self.assertFalse(prefix.can_end())
        connected = prefix.advance_text("\nEDGE a likes b")
        self.assertTrue(connected.can_end(), connected.finish().dead_reason)

    def test_surface_token_controller_splits_and_spans_records(self):
        vocabulary = SurfaceVocabulary(
            fragments={
                0: "NO",
                1: "DE a Person\nNODE b Book\n",
                2: "EDGE a likes b",
                3: "EDGE z likes b",
            },
            eos_token_ids=frozenset({4}),
        )
        controller = TokenConstraintController[CanonicalTripleConstraintState](
            vocabulary
        )
        state = CanonicalTripleConstraintState.initial()
        for token_id in (0, 1, 2):
            state = controller.transition(state, token_id)  # type: ignore[assignment]
            self.assertIsNotNone(state)
        finished = controller.transition(state, 4)
        self.assertIsNotNone(finished)
        self.assertTrue(finished.is_accepting)  # type: ignore[union-attr]
        self.assertIsNone(
            controller.transition(CanonicalTripleConstraintState.initial(), 3)
        )

    def test_exact_byte_lifting_uses_the_same_third_representation(self):
        vocabulary = ByteLevelVocabulary(
            fragments={
                0: b"NODE a Person\nNODE b Book\nEDGE a ",
                1: b"likes b",
                2: b"likes z",
            },
            eos_token_ids=frozenset({3}),
            byte_token_ids={},
        )
        controller = ByteLevelTokenConstraintController[
            CanonicalTripleConstraintState
        ](vocabulary)
        state = controller.initial(CanonicalTripleConstraintState.initial())
        state = controller.transition(state, 0)  # type: ignore[assignment]
        self.assertIsNotNone(state)
        self.assertIsNone(controller.transition(state, 2))  # type: ignore[arg-type]
        state = controller.transition(state, 1)  # type: ignore[arg-type,assignment]
        finished = controller.transition(state, 3)  # type: ignore[arg-type]
        self.assertIsNotNone(finished)
        self.assertTrue(finished.is_accepting)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
