from __future__ import annotations

import unittest

from automata_drs import (
    ByteLevelTokenConstraintController,
    ByteLevelVocabulary,
    CanonicalTripleConstraintState,
    PENMANConstraintState,
    SBNConstraintState,
)


class RepresentationAdapterContractTest(unittest.TestCase):
    CASES = (
        (
            "sbn",
            SBNConstraintState.initial,
            'person.n.01 Name "Tom"',
            "? ",
        ),
        (
            "penman",
            PENMANConstraintState.initial,
            "(p / person)",
            "x",
        ),
        (
            "canonical-kg",
            CanonicalTripleConstraintState.initial,
            "NODE a Person",
            "E",
        ),
    )

    def test_a1_fragment_composition_and_hashable_immutable_states(self):
        for name, initial, surface, _ in self.CASES:
            whole = initial().advance_text(surface)
            self.assertTrue(whole.is_extendable, name)
            self.assertFalse(whole.is_accepting, name)
            self.assertIsInstance(hash(whole), int, name)
            for split in range(len(surface) + 1):
                with self.subTest(adapter=name, split=split):
                    state = initial().advance_text(surface[:split])
                    self.assertTrue(state.is_extendable)
                    composed = state.advance_text(surface[split:])
                    self.assertEqual(whole, composed)
                    finished = composed.finish()
                    self.assertTrue(finished.is_accepting)
                    self.assertEqual(finished, finished.finish())

    def test_a2_dead_states_are_absorbing_and_finished_states_cannot_advance(self):
        for name, initial, surface, invalid in self.CASES:
            with self.subTest(adapter=name):
                dead = initial().advance_text(invalid)
                self.assertFalse(dead.is_extendable)
                self.assertFalse(dead.is_accepting)
                self.assertEqual(dead, dead.advance_text(surface))

                finished = initial().advance_text(surface).finish()
                self.assertTrue(finished.is_accepting)
                after = finished.advance_text("x")
                self.assertFalse(after.is_extendable)
                self.assertFalse(after.is_accepting)

    def test_a4_a5_exact_byte_tokens_reach_acceptance_only_through_eos(self):
        for name, initial, surface, _ in self.CASES:
            encoded = surface.encode("utf-8")
            split = len(encoded) // 2
            vocabulary = ByteLevelVocabulary(
                fragments={0: encoded[:split], 1: encoded[split:]},
                eos_token_ids=frozenset({2}),
                byte_token_ids={},
            )
            controller = ByteLevelTokenConstraintController(vocabulary)
            state = controller.initial(initial())
            with self.subTest(adapter=name):
                self.assertIsNone(controller.transition(state, 2))
                state = controller.transition(state, 0)
                self.assertIsNotNone(state)
                self.assertFalse(state.is_accepting)  # type: ignore[union-attr]
                state = controller.transition(state, 1)  # type: ignore[arg-type]
                self.assertIsNotNone(state)
                self.assertFalse(state.is_accepting)  # type: ignore[union-attr]
                finished = controller.transition(state, 2)  # type: ignore[arg-type]
                self.assertIsNotNone(finished)
                self.assertTrue(finished.is_accepting)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()

