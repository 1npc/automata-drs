from __future__ import annotations

import unittest
from dataclasses import dataclass, replace

from automata_drs import (
    CompletionDistanceCache,
    SBNConstraintState,
    SurfaceVocabulary,
    TokenConstraintController,
)


class CompletionDistanceCacheTest(unittest.TestCase):
    def test_positive_distance_dominates_all_other_budgets(self):
        cache = CompletionDistanceCache[str](max_entries=4)
        self.assertEqual((False, None), cache.lookup("state", 4))
        cache.store("state", 4, 3)
        self.assertEqual((True, 3), cache.lookup("state", 3))
        self.assertEqual((True, 3), cache.lookup("state", 8))
        self.assertEqual((True, None), cache.lookup("state", 2))
        info = cache.cache_info()
        self.assertEqual(3, info.hits)
        self.assertEqual(1, info.misses)
        self.assertEqual(1, info.size)

    def test_failed_budget_only_dominates_smaller_budgets(self):
        cache = CompletionDistanceCache[str](max_entries=4)
        cache.store("state", 3, None)
        self.assertEqual((True, None), cache.lookup("state", 2))
        self.assertEqual((False, None), cache.lookup("state", 4))
        cache.store("state", 4, 4)
        self.assertEqual((True, 4), cache.lookup("state", 10))

    def test_lru_bound_and_disabled_cache(self):
        cache = CompletionDistanceCache[str](max_entries=1)
        cache.store("first", 2, 1)
        cache.store("second", 2, 1)
        self.assertEqual((False, None), cache.lookup("first", 2))
        info = cache.cache_info()
        self.assertEqual(1, info.size)
        self.assertEqual(1, info.evictions)

        disabled = CompletionDistanceCache[str](max_entries=0)
        disabled.store("state", 2, 1)
        self.assertEqual((False, None), disabled.lookup("state", 2))
        self.assertEqual(0, disabled.cache_info().size)

    def test_inconsistent_or_out_of_budget_distance_is_rejected(self):
        cache = CompletionDistanceCache[str]()
        with self.assertRaisesRegex(ValueError, "fit"):
            cache.store("state", 2, 3)
        cache.store("state", 3, 2)
        with self.assertRaisesRegex(AssertionError, "changed"):
            cache.store("state", 3, 1)


class TokenConstraintControllerTest(unittest.TestCase):
    def setUp(self):
        self.vocabulary = SurfaceVocabulary(
            fragments={
                0: "male",
                1: ".n.02",
                2: " Name ",
                3: '"Tom"',
                4: " Agent -2",
                5: "male.n.02",
                6: " Theme +1 book.n.02",
                7: " Unknown ",
                8: "",
            },
            eos_token_ids=frozenset({9}),
        )
        self.controller = TokenConstraintController(self.vocabulary)

    def test_token_fragments_can_split_and_cross_lexemes(self):
        state = SBNConstraintState.initial()
        for token_id in [0, 1, 2, 3]:
            successor = self.controller.transition(state, token_id)
            self.assertIsNotNone(successor)
            state = successor  # type: ignore[assignment]
        finished = self.controller.transition(state, 9)
        self.assertIsNotNone(finished)
        self.assertTrue(finished.is_accepting)  # type: ignore[union-attr]
        self.assertEqual(1, finished.concept_count)  # type: ignore[union-attr]

    def test_token_may_span_several_complete_lexemes(self):
        state = self.controller.transition(SBNConstraintState.initial(), 5)
        self.assertIsNotNone(state)
        state = self.controller.transition(state, 6)  # type: ignore[arg-type]
        self.assertIsNotNone(state)
        self.assertTrue(state.can_end())  # type: ignore[union-attr]
        self.assertEqual(2, state.finish().concept_count)  # type: ignore[union-attr]

    def test_irreparable_reference_fragment_is_masked(self):
        state = self.controller.transition(SBNConstraintState.initial(), 5)
        self.assertIsNotNone(state)
        self.assertIsNone(self.controller.transition(state, 4))  # type: ignore[arg-type]

    def test_unknown_or_empty_fragments_are_masked(self):
        state = SBNConstraintState.initial()
        self.assertIsNone(self.controller.transition(state, 7))
        self.assertIsNone(self.controller.transition(state, 8))

    def test_eos_is_only_available_at_an_accepting_prefix(self):
        initial = SBNConstraintState.initial()
        self.assertNotIn(9, self.controller.allowed_token_ids(initial))
        state = self.controller.transition(initial, 5)
        self.assertIsNotNone(state)
        self.assertIn(9, self.controller.allowed_token_ids(state))  # type: ignore[arg-type]

    def test_transition_cache(self):
        state = SBNConstraintState.initial()
        self.controller.transition(state, 5)
        first = self.controller.cache_info()
        self.controller.transition(state, 5)
        second = self.controller.cache_info()
        self.assertEqual(1, first.misses)
        self.assertEqual(1, second.hits)
        self.assertEqual(1, second.size)

    def test_boolean_mask(self):
        mask = self.controller.boolean_mask(SBNConstraintState.initial(), 10)
        self.assertEqual(10, len(mask))
        self.assertTrue(mask[0])
        self.assertTrue(mask[5])
        self.assertFalse(mask[7])
        self.assertFalse(mask[8])
        self.assertFalse(mask[9])

    def test_controller_is_not_tied_to_sbn(self):
        @dataclass(frozen=True)
        class LiteralState:
            prefix: str = ""
            dead: bool = False
            finished: bool = False

            @property
            def is_extendable(self) -> bool:
                return not self.dead and not self.finished

            @property
            def is_accepting(self) -> bool:
                return self.finished and not self.dead

            def advance_text(self, fragment: str) -> "LiteralState":
                candidate = self.prefix + fragment
                return replace(
                    self,
                    prefix=candidate,
                    dead=not "graph".startswith(candidate),
                )

            def finish(self) -> "LiteralState":
                return replace(
                    self,
                    dead=self.prefix != "graph",
                    finished=self.prefix == "graph",
                )

        vocabulary = SurfaceVocabulary(
            fragments={0: "gr", 1: "aph", 2: "x"},
            eos_token_ids=frozenset({3}),
        )
        controller = TokenConstraintController[LiteralState](vocabulary)
        self.assertIsInstance(hash(LiteralState()), int)
        state = controller.transition(LiteralState(), 0)
        self.assertIsNotNone(state)
        state = controller.transition(state, 1)  # type: ignore[arg-type,assignment]
        self.assertIsNotNone(state)
        finished = controller.transition(state, 3)  # type: ignore[arg-type]
        self.assertIsNotNone(finished)
        self.assertTrue(finished.is_accepting)  # type: ignore[union-attr]
        self.assertIsNone(controller.transition(LiteralState(), 2))


if __name__ == "__main__":
    unittest.main()
