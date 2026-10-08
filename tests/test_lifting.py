from __future__ import annotations

import unittest

from automata_drs import (
    ByteLevelConstraintLogitsProcessor,
    ByteLevelPrefixStateCache,
    ByteLevelTokenConstraintController,
    ByteLevelVocabulary,
    PENMANConstraintState,
)


class RepresentationNeutralByteLiftingTest(unittest.TestCase):
    def setUp(self):
        # The Chinese scalar is deliberately split over three model tokens.
        self.vocabulary = ByteLevelVocabulary(
            fragments={
                0: b'(p / person :name (n / name :op1 "',
                1: b"\xe5",
                2: b"\xbc",
                3: b"\xa0",
                4: b'"))',
                6: "张".encode("utf-8"),
                7: b"A",
            },
            eos_token_ids=frozenset({5}),
            byte_token_ids={},
        )
        self.controller = ByteLevelTokenConstraintController(
            self.vocabulary,
            incomplete_utf8_is_extendable=(
                PENMANConstraintState.incomplete_utf8_is_extendable
            ),
        )

    def test_representation_state_is_hashable_for_transition_cache(self):
        self.assertIsInstance(hash(PENMANConstraintState.initial()), int)

    def test_split_unicode_penman_surface_reaches_exact_acceptance(self):
        state = self.controller.initial(PENMANConstraintState.initial())
        # A non-ASCII partial scalar is impossible in PENMAN's structural
        # start mode and is rejected before it can create a dead-end prefix.
        self.assertIsNone(self.controller.transition(state, 1))

        state = self.controller.transition(state, 0)  # type: ignore[assignment]
        self.assertIsNotNone(state)
        state = self.controller.transition(state, 1)  # type: ignore[arg-type,assignment]
        self.assertEqual(b"\xe5", state.pending_utf8)  # type: ignore[union-attr]
        state = self.controller.transition(state, 2)  # type: ignore[arg-type,assignment]
        self.assertEqual(b"\xe5\xbc", state.pending_utf8)  # type: ignore[union-attr]
        state = self.controller.transition(state, 3)  # type: ignore[arg-type,assignment]
        self.assertEqual(b"", state.pending_utf8)  # type: ignore[union-attr]
        self.assertIn("张", state.representation.buffer)  # type: ignore[union-attr]
        state = self.controller.transition(state, 4)  # type: ignore[arg-type,assignment]
        finished = self.controller.transition(state, 5)  # type: ignore[arg-type]
        self.assertIsNotNone(finished)
        self.assertTrue(finished.is_accepting)  # type: ignore[union-attr]
        self.assertIn(
            ("n", "op1", '"张"'),
            finished.representation.constant_edges,  # type: ignore[union-attr]
        )

    def test_invalid_utf8_continuation_and_eos_are_rejected(self):
        state = self.controller.initial(PENMANConstraintState.initial())
        state = self.controller.transition(state, 0)  # type: ignore[assignment]
        partial = self.controller.transition(state, 1)  # type: ignore[arg-type]
        self.assertIsNotNone(partial)
        self.assertIsNone(self.controller.transition(partial, 7))  # type: ignore[arg-type]
        self.assertIsNone(self.controller.transition(partial, 5))  # type: ignore[arg-type]

    def test_missing_partial_scalar_guard_fails_closed(self):
        controller = ByteLevelTokenConstraintController(self.vocabulary)
        state = controller.initial(PENMANConstraintState.initial())
        state = controller.transition(state, 0)  # type: ignore[assignment]
        self.assertIsNone(controller.transition(state, 1))  # type: ignore[arg-type]
        # A complete UTF-8 token does not need a partial-scalar guard.
        complete = controller.transition(state, 6)  # type: ignore[arg-type]
        self.assertIsNotNone(complete)
        self.assertIn("张", complete.representation.buffer)  # type: ignore[union-attr]

    def test_prefix_cache_is_prompt_safe_and_beam_order_independent(self):
        cache = ByteLevelPrefixStateCache(
            self.controller,
            prompt_token_ids=(8, 9),
            initial_state=PENMANConstraintState.initial(),
        )
        partial = cache.state_for((8, 9, 0, 1, 2))
        self.assertEqual(b"\xe5\xbc", partial.pending_utf8)  # type: ignore[union-attr]
        shorter = cache.state_for((8, 9, 0))
        self.assertEqual(b"", shorter.pending_utf8)  # type: ignore[union-attr]
        complete = cache.state_for((8, 9, 0, 6, 4, 5))
        self.assertTrue(complete.is_accepting)  # type: ignore[union-attr]
        with self.assertRaisesRegex(ValueError, "fixed prompt"):
            cache.state_for((8, 10))

    def test_score_ranked_processor_masks_by_penman_state(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch runtime is not installed")
        processor = ByteLevelConstraintLogitsProcessor(
            tokenizer=object(),
            vocabulary=self.vocabulary,
            prompt_token_ids=(9,),
            initial_state=PENMANConstraintState.initial(),
            incomplete_utf8_is_extendable=(
                PENMANConstraintState.incomplete_utf8_is_extendable
            ),
            retained_tokens_per_row=1,
        )
        scores = torch.full((1, 10), -100.0)
        scores[0, 5] = 10.0  # EOS is illegal at the start.
        scores[0, 1] = 9.0  # Partial Unicode is impossible at the start.
        scores[0, 0] = 8.0  # Highest-scoring valid PENMAN token.
        masked = processor(torch.tensor([[9]]), scores)
        finite = torch.isfinite(masked[0]).nonzero().flatten().tolist()
        self.assertEqual([0], finite)

        unicode_scores = torch.full((1, 10), -100.0)
        unicode_scores[0, 5] = 10.0
        unicode_scores[0, 1] = 9.0
        unicode_masked = processor(torch.tensor([[9, 0]]), unicode_scores)
        finite = torch.isfinite(unicode_masked[0]).nonzero().flatten().tolist()
        self.assertEqual([1], finite)

        eos_scores = torch.full((1, 10), -100.0)
        eos_scores[0, 5] = 10.0
        eos_masked = processor(
            torch.tensor([[9, 0, 1, 2, 3, 4]]), eos_scores
        )
        finite = torch.isfinite(eos_masked[0]).nonzero().flatten().tolist()
        self.assertEqual([5], finite)
        self.assertEqual(0, processor.stats().all_masked_rows)


if __name__ == "__main__":
    unittest.main()
