from __future__ import annotations

import unittest

from automata_drs import (
    ByteLevelConstraintLogitsProcessor,
    ByteLevelFiniteHorizonConstraintLogitsProcessor,
    ByteLevelWitnessCarryingLogitsProcessor,
    ByteLevelStructuralCompletionStoppingCriteria,
    ByteLevelVocabulary,
    PENMANConstraintState,
    construct_penman_completion_suffix,
)


class PENMANFiniteHorizonLogitsProcessorTest(unittest.TestCase):
    def setUp(self):
        self.vocabulary = ByteLevelVocabulary(
            fragments={
                0: b"(a / alpha :ARG0 (a ",  # OPEN but not append-only live.
                1: b"(a / alpha)",
                3: b"(a / ",
                4: b"alpha)",
                5: b"thing)",
                6: b"(x / thing)",
                7: b"(a / alpha)\n",
            },
            eos_token_ids=frozenset({2}),
            byte_token_ids={},
        )
        self.fragment_ids = {
            fragment.decode("utf-8"): token_id
            for token_id, fragment in self.vocabulary.fragments.items()
        }

    def encode_surface(self, text: str) -> tuple[int, ...]:
        if not text:
            return ()
        token_id = self.fragment_ids.get(text)
        return () if token_id is None else (token_id,)

    def processor(self, *, max_new_tokens: int = 3):
        return ByteLevelFiniteHorizonConstraintLogitsProcessor(
            tokenizer=object(),
            vocabulary=self.vocabulary,
            prompt_token_ids=(9,),
            initial_state=PENMANConstraintState.initial(),
            incomplete_utf8_is_extendable=(
                PENMANConstraintState.incomplete_utf8_is_extendable
            ),
            construct_surface_suffix=construct_penman_completion_suffix,
            encode_surface=self.encode_surface,
            max_new_tokens=max_new_tokens,
            retained_tokens_per_row=1,
            max_completion_cache_entries=8,
        )

    def wcd_processor(self, *, max_new_tokens: int = 3):
        return ByteLevelWitnessCarryingLogitsProcessor(
            tokenizer=object(),
            vocabulary=self.vocabulary,
            prompt_token_ids=(9,),
            initial_state=PENMANConstraintState.initial(),
            incomplete_utf8_is_extendable=(
                PENMANConstraintState.incomplete_utf8_is_extendable
            ),
            construct_surface_suffix=construct_penman_completion_suffix,
            encode_surface=self.encode_surface,
            max_new_tokens=max_new_tokens,
            retained_tokens_per_row=1,
            max_completion_cache_entries=8,
            max_prefix_certificates=8,
        )

    def test_pf_rejects_structurally_open_duplicate_declaration(self):
        prefix_only = ByteLevelConstraintLogitsProcessor(
            tokenizer=object(),
            vocabulary=self.vocabulary,
            prompt_token_ids=(9,),
            initial_state=PENMANConstraintState.initial(),
            incomplete_utf8_is_extendable=(
                PENMANConstraintState.incomplete_utf8_is_extendable
            ),
        )
        finite = self.processor(max_new_tokens=2)
        self.assertTrue(prefix_only.token_is_allowed_for_prefix((9,), 0))
        self.assertFalse(finite.token_is_allowed_for_prefix((9,), 0))
        self.assertTrue(finite.token_is_allowed_for_prefix((9,), 1))

    def test_exact_remaining_budget_includes_candidate_and_eos(self):
        enough = self.processor(max_new_tokens=2)
        self.assertTrue(enough.token_is_allowed_for_prefix((9,), 1))
        self.assertTrue(enough.token_is_allowed_for_prefix((9, 1), 2))
        too_short = self.processor(max_new_tokens=1)
        self.assertFalse(too_short.token_is_allowed_for_prefix((9,), 1))

    def test_graph_closing_token_with_transport_newline_is_certified(self):
        finite = self.processor(max_new_tokens=2)
        self.assertTrue(finite.token_is_allowed_for_prefix((9,), 7))
        self.assertTrue(finite.prefix_can_finish((9, 7)))
        self.assertTrue(finite.token_is_allowed_for_prefix((9, 7), 2))
        stopping = ByteLevelStructuralCompletionStoppingCriteria(finite)
        self.assertEqual((False, True), stopping.completed_rows(((9,), (9, 7))))

    def test_compositional_prefix_path_keeps_an_accepting_witness(self):
        finite = self.processor(max_new_tokens=3)
        self.assertTrue(finite.token_is_allowed_for_prefix((9,), 3))
        self.assertTrue(finite.token_is_allowed_for_prefix((9, 3), 4))
        self.assertTrue(finite.token_is_allowed_for_prefix((9, 3, 4), 2))

    def test_completion_distance_cache_reuses_state_across_budgets(self):
        finite = self.processor(max_new_tokens=3)
        state = finite.prefix_states.state_for((9, 1))
        self.assertIsNotNone(state)
        self.assertEqual(1, finite._completion_distance_within(state, 1))
        self.assertEqual(1, finite._completion_distance_within(state, 5))
        info = finite.completion_cache_info()
        self.assertEqual(1, info.misses)
        self.assertEqual(1, info.hits)
        self.assertEqual(1, info.size)

    def test_adversarial_scores_keep_only_certified_path(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch runtime is not installed")
        finite = self.processor(max_new_tokens=2)
        scores = torch.full((1, 10), -100.0)
        scores[0, 0] = 10.0  # P accepts it, but no accepting suffix exists.
        scores[0, 2] = 9.0  # EOS is structurally premature.
        scores[0, 1] = 8.0  # Lower-scoring but positively certified graph.
        masked = finite(torch.tensor([[9]]), scores)
        self.assertEqual(
            [1], torch.isfinite(masked[0]).nonzero().flatten().tolist()
        )

        eos_scores = torch.full((1, 10), -100.0)
        eos_scores[0, 2] = 10.0
        ended = finite(torch.tensor([[9, 1]]), eos_scores)
        self.assertEqual(
            [2], torch.isfinite(ended[0]).nonzero().flatten().tolist()
        )
        stats = finite.stats()
        self.assertEqual(0, stats.all_masked_rows)
        self.assertEqual(stats.rows, stats.bounded_rows)
        self.assertEqual(2, stats.allowed_total)
        self.assertEqual(2, stats.completion_certificates_accepted)
        self.assertGreater(stats.completion_certificates_tested, 2)
        self.assertEqual(0, stats.completion_slack_min)

    def test_invalid_constructor_arguments_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            self.processor(max_new_tokens=0)
        with self.assertRaisesRegex(ValueError, "max_completion_cache_entries"):
            ByteLevelFiniteHorizonConstraintLogitsProcessor(
                tokenizer=object(),
                vocabulary=self.vocabulary,
                prompt_token_ids=(9,),
                initial_state=PENMANConstraintState.initial(),
                construct_surface_suffix=construct_penman_completion_suffix,
                encode_surface=self.encode_surface,
                max_new_tokens=3,
                max_completion_cache_entries=-1,
            )

    def test_wcd_carries_head_and_replans_only_for_certified_deviation(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch runtime is not installed")
        wcd = self.wcd_processor(max_new_tokens=3)

        # The deterministic initial witness uses token 6, but higher-scoring
        # token 3 has its own two-step completion and may safely deviate.
        scores = torch.full((1, 10), -100.0)
        scores[0, 0] = 10.0  # Structurally live but cannot finish in budget.
        scores[0, 3] = 9.0   # Replans to token 4 followed by EOS.
        scores[0, 6] = 8.0   # Original witness head remains the fallback.
        first = wcd(torch.tensor([[9]]), scores)
        self.assertEqual(
            [3], torch.isfinite(first[0]).nonzero().flatten().tolist()
        )

        second_scores = torch.full((1, 10), -100.0)
        second_scores[0, 5] = 5.0
        second = wcd(torch.tensor([[9, 3]]), second_scores)
        self.assertEqual(
            [5], torch.isfinite(second[0]).nonzero().flatten().tolist()
        )
        eos_scores = torch.full((1, 10), -100.0)
        eos_scores[0, 2] = 5.0
        ended = wcd(torch.tensor([[9, 3, 5]]), eos_scores)
        self.assertEqual(
            [2], torch.isfinite(ended[0]).nonzero().flatten().tolist()
        )
        stats = wcd.stats()
        self.assertEqual(0, stats.all_masked_rows)
        self.assertGreaterEqual(stats.replanned_tokens, 1)
        self.assertGreaterEqual(stats.carried_tokens, 2)
        self.assertGreaterEqual(stats.rejected_without_certificate, 1)

    def test_wcd_initial_certificate_failure_masks_everything(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch runtime is not installed")
        wcd = ByteLevelWitnessCarryingLogitsProcessor(
            tokenizer=object(),
            vocabulary=self.vocabulary,
            prompt_token_ids=(9,),
            initial_state=PENMANConstraintState.initial(),
            construct_surface_suffix=lambda state: "]",
            encode_surface=self.encode_surface,
            max_new_tokens=3,
        )
        masked = wcd(torch.tensor([[9]]), torch.zeros((1, 10)))
        self.assertFalse(bool(torch.isfinite(masked).any().item()))
        self.assertEqual(1, wcd.stats().prefix_rows_unavailable)
        self.assertEqual(1, wcd.stats().all_masked_rows)

    def test_wcd_certificate_ownership_survives_beam_row_reorder(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch runtime is not installed")
        wcd = self.wcd_processor(max_new_tokens=3)
        scores = torch.full((2, 10), -100.0)
        scores[0, 3] = 10.0  # Child certificate is (5, EOS).
        scores[1, 6] = 10.0  # Child certificate is only EOS.
        first = wcd(torch.tensor([[9], [9]]), scores)
        self.assertEqual(
            [[3], [6]],
            [
                torch.isfinite(row).nonzero().flatten().tolist()
                for row in first
            ],
        )

        # Reverse the two hypotheses. Prefix identity, not row position, must
        # select the residual certificate appropriate for each child.
        reordered_scores = torch.full((2, 10), -100.0)
        reordered_scores[0, 2] = 10.0
        reordered_scores[1, 5] = 10.0
        reordered = wcd(
            torch.tensor([[9, 6], [9, 3]]), reordered_scores
        )
        self.assertEqual(
            [[2], [5]],
            [
                torch.isfinite(row).nonzero().flatten().tolist()
                for row in reordered
            ],
        )
        self.assertEqual(0, wcd.stats().all_masked_rows)


if __name__ == "__main__":
    unittest.main()
