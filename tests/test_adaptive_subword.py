from __future__ import annotations

import unittest

from automata_drs.adaptive_subword import (
    AdaptiveSubwordSBNLogitsProcessor,
    StableSubwordSBNLogitsProcessor,
    ThreeWaySubwordRankingAuditProcessor,
)
from automata_drs.subword import gpt2_byte_decoder


class AdaptiveSubwordTest(unittest.TestCase):
    @staticmethod
    def torch():
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest("optional Torch runtime is unavailable") from error
        return torch

    @staticmethod
    def fake_tokenizer():
        inverse = gpt2_byte_decoder()
        encoder = {byte: character for character, byte in inverse.items()}

        class ByteLevel:
            pass

        class Backend:
            decoder = ByteLevel()

        class Tokenizer:
            backend_tokenizer = Backend()
            all_special_ids = [400]
            eos_token_id = 400

            @staticmethod
            def get_vocab():
                vocabulary = {encoder[byte]: byte for byte in range(256)}
                vocabulary.update(
                    {
                        "male.n.02": 300,
                        "dog.n.01": 301,
                        "<eos>": 400,
                    }
                )
                return vocabulary

            @staticmethod
            def __call__(text, *, add_special_tokens):
                del add_special_tokens
                return {"input_ids": list(text.encode("utf-8"))}

        return Tokenizer()

    def processors(self, **kwargs):
        common = {
            "tokenizer": self.fake_tokenizer(),
            "prompt_token_ids": (399,),
            "retained_tokens_per_row": 1,
            **kwargs,
        }
        return (
            StableSubwordSBNLogitsProcessor(**common),
            AdaptiveSubwordSBNLogitsProcessor(**common, initial_k=2),
        )

    def test_stable_and_adaptive_masks_match_with_ties_and_unmapped_ids(self):
        torch = self.torch()
        stable, adaptive = self.processors()
        scores = torch.full((1, 405), -20.0)
        scores[0, 400] = 10.0  # premature EOS
        scores[0, 401] = 10.0  # unmapped model position
        scores[0, 32] = 9.0  # invalid leading whitespace
        scores[0, 300] = 8.0
        scores[0, 301] = 8.0
        prefix = torch.tensor([[399]])
        stable_mask = stable(prefix, scores)
        adaptive_mask = adaptive(prefix, scores)
        self.assertTrue(torch.equal(stable_mask, adaptive_mask))
        self.assertEqual(
            [300],
            torch.isfinite(adaptive_mask[0]).nonzero().flatten().tolist(),
        )
        ranking = adaptive.ranking_stats()
        self.assertEqual(1, ranking.ranked_rows)
        self.assertGreater(ranking.topk_rounds, 1)
        self.assertEqual(
            2 * ranking.cumulative_candidate_pool,
            ranking.python_scalar_values_materialized,
        )
        self.assertEqual(
            adaptive.stats().candidates_tested,
            ranking.certificate_tests,
        )

    def test_bounded_completion_decision_is_equivalent(self):
        torch = self.torch()
        stable, adaptive = self.processors(
            bounded_completion=True,
            max_new_tokens=2,
        )
        scores = torch.full((1, 401), -20.0)
        scores[0, 301] = 5.0
        scores[0, 300] = 4.0
        prefix = torch.tensor([[399]])
        stable_mask = stable(prefix, scores)
        adaptive_mask = adaptive(prefix, scores)
        self.assertTrue(torch.equal(stable_mask, adaptive_mask))
        self.assertEqual(1, adaptive.stats().bounded_rows)
        self.assertGreater(
            adaptive.stats().completion_certificates_tested, 0
        )

    def test_three_way_shadow_returns_frozen_mask_and_records_rows(self):
        torch = self.torch()
        audit = ThreeWaySubwordRankingAuditProcessor.from_shared_configuration(
            tokenizer=self.fake_tokenizer(),
            prompt_token_ids=(399,),
            retained_tokens_per_row=1,
            initial_k=2,
        )
        scores = torch.full((1, 401), -20.0)
        scores[0, 400] = 10.0
        scores[0, 300] = 8.0
        prefix = torch.tensor([[399]])
        expected = audit.frozen(prefix, scores)
        # Use a fresh audit so the assertion also covers independent caches.
        audit = ThreeWaySubwordRankingAuditProcessor.from_shared_configuration(
            tokenizer=self.fake_tokenizer(),
            prompt_token_ids=(399,),
            retained_tokens_per_row=1,
            initial_k=2,
        )
        observed = audit(prefix, scores)
        self.assertTrue(torch.equal(expected, observed))
        stats = audit.stats()
        self.assertEqual(1, stats.calls)
        self.assertEqual(1, stats.rows)
        self.assertEqual(0, stats.stable_adaptive_mismatches)

    def test_nonfinite_raw_logits_fail_closed(self):
        torch = self.torch()
        _, adaptive = self.processors()
        scores = torch.zeros((1, 401))
        scores[0, 300] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            adaptive(torch.tensor([[399]]), scores)

    def test_three_way_shadow_aborts_on_stable_adaptive_mismatch(self):
        torch = self.torch()
        stable, adaptive = self.processors()

        class BrokenAdaptive:
            @staticmethod
            def __call__(input_ids, scores):
                del input_ids
                return scores.new_full(scores.shape, float("-inf"))

        audit = ThreeWaySubwordRankingAuditProcessor(
            frozen=StableSubwordSBNLogitsProcessor(
                tokenizer=self.fake_tokenizer(),
                prompt_token_ids=(399,),
                retained_tokens_per_row=1,
            ),
            stable=stable,
            adaptive=BrokenAdaptive(),  # type: ignore[arg-type]
        )
        scores = torch.full((1, 401), -20.0)
        scores[0, 300] = 8.0
        with self.assertRaisesRegex(
            AssertionError, "stable/adaptive ranking mismatch"
        ):
            audit(torch.tensor([[399]]), scores)
        self.assertEqual(1, audit.stats().stable_adaptive_mismatches)

    def test_three_way_shadow_records_frozen_tie_difference_but_returns_it(self):
        torch = self.torch()
        stable, adaptive = self.processors()

        class DifferentFrozen:
            @staticmethod
            def __call__(input_ids, scores):
                del input_ids
                masked = scores.new_full(scores.shape, float("-inf"))
                masked[:, 301] = scores[:, 301]
                return masked

        audit = ThreeWaySubwordRankingAuditProcessor(
            frozen=DifferentFrozen(),  # type: ignore[arg-type]
            stable=stable,
            adaptive=adaptive,
        )
        scores = torch.full((1, 401), -20.0)
        scores[0, 300] = 8.0
        scores[0, 301] = 7.0
        observed = audit(torch.tensor([[399]]), scores)
        self.assertEqual(
            [301], torch.isfinite(observed[0]).nonzero().flatten().tolist()
        )
        stats = audit.stats()
        self.assertEqual(1, stats.frozen_stable_mismatches)
        self.assertEqual(0, stats.stable_adaptive_mismatches)
        self.assertEqual((301,), stats.frozen_stable_examples[0].frozen_token_ids)
        self.assertEqual((300,), stats.frozen_stable_examples[0].stable_token_ids)


if __name__ == "__main__":
    unittest.main()
