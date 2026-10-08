from __future__ import annotations

import random
import unittest

from automata_drs.ranking import (
    adaptive_certified_tokens,
    exhaustive_certified_tokens,
)


class AdaptiveRankingTest(unittest.TestCase):
    @staticmethod
    def torch():
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest("optional Torch runtime is unavailable") from error
        return torch

    def test_matches_exhaustive_across_random_sparse_predicates(self):
        torch = self.torch()
        for seed in range(12):
            rng = random.Random(seed)
            scores = torch.tensor(
                [rng.randrange(-8, 9) / 4 for _ in range(257)],
                dtype=torch.float32,
            )
            certified = {
                token_id for token_id in range(257) if rng.random() < 0.08
            }
            for required in (1, 2, 8):
                reference, _ = exhaustive_certified_tokens(
                    scores, certified.__contains__, required=required
                )
                observed, stats = adaptive_certified_tokens(
                    scores,
                    certified.__contains__,
                    required=required,
                    initial_k=3,
                )
                self.assertEqual(reference, observed, (seed, required))
                self.assertLessEqual(stats.certificate_tests, 257)

    def test_boundary_ties_are_closed_and_token_id_breaks_ties(self):
        torch = self.torch()
        scores = torch.tensor([1.0, 5.0, 5.0, 5.0, 0.0])
        certified = {2, 3}
        observed, stats = adaptive_certified_tokens(
            scores, certified.__contains__, required=1, initial_k=1
        )
        self.assertEqual((2,), observed)
        self.assertEqual(2, stats.boundary_ties_added)
        self.assertEqual(3, stats.cumulative_candidate_pool)
        self.assertEqual(6, stats.python_scalar_values_materialized)
        self.assertFalse(stats.full_vocabulary_reached)

    def test_expands_until_sparse_certificates_are_proven(self):
        torch = self.torch()
        scores = torch.arange(64, dtype=torch.float32)
        observed, stats = adaptive_certified_tokens(
            scores, {0, 1}.__contains__, required=2, initial_k=2
        )
        self.assertEqual((1, 0), observed)
        self.assertTrue(stats.full_vocabulary_reached)
        self.assertGreater(stats.topk_rounds, 1)
        self.assertEqual(64, stats.certificate_tests)
        self.assertGreater(stats.cumulative_candidate_pool, 64)
        self.assertEqual(
            2 * stats.cumulative_candidate_pool,
            stats.python_scalar_values_materialized,
        )

    def test_each_certificate_is_evaluated_at_most_once(self):
        torch = self.torch()
        scores = torch.arange(32, dtype=torch.float32)
        calls: dict[int, int] = {}

        def certify(token_id: int) -> bool:
            calls[token_id] = calls.get(token_id, 0) + 1
            return token_id == 0

        observed, stats = adaptive_certified_tokens(
            scores, certify, required=1, initial_k=2
        )
        self.assertEqual((0,), observed)
        self.assertEqual(stats.certificate_tests, len(calls))
        self.assertTrue(all(count == 1 for count in calls.values()))

    def test_invalid_inputs_fail_closed(self):
        torch = self.torch()
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            adaptive_certified_tokens(
                torch.zeros((1, 2)), lambda _: True, required=1
            )
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite"):
                    adaptive_certified_tokens(
                        torch.tensor([0.0, invalid]),
                        lambda _: True,
                        required=1,
                    )
        with self.assertRaisesRegex(ValueError, "required"):
            adaptive_certified_tokens(
                torch.zeros(2), lambda _: True, required=0
            )
        with self.assertRaisesRegex(ValueError, "initial_k"):
            adaptive_certified_tokens(
                torch.zeros(2), lambda _: True, required=1, initial_k=0
            )


if __name__ == "__main__":
    unittest.main()
