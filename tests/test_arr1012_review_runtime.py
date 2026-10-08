import sys
import unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from arr1012_review_runtime import (InstrumentedWF, InstrumentedAPD, denial_key,
    AllMaskedDecodingFailure, generate_with_failure_capture, deliverable_raw_valid)
from arr1012_review_rollback import rollback
from automata_drs import SBNPolicy
from automata_drs.subword import ByteLevelVocabulary, SubwordSBNLogitsProcessor
from test_subword import SubwordConstraintTest


class ReviewRuntimeTest(unittest.TestCase):
    def processor(self, cls, rate=0):
        return cls(tokenizer=SubwordConstraintTest.fake_tokenizer(), prompt_token_ids=[],
                   policy=SBNPolicy.for_configuration("W"), max_new_tokens=30,
                   review_identifier="fixed-id", review_denial_rate=rate)

    def test_initialization_is_exempt_even_at_total_denial(self):
        proc = self.processor(InstrumentedAPD, 1)
        self.assertTrue(proc.initial_certificate.final_state.is_accepting)
        self.assertEqual(proc.operations["forced_unknown"], 0)

    def test_total_denial_still_carries_initial_certificate(self):
        proc = self.processor(InstrumentedAPD, 1)
        prefix = ()
        state = proc.initial_certificate.start_state
        for token in proc.initial_certificate.token_ids:
            allowed = proc._ranked_allowed(state, [token], generated=len(prefix), prefix_token_ids=prefix)
            self.assertEqual(allowed, [token])
            state = proc.controller.transition(state, token)
            prefix += (token,)
        self.assertTrue(state.is_accepting)
        self.assertEqual(proc.operations["fresh_proposer_calls"], 0)

    def test_zero_denial_matches_original_wf_witness(self):
        proc = self.processor(InstrumentedWF)
        base = SubwordSBNLogitsProcessor(tokenizer=proc.tokenizer, prompt_token_ids=[], policy=SBNPolicy.for_configuration("W"),
                                        max_new_tokens=30, bounded_completion=True)
        state = proc.initial_certificate.start_state
        self.assertEqual(proc._completion_token_witness_within(state, 30), base._completion_token_witness_within(state, 30))

    def test_denial_key_is_stable(self):
        proc = self.processor(InstrumentedWF)
        state = proc.initial_certificate.start_state
        self.assertEqual(denial_key("x", state, 30), denial_key("x", state, 30))

    def test_failure_prefix_is_not_released_even_if_parseable(self):
        row = {"raw_prediction": "person.n.01", "generation": {}}
        self.assertTrue(deliverable_raw_valid(row))
        row["generation"]["decoder_failure"] = {"reason": "all-masked"}
        self.assertFalse(deliverable_raw_valid(row))

    def test_first_all_masked_raises_before_argmax_and_keeps_cost(self):
        import torch
        proc = self.processor(InstrumentedWF)
        proc.review_fail_fast = True
        ids = torch.tensor([[]], dtype=torch.long)
        with patch.object(proc, "_ranked_allowed", return_value=[]):
            with self.assertRaises(AllMaskedDecodingFailure) as caught:
                proc(ids, torch.zeros((1, 3)))
        self.assertEqual(caught.exception.prefix_token_ids, ())
        self.assertEqual(proc.stats().all_masked_rows, 1)
        self.assertEqual(proc.stats().calls, 1)
        self.assertGreater(proc.stats().seconds, 0)

    def test_failure_capture_preserves_exact_prefix_without_spurious_token(self):
        import torch
        class FakeModel:
            def generate(self, **kwargs):
                raise AllMaskedDecodingFailure([7, 8, 3, 4])
        suffix, failure = generate_with_failure_capture(
            FakeModel(), {"input_ids": torch.tensor([[7, 8]])}, 2)
        self.assertEqual(suffix.tolist(), [3, 4])
        self.assertEqual(failure["position"], 2)
        self.assertEqual(failure["native_prefix_token_ids"], [3, 4])

    def test_failure_capture_does_not_swallow_real_errors(self):
        class BrokenModel:
            def generate(self, **kwargs):
                raise RuntimeError("real failure")
        with self.assertRaisesRegex(RuntimeError, "real failure"):
            generate_with_failure_capture(BrokenModel(), {}, 0)

    def test_fail_fast_successful_logits_are_identical(self):
        import torch
        old, new = self.processor(InstrumentedWF), self.processor(InstrumentedWF)
        new.review_fail_fast = True
        ids = torch.tensor([[]], dtype=torch.long)
        scores = torch.tensor([[1., 3., 2.]])
        with patch.object(old, "_ranked_allowed", return_value=[1]), \
             patch.object(new, "_ranked_allowed", return_value=[1]):
            self.assertTrue(torch.equal(old(ids, scores), new(ids, scores)))

    def test_rollback_uses_latest_fitting_prefix(self):
        # Native path 0->1->2->3, suffix 2->accept costs 1; H=3.
        transition = lambda state, token: state+1 if token == 1 else (9 if token == 9 else None)
        result, info = rollback([1,1,1], 3, 0, lambda state, remaining: [9], transition, lambda state: state==9, [9])
        self.assertEqual(result, (1,1,9))
        self.assertEqual(info["kept_tokens"], 2)

    def test_rollback_initial_certificate_fallback(self):
        transition = lambda state, token: state+1 if token == 1 else 9
        result, info = rollback([1,1], 2, 0, lambda state, remaining: None, transition, lambda state: state==9, [9])
        self.assertEqual(result, (9,))
        self.assertTrue(info["fallback"])

    def test_rollback_rejects_unverified_completion(self):
        transition = lambda state, token: state+1 if token == 1 else (9 if token == 9 else None)
        result, info = rollback([1,1], 2, 0, lambda state, remaining: [7], transition, lambda state: state==9, [9])
        self.assertEqual(result, (9,))


if __name__ == "__main__":
    unittest.main()
