from __future__ import annotations

import unittest

from automata_drs import (
    CertificateSearchResult,
    PrefixCertificateStore,
    ReachabilityStatus,
    WitnessCarryingController,
    WitnessDecisionKind,
    certificate_result_from_exact_search,
    certify_constructive_plan,
    exact_bounded_reachability,
    verify_accepting_certificate,
)


class AcceptingCertificateTest(unittest.TestCase):
    transitions = {
        (0, 10): 1,
        (0, 11): 3,
        (1, 20): 2,
        (3, 30): 3,
        (3, 40): 2,
    }

    @classmethod
    def transition(cls, state: int, token_id: int) -> int | None:
        return cls.transitions.get((state, token_id))

    @staticmethod
    def is_accepting(state: int) -> bool:
        return state == 2

    def test_verifies_and_consumes_prefix_owned_residual(self):
        result = verify_accepting_certificate(
            0,
            (10, 20),
            transition=self.transition,
            is_accepting=self.is_accepting,
            max_steps=2,
        )
        self.assertTrue(result.accepted)
        certificate = result.certificate
        self.assertIsNotNone(certificate)
        self.assertEqual(2, certificate.distance)  # type: ignore[union-attr]
        residual = certificate.consume_head(  # type: ignore[union-attr]
            10,
            transition=self.transition,
            observed_successor=1,
        )
        self.assertEqual(1, residual.start_state)
        self.assertEqual((20,), residual.token_ids)
        finished = residual.consume_head(
            20,
            transition=self.transition,
            observed_successor=2,
        )
        self.assertEqual((), finished.token_ids)
        self.assertEqual(2, finished.start_state)
        self.assertEqual(finished.start_state, finished.final_state)

    def test_residual_rejects_wrong_token_or_prefix_owner(self):
        certificate = verify_accepting_certificate(
            0,
            (10, 20),
            transition=self.transition,
            is_accepting=self.is_accepting,
            max_steps=2,
        ).certificate
        self.assertIsNotNone(certificate)
        with self.assertRaisesRegex(ValueError, "head"):
            certificate.consume_head(11, transition=self.transition)  # type: ignore[union-attr]
        with self.assertRaisesRegex(ValueError, "own"):
            certificate.consume_head(  # type: ignore[union-attr]
                10,
                transition=self.transition,
                observed_successor=3,
            )

    def test_verifier_reports_budget_transition_and_acceptance_failures(self):
        over_budget = verify_accepting_certificate(
            0,
            (10, 20),
            transition=self.transition,
            is_accepting=self.is_accepting,
            max_steps=1,
        )
        self.assertEqual("budget-exceeded", over_budget.failure_reason)
        invalid = verify_accepting_certificate(
            0,
            (10, 30),
            transition=self.transition,
            is_accepting=self.is_accepting,
            max_steps=2,
        )
        self.assertEqual("invalid-transition", invalid.failure_reason)
        self.assertEqual(1, invalid.failure_index)
        open_path = verify_accepting_certificate(
            0,
            (11,),
            transition=self.transition,
            is_accepting=self.is_accepting,
            max_steps=1,
        )
        self.assertEqual("non-accepting-final-state", open_path.failure_reason)

    def test_constructive_failure_is_unknown_not_unreachable(self):
        missing = certify_constructive_plan(
            0,
            max_steps=2,
            propose=lambda state, budget: None,
            transition=self.transition,
            is_accepting=self.is_accepting,
        )
        self.assertEqual(ReachabilityStatus.UNKNOWN, missing.status)
        malformed = certify_constructive_plan(
            0,
            max_steps=2,
            propose=lambda state, budget: (10, 30),
            transition=self.transition,
            is_accepting=self.is_accepting,
        )
        self.assertEqual(ReachabilityStatus.UNKNOWN, malformed.status)
        self.assertIn("invalid-transition", malformed.reason or "")

    def test_exact_oracle_result_is_independently_replayed(self):
        exact = exact_bounded_reachability(
            0,
            max_steps=2,
            token_ids=lambda state: (10, 11, 20, 30),
            transition=self.transition,
            is_accepting=self.is_accepting,
        )
        result = certificate_result_from_exact_search(
            0,
            exact,
            max_steps=2,
            transition=self.transition,
            is_accepting=self.is_accepting,
        )
        self.assertEqual(ReachabilityStatus.REACHABLE, result.status)
        self.assertEqual((10, 20), result.certificate.token_ids)  # type: ignore[union-attr]

        unreachable = exact_bounded_reachability(
            3,
            max_steps=2,
            token_ids=lambda state: (30,),
            transition=self.transition,
            is_accepting=self.is_accepting,
        )
        negative = certificate_result_from_exact_search(
            3,
            unreachable,
            max_steps=2,
            transition=self.transition,
            is_accepting=self.is_accepting,
        )
        self.assertEqual(ReachabilityStatus.UNREACHABLE, negative.status)
        self.assertIsNone(negative.certificate)

    def test_result_invariants_prevent_unverified_reachability_claims(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            CertificateSearchResult(
                status=ReachabilityStatus.REACHABLE,
                certificate=None,
            )
        with self.assertRaisesRegex(ValueError, "explanation"):
            CertificateSearchResult(
                status=ReachabilityStatus.UNKNOWN,
                certificate=None,
            )

    def test_prefix_store_survives_beam_reorder_without_slot_identity(self):
        first = verify_accepting_certificate(
            0,
            (10, 20),
            transition=self.transition,
            is_accepting=self.is_accepting,
            max_steps=2,
        ).certificate
        second = verify_accepting_certificate(
            1,
            (20,),
            transition=self.transition,
            is_accepting=self.is_accepting,
            max_steps=1,
        ).certificate
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        store = PrefixCertificateStore[int](max_entries=4)
        store.bind((100,), 0, first)  # type: ignore[arg-type]
        store.bind((200,), 1, second)  # type: ignore[arg-type]

        # Retrieval order is the opposite of insertion, like a beam reorder.
        self.assertEqual((20,), store.get((200,)).token_ids)  # type: ignore[union-attr]
        self.assertEqual((10, 20), store.get((100,)).token_ids)  # type: ignore[union-attr]
        residual = store.carry_if_head(
            (100,),
            10,
            transition=self.transition,
            observed_successor=1,
        )
        self.assertEqual((20,), residual.token_ids)  # type: ignore[union-attr]
        self.assertIs(store.get((100, 10)), residual)
        self.assertIsNone(
            store.carry_if_head(
                (200,),
                30,
                transition=self.transition,
                observed_successor=3,
            )
        )

    def test_prefix_store_fails_closed_on_wrong_state_and_is_bounded(self):
        certificate = verify_accepting_certificate(
            0,
            (10, 20),
            transition=self.transition,
            is_accepting=self.is_accepting,
            max_steps=2,
        ).certificate
        self.assertIsNotNone(certificate)
        store = PrefixCertificateStore[int](max_entries=1)
        with self.assertRaisesRegex(ValueError, "start"):
            store.bind((1,), 1, certificate)  # type: ignore[arg-type]
        store.bind((1,), 0, certificate)  # type: ignore[arg-type]
        store.bind((2,), 0, certificate)  # type: ignore[arg-type]
        self.assertIsNone(store.get((1,)))
        self.assertEqual(1, len(store))
        store.retain(())
        self.assertEqual(0, len(store))

    def witness_controller(self) -> WitnessCarryingController[int]:
        plans = {0: (10, 20), 1: (20,), 2: (), 3: (40,)}

        def plan(state: int, remaining: int):
            return certify_constructive_plan(
                state,
                max_steps=remaining,
                propose=lambda current, budget: plans.get(current),
                transition=self.transition,
                is_accepting=self.is_accepting,
            )

        return WitnessCarryingController(
            transition=self.transition,
            plan=plan,
            max_prefix_certificates=8,
        )

    def test_wcd_always_admits_the_owned_witness_head(self):
        controller = self.witness_controller()
        initial = controller.ensure_prefix((), 0, 2)
        self.assertEqual(ReachabilityStatus.REACHABLE, initial.status)
        first = controller.certify_token(
            (), 0, 10, remaining_after_token=1
        )
        self.assertTrue(first.allowed)
        self.assertEqual(WitnessDecisionKind.CARRIED, first.kind)
        self.assertEqual((20,), first.certificate.token_ids)  # type: ignore[union-attr]
        final = controller.certify_token(
            (10,), 1, 20, remaining_after_token=0
        )
        self.assertTrue(final.allowed)
        self.assertEqual(WitnessDecisionKind.CARRIED, final.kind)
        self.assertEqual((), final.certificate.token_ids)  # type: ignore[union-attr]

    def test_wcd_accepts_deviation_only_with_replacement_certificate(self):
        controller = self.witness_controller()
        deviation = controller.certify_token(
            (), 0, 11, remaining_after_token=1
        )
        self.assertTrue(deviation.allowed)
        self.assertEqual(WitnessDecisionKind.REPLANNED, deviation.kind)
        self.assertEqual((40,), deviation.certificate.token_ids)  # type: ignore[union-attr]
        cached = controller.certify_token(
            (), 0, 11, remaining_after_token=1
        )
        self.assertEqual(WitnessDecisionKind.CACHED, cached.kind)

        impossible = controller.certify_token(
            (), 0, 11, remaining_after_token=0
        )
        self.assertFalse(impossible.allowed)
        self.assertEqual(WitnessDecisionKind.NO_CERTIFICATE, impossible.kind)

    def test_wcd_rejects_invalid_transition_without_planner_claim(self):
        decision = self.witness_controller().certify_token(
            (), 0, 99, remaining_after_token=1
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(WitnessDecisionKind.INVALID_TRANSITION, decision.kind)


if __name__ == "__main__":
    unittest.main()
