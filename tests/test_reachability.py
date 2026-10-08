from __future__ import annotations

import unittest

from automata_drs import ReachabilityStatus, exact_bounded_reachability


class ExactBoundedReachabilityTest(unittest.TestCase):
    def test_returns_a_shortest_stably_ordered_witness(self):
        transitions = {
            (0, 1): 1,
            (0, 2): 2,
            (1, 9): 3,
            (2, 0): 3,
        }
        result = exact_bounded_reachability(
            0,
            max_steps=2,
            token_ids=lambda state: [2, 1, 1] if state == 0 else [9, 0],
            transition=lambda state, token: transitions.get((state, token)),
            is_accepting=lambda state: state == 3,
        )
        self.assertEqual(ReachabilityStatus.REACHABLE, result.status)
        self.assertEqual((1, 9), result.witness)
        self.assertEqual(2, result.distance)
        self.assertEqual(3, result.accepting_state)

    def test_unreachable_is_exact_only_for_the_requested_horizon(self):
        transitions = {(0, 1): 1, (1, 2): 2}
        too_short = exact_bounded_reachability(
            0,
            max_steps=1,
            token_ids=lambda state: [1, 2],
            transition=lambda state, token: transitions.get((state, token)),
            is_accepting=lambda state: state == 2,
        )
        self.assertEqual(ReachabilityStatus.UNREACHABLE, too_short.status)
        self.assertIsNone(too_short.witness)

        fitting = exact_bounded_reachability(
            0,
            max_steps=2,
            token_ids=lambda state: [1, 2],
            transition=lambda state, token: transitions.get((state, token)),
            is_accepting=lambda state: state == 2,
        )
        self.assertEqual(ReachabilityStatus.REACHABLE, fitting.status)
        self.assertEqual((1, 2), fitting.witness)

    def test_resource_cutoffs_return_unknown_not_unreachable(self):
        by_states = exact_bounded_reachability(
            0,
            max_steps=4,
            token_ids=lambda state: [1],
            transition=lambda state, token: state + token,
            is_accepting=lambda state: state == 4,
            max_states=1,
        )
        self.assertEqual(ReachabilityStatus.UNKNOWN, by_states.status)
        self.assertEqual("max-states", by_states.cutoff_reason)

        by_transitions = exact_bounded_reachability(
            0,
            max_steps=1,
            token_ids=lambda state: [1],
            transition=lambda state, token: state + token,
            is_accepting=lambda state: state == 1,
            max_transitions=0,
        )
        self.assertEqual(ReachabilityStatus.UNKNOWN, by_transitions.status)
        self.assertEqual("max-transitions", by_transitions.cutoff_reason)

    def test_initial_acceptance_and_zero_horizon_are_distinct(self):
        accepting = exact_bounded_reachability(
            "done",
            max_steps=0,
            token_ids=lambda state: (),
            transition=lambda state, token: None,
            is_accepting=lambda state: state == "done",
        )
        self.assertEqual(ReachabilityStatus.REACHABLE, accepting.status)
        self.assertEqual((), accepting.witness)

        open_state = exact_bounded_reachability(
            "open",
            max_steps=0,
            token_ids=lambda state: (),
            transition=lambda state, token: None,
            is_accepting=lambda state: False,
        )
        self.assertEqual(ReachabilityStatus.UNREACHABLE, open_state.status)

    def test_invalid_resource_limits_fail_before_search(self):
        arguments = {
            "initial_state": 0,
            "token_ids": lambda state: (),
            "transition": lambda state, token: None,
            "is_accepting": lambda state: False,
        }
        with self.assertRaisesRegex(ValueError, "max_steps"):
            exact_bounded_reachability(max_steps=-1, **arguments)
        with self.assertRaisesRegex(ValueError, "max_states"):
            exact_bounded_reachability(max_steps=1, max_states=0, **arguments)
        with self.assertRaisesRegex(ValueError, "max_transitions"):
            exact_bounded_reachability(
                max_steps=1, max_transitions=-1, **arguments
            )


if __name__ == "__main__":
    unittest.main()
