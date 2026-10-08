from __future__ import annotations

import unittest

from automata_drs import (
    ByteLevelFiniteHorizonPlanner,
    ByteLevelTokenConstraintController,
    ByteLevelVocabulary,
    PENMANConstraintState,
    PENMANMode,
    ReachabilityStatus,
    construct_penman_completion_suffix,
)


def byte_vocabulary(
    extra: dict[int, bytes] | None = None,
) -> ByteLevelVocabulary:
    fragments = {byte: bytes((byte,)) for byte in range(256)}
    fragments.update(extra or {})
    return ByteLevelVocabulary(
        fragments=fragments,
        eos_token_ids=frozenset({256}),
        byte_token_ids={byte: byte for byte in range(256)},
    )


class PENMANCompletionPlannerTest(unittest.TestCase):
    def test_surface_constructor_replays_every_parser_mode(self):
        prefixes = {
            PENMANMode.START: "",
            PENMANMode.NODE_VARIABLE: "(",
            PENMANMode.NODE_SLASH: "(x ",
            PENMANMode.NODE_CONCEPT: "(x / thi",
            PENMANMode.NODE_CONCEPT_STRING: '(x / "thi',
            PENMANMode.NODE_BODY: "(x / thing ",
            PENMANMode.ROLE: "(x / thing :AR",
            PENMANMode.TARGET: "(x / thing :ARG ",
            PENMANMode.TARGET_SYMBOL: "(x / thing :ARG y",
            PENMANMode.TARGET_STRING: '(x / thing :name "Ada',
            PENMANMode.COMPLETE: "(x / thing)",
        }
        for expected_mode, prefix in prefixes.items():
            with self.subTest(mode=expected_mode.value):
                state = PENMANConstraintState.initial().advance_text(prefix)
                self.assertEqual(expected_mode, state.mode)
                suffix = construct_penman_completion_suffix(state)
                self.assertIsNotNone(suffix)
                finished = state.advance_text(suffix or "").finish()
                self.assertTrue(finished.is_accepting, finished.dead_reason)

    def test_nested_string_escape_closes_every_open_node(self):
        prefix = '(w / want-01 :ARG0 (p / person :name "A\\'
        state = PENMANConstraintState.initial().advance_text(prefix)
        self.assertTrue(state.string_escape)
        suffix = construct_penman_completion_suffix(state)
        self.assertIsNotNone(suffix)
        self.assertTrue(state.advance_text(suffix or "").finish().is_accepting)

    def test_operationally_open_duplicate_slash_state_has_no_witness(self):
        state = PENMANConstraintState.initial().advance_text(
            "(x / thing :ARG0 (x "
        )
        self.assertEqual(PENMANMode.NODE_SLASH, state.mode)
        self.assertTrue(state.is_extendable)
        self.assertIsNone(construct_penman_completion_suffix(state))

    def test_exact_model_token_budget_includes_eos(self):
        vocabulary = byte_vocabulary()
        controller = ByteLevelTokenConstraintController(
            vocabulary,
            incomplete_utf8_is_extendable=(
                PENMANConstraintState.incomplete_utf8_is_extendable
            ),
        )
        planner = ByteLevelFiniteHorizonPlanner(
            controller,
            construct_surface_suffix=construct_penman_completion_suffix,
        )
        prefix = PENMANConstraintState.initial().advance_text(
            "(w / want-01 :ARG0"
        )
        state = controller.initial(prefix)
        loose = planner.completion_within(state, 100)
        self.assertIsNotNone(loose)
        assert loose is not None
        self.assertTrue(loose.final_state.is_accepting)
        exact = planner.completion_within(state, loose.distance)
        self.assertEqual(loose.token_ids, exact.token_ids)  # type: ignore[union-attr]
        self.assertIsNone(planner.completion_within(state, loose.distance - 1))
        self.assertEqual(256, loose.token_ids[-1])

    def test_exact_encoder_is_used_and_inexact_encoder_falls_back_to_bytes(self):
        state = PENMANConstraintState.initial().advance_text("(x / thing :ARG")
        suffix = construct_penman_completion_suffix(state)
        assert suffix is not None
        compressed_id = 300
        vocabulary = byte_vocabulary({compressed_id: suffix.encode("utf-8")})
        controller = ByteLevelTokenConstraintController(vocabulary)
        lifted = controller.initial(state)
        compressed = ByteLevelFiniteHorizonPlanner(
            controller,
            construct_surface_suffix=construct_penman_completion_suffix,
            encode_surface=lambda text: (compressed_id,),
        ).completion_within(lifted, 2)
        self.assertEqual((compressed_id, 256), compressed.token_ids)  # type: ignore[union-attr]

        fallback = ByteLevelFiniteHorizonPlanner(
            controller,
            construct_surface_suffix=construct_penman_completion_suffix,
            encode_surface=lambda text: (ord("!"),),
        ).completion_within(lifted, len(suffix.encode("utf-8")) + 1)
        self.assertIsNotNone(fallback)
        self.assertNotIn(compressed_id, fallback.token_ids)  # type: ignore[union-attr]

    def test_pending_utf8_scalar_is_completed_before_surface_planning(self):
        vocabulary = byte_vocabulary()
        controller = ByteLevelTokenConstraintController(
            vocabulary,
            incomplete_utf8_is_extendable=(
                PENMANConstraintState.incomplete_utf8_is_extendable
            ),
        )
        representation = PENMANConstraintState.initial().advance_text(
            '(p / person :name "'
        )
        partial = controller.transition(controller.initial(representation), 0xE5)
        self.assertEqual(b"\xe5", partial.pending_utf8)  # type: ignore[union-attr]
        planner = ByteLevelFiniteHorizonPlanner(
            controller,
            construct_surface_suffix=construct_penman_completion_suffix,
        )
        witness = planner.completion_within(partial, 100)  # type: ignore[arg-type]
        self.assertIsNotNone(witness)
        assert witness is not None
        self.assertTrue(witness.final_state.is_accepting)
        self.assertTrue(witness.appended_surface_bytes.startswith(b"\x80\x80"))

    def test_invalid_constructor_or_dead_state_never_produces_a_positive(self):
        vocabulary = byte_vocabulary()
        controller = ByteLevelTokenConstraintController(vocabulary)
        initial = controller.initial(PENMANConstraintState.initial())
        invalid = ByteLevelFiniteHorizonPlanner(
            controller,
            construct_surface_suffix=lambda state: "]",
        )
        self.assertIsNone(invalid.completion_within(initial, 100))
        dead = PENMANConstraintState.initial().advance_text("]")
        self.assertIsNone(construct_penman_completion_suffix(dead))

    def test_generic_certificate_bridge_preserves_positive_and_unknown(self):
        vocabulary = byte_vocabulary()
        controller = ByteLevelTokenConstraintController(vocabulary)
        initial = controller.initial(PENMANConstraintState.initial())
        planner = ByteLevelFiniteHorizonPlanner(
            controller,
            construct_surface_suffix=construct_penman_completion_suffix,
        )
        positive = planner.certificate_within(initial, 100)
        self.assertEqual(ReachabilityStatus.REACHABLE, positive.status)
        self.assertIsNotNone(positive.certificate)
        self.assertTrue(positive.certificate.final_state.is_accepting)  # type: ignore[union-attr]

        invalid = ByteLevelFiniteHorizonPlanner(
            controller,
            construct_surface_suffix=lambda state: "]",
        ).certificate_within(initial, 100)
        self.assertEqual(ReachabilityStatus.UNKNOWN, invalid.status)
        self.assertIsNone(invalid.certificate)


if __name__ == "__main__":
    unittest.main()
