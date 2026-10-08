from __future__ import annotations

import unittest

from automata_drs import (
    ByT5ConstraintState,
    ByT5PrefixStateCache,
    ByT5SBNLogitsProcessor,
    ByT5TokenConstraintController,
    ByT5Vocabulary,
    SBNPolicy,
)


def encode_byt5(text: str, vocabulary: ByT5Vocabulary) -> list[int]:
    return [vocabulary.token_for_byte(byte) for byte in text.encode("utf-8")]


class FakeValues(list[float]):
    def clone(self) -> "FakeValues":
        return type(self)(self)


class FakeScoreRow:
    def __init__(self, values: list[float]):
        self.values = values

    def __getitem__(self, key):
        if isinstance(key, list):
            return FakeValues(self.values[index] for index in key)
        return self.values[key]

    def __setitem__(self, key, value):
        if isinstance(key, list):
            for index, item in zip(key, value):
                self.values[index] = item
            return
        self.values[key] = value

    def fill_(self, value: float) -> None:
        self.values[:] = [value] * len(self.values)


class FakeScores:
    def __init__(self, rows: list[list[float]]):
        self.rows = [FakeScoreRow(row) for row in rows]
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, index: int) -> FakeScoreRow:
        return self.rows[index]

    def tolist(self) -> list[list[float]]:
        return [list(row.values) for row in self.rows]


class FakeInputRow(list[int]):
    def tolist(self) -> list[int]:
        return list(self)


class ByT5ConstraintTest(unittest.TestCase):
    def setUp(self):
        self.vocabulary = ByT5Vocabulary()
        self.controller = ByT5TokenConstraintController(self.vocabulary)

    def advance(self, text: str) -> ByT5ConstraintState:
        state = ByT5ConstraintState.initial()
        for token_id in encode_byt5(text, self.vocabulary):
            successor = self.controller.transition(state, token_id)
            self.assertIsNotNone(successor, (text, token_id, state))
            state = successor  # type: ignore[assignment]
        return state

    def test_checkpoint_vocabulary_layout(self):
        self.assertEqual(100, self.vocabulary.token_for_byte(ord("a")))
        self.assertEqual(0, self.vocabulary.byte_for_token(3))
        self.assertEqual(255, self.vocabulary.byte_for_token(258))
        self.assertIsNone(self.vocabulary.byte_for_token(259))
        self.assertEqual(384, self.vocabulary.vocab_size)

    def test_ascii_sbn_and_eos(self):
        state = self.advance('male.n.02 Name "Tom"')
        self.assertTrue(state.can_end())
        finished = self.controller.transition(state, self.vocabulary.eos_token_id)
        self.assertIsNotNone(finished)
        self.assertTrue(finished.is_accepting)  # type: ignore[union-attr]

    def test_unicode_scalar_can_span_byte_tokens(self):
        prefix = "caf"
        state = self.advance(prefix)
        encoded = "é".encode("utf-8")

        first = self.controller.transition(
            state, self.vocabulary.token_for_byte(encoded[0])
        )
        self.assertIsNotNone(first)
        self.assertEqual(encoded[:1], first.pending_utf8)  # type: ignore[union-attr]
        self.assertEqual(prefix, first.sbn.current_lexeme)  # type: ignore[union-attr]
        self.assertNotIn(
            self.vocabulary.eos_token_id,
            self.controller.allowed_token_ids(first),  # type: ignore[arg-type]
        )

        second = self.controller.transition(
            first, self.vocabulary.token_for_byte(encoded[1])  # type: ignore[arg-type]
        )
        self.assertIsNotNone(second)
        self.assertEqual(b"", second.pending_utf8)  # type: ignore[union-attr]
        self.assertEqual("café", second.sbn.current_lexeme)  # type: ignore[union-attr]

        state = second  # type: ignore[assignment]
        for token_id in encode_byt5(".n.01", self.vocabulary):
            state = self.controller.transition(state, token_id)  # type: ignore[assignment]
            self.assertIsNotNone(state)
        self.assertTrue(state.can_end())  # type: ignore[union-attr]

    def test_utf8_lead_is_rejected_when_no_scalar_can_complete_the_sbn(self):
        state = self.advance("male.n.02 EQU ")
        # A general term target may be a one-letter alphabetic constant, so
        # `é` starts a potentially valid target here.
        lead = self.vocabulary.token_for_byte("é".encode("utf-8")[0])
        self.assertIsNotNone(self.controller.transition(state, lead))

        proposition = self.advance("male.n.02 Proposition ")
        self.assertIsNone(self.controller.transition(proposition, lead))

        name = self.advance("male.n.02 Name ")
        self.assertIsNone(self.controller.transition(name, lead))

        separator = self.advance("NEGATION ")
        self.assertIsNone(self.controller.transition(separator, lead))

    def test_invalid_utf8_and_control_bytes_are_masked(self):
        initial = ByT5ConstraintState.initial()
        continuation = self.vocabulary.token_for_byte(0x80)
        null = self.vocabulary.token_for_byte(0)
        self.assertIsNone(self.controller.transition(initial, continuation))
        self.assertIsNone(self.controller.transition(initial, null))

    def test_special_and_sentinel_tokens_are_masked(self):
        initial = ByT5ConstraintState.initial()
        for token_id in (0, 2, 259, 383):
            self.assertIsNone(self.controller.transition(initial, token_id))

    def test_structural_candidate_shortcut_matches_exhaustive_mask(self):
        states = [
            ByT5ConstraintState.initial(),
            self.advance("male.n.02 "),
            self.advance("male.n.02 Name "),
            self.advance('male.n.02 Name "To'),
            self.advance("male.n.02 Agent "),
            self.advance("NEGATION "),
        ]
        for state in states:
            optimized = self.controller.allowed_token_ids(state)
            exhaustive = self.controller.allowed_token_ids(
                state, range(self.vocabulary.vocab_size)
            )
            self.assertEqual(exhaustive, optimized)

    def test_transition_cache_is_bounded(self):
        controller = ByT5TokenConstraintController(
            self.vocabulary, max_cache_entries=2
        )
        initial = ByT5ConstraintState.initial()
        for char in "abc":
            controller.transition(
                initial, self.vocabulary.token_for_byte(ord(char))
            )
        info = controller.cache_info()
        self.assertEqual(2, info.size)
        self.assertEqual(1, info.evictions)

    def test_prefix_cache_ignores_decoder_start_and_survives_beam_reordering(self):
        cache = ByT5PrefixStateCache(self.controller, decoder_start_token_id=0)
        male = encode_byt5("male.n.02", self.vocabulary)
        book = encode_byt5("book.n.01", self.vocabulary)

        male_state = cache.state_for([0, *male])
        book_state = cache.state_for([0, *book])
        self.assertIsNotNone(male_state)
        self.assertIsNotNone(book_state)
        self.assertEqual("male.n.02", male_state.sbn.current_lexeme)  # type: ignore[union-attr]
        self.assertEqual("book.n.01", book_state.sbn.current_lexeme)  # type: ignore[union-attr]
        self.assertIs(male_state, cache.state_for([0, *male]))

    def test_logits_processor_exposes_exact_allowed_ids(self):
        processor = ByT5SBNLogitsProcessor(vocabulary=self.vocabulary)
        start_allowed = processor.allowed_token_ids_for_prefix([0])
        self.assertIn(self.vocabulary.token_for_byte(ord("m")), start_allowed)
        self.assertNotIn(self.vocabulary.eos_token_id, start_allowed)
        self.assertNotIn(259, start_allowed)

        prefix = [0, *encode_byt5("male.n.02", self.vocabulary)]
        allowed = processor.allowed_token_ids_for_prefix(prefix)
        self.assertIn(self.vocabulary.eos_token_id, allowed)
        self.assertIn(self.vocabulary.token_for_byte(ord(" ")), allowed)

    def test_bounded_completion_requires_an_explicit_generation_budget(self):
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            ByT5SBNLogitsProcessor(
                vocabulary=self.vocabulary,
                bounded_completion_window=8,
            )
        with self.assertRaisesRegex(ValueError, "max_certified_candidates"):
            ByT5SBNLogitsProcessor(
                vocabulary=self.vocabulary,
                max_new_tokens=8,
                bounded_completion_window=8,
                max_certified_candidates=0,
            )
        with self.assertRaisesRegex(
            ValueError, "initial_max_certified_candidates"
        ):
            ByT5SBNLogitsProcessor(
                vocabulary=self.vocabulary,
                max_new_tokens=8,
                bounded_completion_window=8,
                initial_max_certified_candidates=0,
            )
        for factor in (0.0, -0.1, 1.1):
            with self.assertRaisesRegex(
                ValueError, "completion_budget_factor"
            ):
                ByT5SBNLogitsProcessor(
                    vocabulary=self.vocabulary,
                    max_new_tokens=8,
                    bounded_completion_window=8,
                    completion_budget_factor=factor,
                )
        with self.assertRaisesRegex(
            ValueError, "max_completion_cache_entries"
        ):
            ByT5SBNLogitsProcessor(
                vocabulary=self.vocabulary,
                max_completion_cache_entries=-1,
            )

    def test_completion_cache_reuses_one_distance_across_budgets(self):
        text = 'male.n.02 Name "Gabi'
        prefix = [0, *encode_byt5(text, self.vocabulary)]
        processor = ByT5SBNLogitsProcessor(
            vocabulary=self.vocabulary,
            max_new_tokens=len(prefix) - 1 + 4,
            bounded_completion_window=len(prefix) - 1 + 4,
            max_completion_cache_entries=2,
        )
        state = processor.prefix_states.state_for(prefix)
        self.assertIsNotNone(state)
        self.assertEqual(
            2,
            processor._completion_distance_within(state, 2),  # type: ignore[arg-type]
        )
        first = processor.completion_cache_info()
        self.assertEqual(1, first.misses)
        self.assertEqual(1, first.size)
        self.assertEqual(
            2,
            processor._completion_distance_within(state, 8),  # type: ignore[arg-type]
        )
        self.assertIsNone(
            processor._completion_distance_within(state, 1)  # type: ignore[arg-type]
        )
        second = processor.completion_cache_info()
        self.assertEqual(2, second.hits)
        self.assertEqual(1, second.size)

    def test_completion_cache_is_decision_equivalent_in_any_budget_order(self):
        cached = ByT5SBNLogitsProcessor(
            vocabulary=self.vocabulary,
            policy=SBNPolicy.for_configuration("W"),
            max_new_tokens=256,
            bounded_completion_window=256,
            max_completion_cache_entries=32,
        )
        uncached = ByT5SBNLogitsProcessor(
            vocabulary=self.vocabulary,
            policy=SBNPolicy.for_configuration("W"),
            max_new_tokens=256,
            bounded_completion_window=256,
            max_completion_cache_entries=0,
        )
        prefixes = (
            "",
            "male.n.02 gracefully.",
            'male.n.02 Name "Gabi',
            "male.n.02 Attrib",
            "person.n.01 Proposition >1 C",
            "time.n.08 EQU now EQU +1 time.n.08 ClockTime 0",
        )
        budgets = (32, 1, 16, 2, 8, 4, 64, 3)
        for text in prefixes:
            prefix = [0, *encode_byt5(text, self.vocabulary)]
            state = cached.prefix_states.state_for(prefix)
            self.assertIsNotNone(state, text)
            for budget in budgets:
                with self.subTest(text=text, budget=budget):
                    expected = uncached._completion_distance_within(
                        state, budget  # type: ignore[arg-type]
                    )
                    observed = cached._completion_distance_within(
                        state, budget  # type: ignore[arg-type]
                    )
                    self.assertEqual(expected, observed)

    def bounded_processor(
        self, text: str, extra_tokens: int
    ) -> tuple[ByT5SBNLogitsProcessor, list[int]]:
        prefix = [0, *encode_byt5(text, self.vocabulary)]
        processor = ByT5SBNLogitsProcessor(
            vocabulary=self.vocabulary,
            max_new_tokens=len(prefix) - 1 + extra_tokens,
            bounded_completion_window=8,
        )
        return processor, prefix

    def assert_w_path_is_certified(
        self, text: str, *, max_new_tokens: int = 256
    ) -> None:
        processor = ByT5SBNLogitsProcessor(
            vocabulary=self.vocabulary,
            policy=SBNPolicy.for_configuration("W"),
            max_new_tokens=max_new_tokens,
            bounded_completion_window=max_new_tokens,
        )
        path = [
            *encode_byt5(text, self.vocabulary),
            self.vocabulary.eos_token_id,
        ]
        self.assertLessEqual(len(path), max_new_tokens)
        prefix = [0]
        for step, token_id in enumerate(path, start=1):
            self.assertTrue(
                processor.token_is_allowed_for_prefix(prefix, token_id),
                (step, text[:step], text[step:]),
            )
            prefix.append(token_id)

    def test_bounded_completion_closes_an_open_name_before_the_budget(self):
        processor, prefix = self.bounded_processor(
            'male.n.02 Name "Gabi', 2
        )
        allowed = processor.allowed_token_ids_for_prefix(prefix)
        self.assertEqual(
            {self.vocabulary.token_for_byte(ord('"'))},
            set(allowed),
        )

        prefix.append(self.vocabulary.token_for_byte(ord('"')))
        self.assertEqual(
            {self.vocabulary.eos_token_id},
            set(processor.allowed_token_ids_for_prefix(prefix)),
        )

    def test_bounded_completion_finishes_a_partial_synset(self):
        processor, prefix = self.bounded_processor(
            "male.n.02 gracefully.", 5
        )
        allowed = processor.allowed_token_ids_for_prefix(prefix)
        expected = {
            self.vocabulary.token_for_byte(ord(pos)) for pos in "nvarx"
        }
        self.assertEqual(expected, set(allowed))

        prefix.extend(encode_byt5("n.00", self.vocabulary))
        self.assertEqual(
            {self.vocabulary.eos_token_id},
            set(processor.allowed_token_ids_for_prefix(prefix)),
        )

    def test_bounded_completion_finishes_a_partial_attachment_label(self):
        processor, prefix = self.bounded_processor("male.n.02 Attrib", 6)
        self.assertEqual(
            {self.vocabulary.token_for_byte(ord("u"))},
            set(processor.allowed_token_ids_for_prefix(prefix)),
        )
        prefix.extend(encode_byt5("ute ?", self.vocabulary))
        self.assertEqual(
            {self.vocabulary.eos_token_id},
            set(processor.allowed_token_ids_for_prefix(prefix)),
        )

    def test_bounded_logits_processor_overrides_infinite_name_repetition(self):
        processor, prefix_values = self.bounded_processor(
            'male.n.02 Name "Gabi', 4
        )
        prefix = FakeInputRow(prefix_values)
        repeated = self.vocabulary.token_for_byte(ord("r"))
        quote = self.vocabulary.token_for_byte(ord('"'))

        for _ in range(4):
            raw_scores = [-100.0] * self.vocabulary.vocab_size
            raw_scores[repeated] = 100.0
            raw_scores[quote] = 50.0
            raw_scores[self.vocabulary.eos_token_id] = 40.0
            scores = FakeScores([raw_scores])
            processor([prefix], scores)
            chosen = max(
                range(self.vocabulary.vocab_size),
                key=scores[0].values.__getitem__,
            )
            prefix.append(chosen)
            if chosen == self.vocabulary.eos_token_id:
                break

        self.assertEqual(
            [repeated, repeated, quote, self.vocabulary.eos_token_id],
            prefix[-4:],
        )
        state = processor.prefix_states.state_for(prefix)
        self.assertIsNotNone(state)
        self.assertTrue(state.is_accepting)  # type: ignore[union-attr]

    def test_completion_budget_factor_reserves_slack_before_the_hard_limit(self):
        text = 'male.n.02 Name "Gabi'
        prefix = [0, *encode_byt5(text, self.vocabulary)]
        processor = ByT5SBNLogitsProcessor(
            vocabulary=self.vocabulary,
            max_new_tokens=len(prefix) - 1 + 4,
            bounded_completion_window=len(prefix) - 1 + 4,
            completion_budget_factor=0.5,
        )
        self.assertEqual(
            {self.vocabulary.token_for_byte(ord('"'))},
            set(processor.allowed_token_ids_for_prefix(prefix)),
        )
        allowed = processor.allowed_token_ids_for_prefix(prefix)
        self.assertEqual(
            allowed,
            frozenset(
                token_id
                for token_id in range(self.vocabulary.vocab_size)
                if processor.token_is_allowed_for_prefix(prefix, token_id)
            ),
        )

    def test_completion_planner_keeps_a_required_context_separator_goal(self):
        text = "person.n.01 Proposition >1 C"
        prefix = [0, *encode_byt5(text, self.vocabulary)]
        processor = ByT5SBNLogitsProcessor(
            vocabulary=self.vocabulary,
            policy=SBNPolicy.for_configuration("B"),
            max_new_tokens=len(prefix) - 1 + 27,
            bounded_completion_window=len(prefix) - 1 + 27,
        )
        state = processor.prefix_states.state_for(prefix)
        self.assertIsNotNone(state)
        self.assertGreater(
            state.sbn.required_box_count,  # type: ignore[union-attr]
            state.sbn.box_count,  # type: ignore[union-attr]
        )
        self.assertEqual(
            "ONTINUATION",
            processor._suggest_completion_suffix(state.sbn),  # type: ignore[union-attr]
        )

        completion = "ONTINUATION <1 entity.n.00"
        prefix.extend(encode_byt5(completion, self.vocabulary))
        self.assertIn(
            self.vocabulary.eos_token_id,
            processor.allowed_token_ids_for_prefix(prefix),
        )

    def test_valid_proposition_path_is_not_removed_by_completion_planner(self):
        # This is the first divergent prefix from tax-challenge line 6. The
        # unconstrained model finishes a valid 243-token SBN, so rejecting the
        # second `p` in `Proposition` is a finite-budget false negative.
        text = "person.n.01 Role +1 doctor.n.01 say.v.01 Pro"
        prefix = [0, *encode_byt5(text, self.vocabulary)]
        processor = ByT5SBNLogitsProcessor(
            vocabulary=self.vocabulary,
            policy=SBNPolicy.for_configuration("B"),
            max_new_tokens=256,
            bounded_completion_window=256,
        )
        self.assertIn(
            self.vocabulary.token_for_byte(ord("p")),
            processor.allowed_token_ids_for_prefix(prefix),
        )

    def test_w_planner_resolves_forward_concept_before_opening_required_box(self):
        # Tax-challenge line 6 has both an outstanding Proposition box and a
        # `Time +1` reference.  Opening the box first puts the Time target in
        # the wrong context and falsely rejects this valid W path.
        text = (
            "person.n.01 Role +1 doctor.n.01 say.v.01 Proposition >1 "
            "Agent -2 Time +"
        )
        prefix = [0, *encode_byt5(text, self.vocabulary)]
        processor = ByT5SBNLogitsProcessor(
            vocabulary=self.vocabulary,
            policy=SBNPolicy.for_configuration("W"),
            max_new_tokens=256,
            bounded_completion_window=256,
        )
        self.assertIn(
            self.vocabulary.token_for_byte(ord("1")),
            processor.allowed_token_ids_for_prefix(prefix),
        )

    def test_w_planner_extends_a_complete_but_cyclic_negative_reference(self):
        # At the final target, `-1` would close the existing 9->10 edge into a
        # cycle.  The same lexical prefix has the valid completion `-10`.
        # A constructive planner must not commit the shorter interpretation.
        text = " ".join(
            [
                "entity.n.01",
                *(f"entity.n.01 Theme -{index}" for index in range(1, 9)),
                "entity.n.01 Theme -9 Agent +1",
                "entity.n.01 Theme -10",
            ]
        )
        self.assert_w_path_is_certified(text)

    def test_completion_planner_distinguishes_separators_from_attachments(self):
        processor = ByT5SBNLogitsProcessor(
            vocabulary=self.vocabulary,
            policy=SBNPolicy.for_configuration("B"),
            max_new_tokens=256,
            bounded_completion_window=256,
        )
        for text, next_char in (("", "N"), ("male.n.02 NEGATION <1 ", "P")):
            with self.subTest(text=text):
                prefix = [0, *encode_byt5(text, self.vocabulary)]
                self.assertIn(
                    self.vocabulary.token_for_byte(ord(next_char)),
                    processor.allowed_token_ids_for_prefix(prefix),
                )

    def test_completion_planner_finishes_clock_time_constant(self):
        text = "time.n.08 EQU now EQU +1 time.n.08 ClockTime 0"
        prefix = [0, *encode_byt5(text, self.vocabulary)]
        processor = ByT5SBNLogitsProcessor(
            vocabulary=self.vocabulary,
            policy=SBNPolicy.for_configuration("W"),
            max_new_tokens=256,
            bounded_completion_window=256,
        )
        self.assertIn(
            self.vocabulary.token_for_byte(ord("8")),
            processor.allowed_token_ids_for_prefix(prefix),
        )

    def test_w_planner_certifies_full_gold_boundary_regressions(self):
        # These four paths represent every failure family found by the first
        # full 9,057-row training audit: a quoted year, a forward edge across
        # CONJUNCTION, a `<0>` boundary repaired by a later Proposition edge,
        # and an exactly-256-token path whose final attachment must be short.
        paths = (
            "male.n.02 time.n.08 TPR now bear.v.02 Patient -2 Time -1 "
            "Time +1 time.n.08 YearOfCentury '198X'",
            "male.n.02 Name \"Tom\" NEGATION <1 NEGATION <1 love.v.01 "
            "Experiencer -1 Stimulus +2 Time +1 time.n.08 TPR now "
            "CONJUNCTION <2 person.n.01 SubOf +1 person.n.01 Sub speaker",
            "person.n.01 EQU hearer POSSIBILITY <1 come.v.01 Theme -1 "
            "Time +1 time.n.08 TPR now CONTINUATION <0 person.n.01 EQU "
            "speaker wish.v.01 Proposition <2 Experiencer -1 Time +1 "
            "Co-Theme +2 time.n.08 EQU now person.n.01 Sub speaker",
            "indictment.n.01 contain.v.01 Pivot -1 Time +1 Theme +3 "
            "time.n.08 EQU now quantity.n.01 EQU 875 page.n.01 Quantity -1 "
            "CONTINUATION <1 trial.n.04 time.n.08 EQU now expect.v.01 "
            "Theme -2 Time -1 Attribute +1 last.v.01 Theme -3 Duration +1 "
            "year.n.01 Quantity +",
        )
        for text in paths:
            with self.subTest(text=text):
                self.assert_w_path_is_certified(text)

    def test_score_ranked_lazy_certification_preserves_greedy_choice(self):
        text = "male.n.02 gracefully."
        prefix = FakeInputRow(
            [0, *encode_byt5(text, self.vocabulary)]
        )
        processor = ByT5SBNLogitsProcessor(
            vocabulary=self.vocabulary,
            max_new_tokens=len(prefix) - 1 + 5,
            bounded_completion_window=8,
            max_certified_candidates=1,
        )
        raw_scores = [-100.0] * self.vocabulary.vocab_size
        impossible = self.vocabulary.token_for_byte(ord("g"))
        best_certified = self.vocabulary.token_for_byte(ord("n"))
        another_certified = self.vocabulary.token_for_byte(ord("a"))
        raw_scores[impossible] = 100.0
        raw_scores[best_certified] = 90.0
        raw_scores[another_certified] = 80.0

        scores = FakeScores([raw_scores])
        processor([prefix], scores)
        chosen = max(
            range(self.vocabulary.vocab_size),
            key=scores[0].values.__getitem__,
        )
        self.assertEqual(best_certified, chosen)
        stats = processor.stats()
        self.assertEqual(2, stats.completion_certificates_tested)
        self.assertEqual(1, stats.completion_certificates_accepted)
        self.assertEqual(1, stats.certified_limit_hits)
        self.assertEqual(1, stats.allowed_max)
        self.assertEqual(4, stats.completion_witness_tokens_total)
        self.assertEqual(4, stats.completion_witness_tokens_min)
        self.assertEqual(4, stats.completion_witness_tokens_max)
        self.assertEqual(0, stats.completion_slack_total)
        self.assertEqual(0, stats.completion_slack_min)
        self.assertEqual(0, stats.completion_slack_max)

    def test_initial_beam_candidate_limit_can_exceed_later_rows(self):
        processor = ByT5SBNLogitsProcessor(
            vocabulary=self.vocabulary,
            max_new_tokens=20,
            bounded_completion_window=20,
            max_certified_candidates=1,
            initial_max_certified_candidates=2,
        )

        initial_scores = [-100.0] * self.vocabulary.vocab_size
        for score, char in ((100.0, "m"), (90.0, "p")):
            initial_scores[self.vocabulary.token_for_byte(ord(char))] = score
        first = FakeScores([initial_scores])
        processor([FakeInputRow([0])], first)
        self.assertEqual(
            2,
            sum(value != float("-inf") for value in first[0].values),
        )

        later_scores = [-100.0] * self.vocabulary.vocab_size
        for score, char in ((100.0, "a"), (90.0, "b")):
            later_scores[self.vocabulary.token_for_byte(ord(char))] = score
        later = FakeScores([later_scores])
        processor(
            [
                FakeInputRow(
                    [0, self.vocabulary.token_for_byte(ord("m"))]
                )
            ],
            later,
        )
        self.assertEqual(
            1,
            sum(value != float("-inf") for value in later[0].values),
        )

    def test_finite_budget_all_masked_row_is_counted(self):
        text = "male.n.02 gracefully."
        prefix = FakeInputRow([0, *encode_byt5(text, self.vocabulary)])
        processor = ByT5SBNLogitsProcessor(
            vocabulary=self.vocabulary,
            max_new_tokens=len(prefix),
            bounded_completion_window=len(prefix),
            max_certified_candidates=1,
        )
        scores = FakeScores([[0.0] * self.vocabulary.vocab_size])
        processor([prefix], scores)
        self.assertEqual(1, processor.stats().all_masked_rows)
        self.assertTrue(
            all(value == float("-inf") for value in scores[0].values)
        )

    def test_logits_processor_drives_an_end_to_end_greedy_decode(self):
        processor = ByT5SBNLogitsProcessor(vocabulary=self.vocabulary)
        target = encode_byt5("male.n.02", self.vocabulary)
        prefix = FakeInputRow([0])
        sentinel = 383

        for desired in [*target, self.vocabulary.eos_token_id]:
            raw_scores = [-100.0] * self.vocabulary.vocab_size
            raw_scores[sentinel] = 100.0  # Always the unconstrained winner.
            raw_scores[desired] = 50.0
            scores = FakeScores([raw_scores])
            processor([prefix], scores)
            chosen = max(
                range(self.vocabulary.vocab_size),
                key=scores[0].values.__getitem__,
            )
            self.assertEqual(desired, chosen)
            prefix.append(chosen)

        state = processor.prefix_states.state_for(prefix)
        self.assertIsNotNone(state)
        self.assertTrue(state.is_accepting)  # type: ignore[union-attr]
        decoded = bytes(token_id - 3 for token_id in prefix[1:-1]).decode("utf-8")
        self.assertEqual("male.n.02", decoded)
        stats = processor.stats()
        self.assertEqual(len(target) + 1, stats.calls)
        self.assertEqual(len(target) + 1, stats.rows)
        self.assertGreater(stats.seconds, 0.0)
        self.assertIsNotNone(stats.allowed_min)
        processor.reset_stats()
        self.assertEqual(0, processor.stats().calls)

    def test_semantic_referee_policy_reaches_the_byte_mask(self):
        structural = ByT5SBNLogitsProcessor(vocabulary=self.vocabulary)
        semantic = ByT5SBNLogitsProcessor(
            vocabulary=self.vocabulary,
            policy=SBNPolicy.semantic_referee(),
        )
        text = "entity.n.01 be.v.01 Co-Theme +1 NEGATION <1 "
        prefix = [0, *encode_byt5(text, self.vocabulary)]
        concept_start = self.vocabulary.token_for_byte(ord("m"))
        self.assertIn(concept_start, structural.allowed_token_ids_for_prefix(prefix))
        self.assertNotIn(concept_start, semantic.allowed_token_ids_for_prefix(prefix))

        disconnected = [
            0,
            *encode_byt5(
                "person.n.01 CONTINUATION <0 person.n.01",
                self.vocabulary,
            ),
        ]
        self.assertNotIn(
            self.vocabulary.eos_token_id,
            semantic.allowed_token_ids_for_prefix(disconnected),
        )


if __name__ == "__main__":
    unittest.main()
