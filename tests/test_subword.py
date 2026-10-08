from __future__ import annotations

import unittest

from automata_drs.sbn import (
    SBNPolicy,
    check_sbn_state_wellformedness,
    validate_sbn,
)
from automata_drs import ReachabilityStatus
from automata_drs.subword import (
    ByteLevelVocabulary,
    CausalPrefixStateCache,
    SubwordSBNLogitsProcessor,
    SubwordTokenConstraintController,
    WitnessCarryingSubwordSBNLogitsProcessor,
    gpt2_byte_decoder,
)


class SubwordConstraintTest(unittest.TestCase):
    def setUp(self):
        self.vocabulary = ByteLevelVocabulary(
            fragments={
                0: b"male",
                1: b".n.02",
                2: b" Name ",
                3: b'"Me\xe1',
                4: b"\xb9\x9b",
                5: b'"',
                6: b" Agent -2",
                7: b" Theme +1 book.n.02",
                **{100 + byte: bytes((byte,)) for byte in range(256)},
            },
            eos_token_ids=frozenset({8}),
            byte_token_ids={byte: 100 + byte for byte in range(256)},
        )
        self.controller = SubwordTokenConstraintController(self.vocabulary)

    def test_multibyte_unicode_may_span_subword_tokens(self):
        state = self.controller.transition(
            self.controller.transition(
                self.controller.transition(
                    self.controller.transition(
                        CausalPrefixStateCache(
                            self.controller,
                            prompt_token_ids=(90, 91),
                        ).state_for((90, 91)),
                        0,
                    ),
                    1,
                ),
                2,
            ),
            3,
        )
        self.assertIsNotNone(state)
        self.assertEqual(b"\xe1", state.pending_utf8)  # type: ignore[union-attr]
        state = self.controller.transition(state, 4)  # type: ignore[arg-type]
        self.assertIsNotNone(state)
        self.assertEqual("\"Meṛ", state.sbn.current_lexeme)  # type: ignore[union-attr]
        state = self.controller.transition(state, 5)  # type: ignore[arg-type]
        self.assertIsNotNone(state)
        finished = self.controller.transition(state, 8)  # type: ignore[arg-type]
        self.assertIsNotNone(finished)
        self.assertTrue(finished.is_accepting)  # type: ignore[union-attr]

    def test_token_can_cross_several_sbn_lexemes(self):
        state = CausalPrefixStateCache(
            self.controller,
            prompt_token_ids=(),
            policy=SBNPolicy.for_configuration("B"),
        ).state_for(())
        state = self.controller.transition(state, 0)  # type: ignore[arg-type]
        state = self.controller.transition(state, 1)  # type: ignore[arg-type]
        state = self.controller.transition(state, 7)  # type: ignore[arg-type]
        self.assertIsNotNone(state)
        self.assertTrue(state.sbn.can_end())  # type: ignore[union-attr]

    def test_prefix_cache_strips_prompt_and_survives_beam_reordering(self):
        cache = CausalPrefixStateCache(
            self.controller,
            prompt_token_ids=(90, 91),
        )
        state = cache.state_for((90, 91, 0, 1, 2))
        self.assertIsNotNone(state)
        self.assertEqual(1, state.sbn.concept_count)  # type: ignore[union-attr]
        shorter = cache.state_for((90, 91, 0, 1))
        self.assertIsNotNone(shorter)
        self.assertEqual(0, shorter.sbn.concept_count)  # type: ignore[union-attr]

    def test_prefix_mismatch_fails_closed(self):
        cache = CausalPrefixStateCache(
            self.controller,
            prompt_token_ids=(90, 91),
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            cache.state_for((90, 92))

    def test_eos_requires_accepting_sbn_and_pending_utf8_is_rejected(self):
        initial = CausalPrefixStateCache(
            self.controller, prompt_token_ids=()
        ).state_for(())
        self.assertIsNone(self.controller.transition(initial, 8))  # type: ignore[arg-type]
        state = initial
        for token_id in (0, 1, 2, 3):
            state = self.controller.transition(state, token_id)  # type: ignore[arg-type]
        self.assertIsNotNone(state)
        self.assertTrue(state.pending_utf8)  # type: ignore[union-attr]
        self.assertIsNone(self.controller.transition(state, 8))  # type: ignore[arg-type]

    def test_bytelevel_tokenizer_vocabulary_is_inverted_without_decoding(self):
        inverse = gpt2_byte_decoder()
        encoder = {byte: character for character, byte in inverse.items()}

        class ByteLevel:
            pass

        class Backend:
            decoder = ByteLevel()

        class Tokenizer:
            backend_tokenizer = Backend()
            all_special_ids = [1000]
            eos_token_id = 1000

            @staticmethod
            def get_vocab():
                vocabulary = {
                    encoder[byte]: byte for byte in range(256)
                }
                vocabulary["male"] = 999
                vocabulary["<eos>"] = 1000
                return vocabulary

        vocabulary = ByteLevelVocabulary.from_tokenizer(Tokenizer())
        self.assertEqual(b"male", vocabulary.fragments[999])
        self.assertEqual(bytes((0xE1,)), vocabulary.fragments[0xE1])
        self.assertEqual(0xE1, vocabulary.byte_token_ids[0xE1])
        self.assertEqual(frozenset({1000}), vocabulary.eos_token_ids)

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
                vocabulary = {
                    encoder[byte]: byte for byte in range(256)
                }
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

    @staticmethod
    def merged_boundary_tokenizer():
        """Tiny ByteLevel tokenizer with a merge spanning a repair boundary."""

        inverse = gpt2_byte_decoder()
        encoder = {byte: character for character, byte in inverse.items()}
        merged_bytes = b" entity"
        merged_symbol = "".join(encoder[byte] for byte in merged_bytes)

        class ByteLevel:
            pass

        class Backend:
            decoder = ByteLevel()

        class Tokenizer:
            backend_tokenizer = Backend()
            all_special_ids = [600]
            eos_token_id = 600

            @staticmethod
            def get_vocab():
                vocabulary = {
                    encoder[byte]: byte for byte in range(256)
                }
                vocabulary[merged_symbol] = 500
                vocabulary["<eos>"] = 600
                return vocabulary

            @staticmethod
            def __call__(text, *, add_special_tokens):
                del add_special_tokens
                raw = text.encode("utf-8")
                token_ids = []
                index = 0
                while index < len(raw):
                    if raw.startswith(merged_bytes, index):
                        token_ids.append(500)
                        index += len(merged_bytes)
                    else:
                        token_ids.append(raw[index])
                        index += 1
                return {"input_ids": token_ids}

        return Tokenizer()

    def test_score_ranked_greedy_mask_keeps_exact_constrained_argmax(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch runtime is not installed")
        tokenizer = self.fake_tokenizer()
        processor = SubwordSBNLogitsProcessor(
            tokenizer=tokenizer,
            prompt_token_ids=(399,),
            retained_tokens_per_row=1,
        )
        scores = torch.full((1, 401), -100.0)
        scores[0, 400] = 10.0  # EOS is invalid before a concept.
        scores[0, 32] = 9.0  # A leading space is invalid.
        scores[0, 300] = 8.0  # Highest-scoring valid SBN token.
        scores[0, 301] = 7.0
        masked = processor(torch.tensor([[399]]), scores)
        finite = torch.isfinite(masked[0]).nonzero().flatten().tolist()
        self.assertEqual([300], finite)

        eos_scores = torch.full((1, 401), -100.0)
        eos_scores[0, 400] = 3.0
        ended = processor(torch.tensor([[399, 300]]), eos_scores)
        finite = torch.isfinite(ended[0]).nonzero().flatten().tolist()
        self.assertEqual([400], finite)
        self.assertEqual(1, processor.stats().allowed_min)

    def test_dynamic_dead_connector_is_skipped_before_all_masked(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch runtime is not installed")
        tokenizer = self.fake_tokenizer()
        base = ByteLevelVocabulary.from_tokenizer(tokenizer)
        fragments = dict(base.fragments)
        fragments[301] = (
            b"person.n.01 EQU speaker be.v.02 Theme -1 Time +1 Co-Theme +2 "
            b"time.n.08 EQU now person.n.01 Role +1 commissar.n.01 "
            b"CONTINUATION <1 male.n.02 say.v.01 Proposition <"
        )
        vocabulary = ByteLevelVocabulary(
            fragments=fragments,
            eos_token_ids=base.eos_token_ids,
            byte_token_ids=base.byte_token_ids,
        )
        processor = SubwordSBNLogitsProcessor(
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            prompt_token_ids=(399,),
            policy=SBNPolicy.for_configuration("B"),
            retained_tokens_per_row=1,
        )
        scores = torch.full((1, 401), -100.0)
        scores[0, 301] = 10.0  # Lexically live '<', dynamically dead target.
        scores[0, 300] = 9.0   # Lower-ranked accepting SBN surface.
        masked = processor(torch.tensor([[399]]), scores)
        finite = torch.isfinite(masked[0]).nonzero().flatten().tolist()
        self.assertEqual([300], finite)
        self.assertEqual(0, processor.stats().all_masked_rows)

    def test_subword_full_horizon_certificate_counts_model_tokens(self):
        tokenizer = self.fake_tokenizer()
        enough = SubwordSBNLogitsProcessor(
            tokenizer=tokenizer,
            prompt_token_ids=(399,),
            retained_tokens_per_row=1,
            max_new_tokens=2,
            bounded_completion=True,
        )
        self.assertTrue(enough.token_is_allowed_for_prefix((399,), 300))
        too_short = SubwordSBNLogitsProcessor(
            tokenizer=tokenizer,
            prompt_token_ids=(399,),
            retained_tokens_per_row=1,
            max_new_tokens=1,
            bounded_completion=True,
        )
        self.assertFalse(too_short.token_is_allowed_for_prefix((399,), 300))

    def test_subword_planner_exposes_generic_verified_certificate(self):
        tokenizer = self.fake_tokenizer()
        processor = SubwordSBNLogitsProcessor(
            tokenizer=tokenizer,
            prompt_token_ids=(399,),
            max_new_tokens=2,
            bounded_completion=True,
        )
        state = processor.prefix_states.state_for((399, 300))
        self.assertIsNotNone(state)
        positive = processor.certificate_within(state, 1)  # type: ignore[arg-type]
        self.assertEqual(ReachabilityStatus.REACHABLE, positive.status)
        self.assertEqual(
            (tokenizer.eos_token_id,),
            positive.certificate.token_ids,  # type: ignore[union-attr]
        )
        missing_budget = processor.certificate_within(  # type: ignore[arg-type]
            state, 0
        )
        self.assertEqual(ReachabilityStatus.UNKNOWN, missing_budget.status)
        self.assertIsNone(missing_budget.certificate)

    def test_completion_compiles_surface_before_bpe_boundary_merge(self):
        tokenizer = self.merged_boundary_tokenizer()
        processor = SubwordSBNLogitsProcessor(
            tokenizer=tokenizer,
            prompt_token_ids=(),
            policy=SBNPolicy.for_configuration("W"),
            max_new_tokens=7,
            bounded_completion=True,
        )
        prefix = tokenizer(
            "person.n.01 Role +1", add_special_tokens=False
        )["input_ids"]
        state = processor.prefix_states.state_for(prefix)
        self.assertIsNotNone(state)

        # The legacy macro-by-macro encoding spends one token on the boundary
        # space and cannot exploit the vocabulary token for `` entity``.
        legacy = processor._incremental_completion_token_witness_within(
            state, 7  # type: ignore[arg-type]
        )
        self.assertIsNone(legacy)

        compiled = processor._surface_compiled_completion_token_witness_within(
            state, 7  # type: ignore[arg-type]
        )
        self.assertIsNotNone(compiled)
        self.assertEqual(7, len(compiled))  # type: ignore[arg-type]
        self.assertEqual(
            compiled,
            processor._completion_token_witness_within(  # type: ignore[arg-type]
                state, 7
            ),
        )
        certificate = processor.certificate_within(  # type: ignore[arg-type]
            state, 7
        )
        self.assertEqual(ReachabilityStatus.REACHABLE, certificate.status)
        self.assertEqual(compiled, certificate.certificate.token_ids)  # type: ignore[union-attr]

    def test_subword_wcd_replans_then_carries_eos(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch runtime is not installed")
        tokenizer = self.fake_tokenizer()
        wcd = WitnessCarryingSubwordSBNLogitsProcessor(
            tokenizer=tokenizer,
            prompt_token_ids=(399,),
            max_new_tokens=32,
            retained_tokens_per_row=1,
            max_prefix_certificates=16,
        )
        scores = torch.full((1, 401), -100.0)
        scores[0, 400] = 10.0  # Premature EOS is structurally invalid.
        scores[0, 300] = 9.0   # Deviates from the bytewise initial witness.
        first = wcd(torch.tensor([[399]]), scores)
        self.assertEqual(
            [300], torch.isfinite(first[0]).nonzero().flatten().tolist()
        )
        eos_scores = torch.full((1, 401), -100.0)
        eos_scores[0, 400] = 10.0
        ended = wcd(torch.tensor([[399, 300]]), eos_scores)
        self.assertEqual(
            [400], torch.isfinite(ended[0]).nonzero().flatten().tolist()
        )
        stats = wcd.stats()
        self.assertEqual(0, stats.all_masked_rows)
        self.assertGreaterEqual(stats.replanned_tokens, 1)
        self.assertGreaterEqual(stats.carried_tokens, 1)

    def test_subword_wcd_refuses_when_initial_constructor_misses_horizon(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch runtime is not installed")
        wcd = WitnessCarryingSubwordSBNLogitsProcessor(
            tokenizer=self.fake_tokenizer(),
            prompt_token_ids=(399,),
            max_new_tokens=2,
        )
        masked = wcd(torch.tensor([[399]]), torch.zeros((1, 401)))
        self.assertFalse(bool(torch.isfinite(masked).any().item()))
        self.assertEqual(1, wcd.stats().prefix_rows_unavailable)

    def test_adversarial_logits_cannot_release_an_invalid_full_horizon_path(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch runtime is not installed")
        tokenizer = self.fake_tokenizer()
        for configuration in ("B", "W"):
            for seed in range(8):
                with self.subTest(configuration=configuration, seed=seed):
                    processor = SubwordSBNLogitsProcessor(
                        tokenizer=tokenizer,
                        prompt_token_ids=(399,),
                        policy=SBNPolicy.for_configuration(configuration),
                        retained_tokens_per_row=1,
                        max_new_tokens=32,
                        bounded_completion=True,
                    )
                    prefix = [399]
                    for step in range(32):
                        scores = torch.tensor(
                            [
                                [
                                    float(
                                        (
                                            (token_id + 17) * (seed * 29 + 31)
                                            + step * 101
                                        )
                                        % 1009
                                    )
                                    for token_id in range(401)
                                ]
                            ]
                        )
                        masked = processor(torch.tensor([prefix]), scores)
                        finite = torch.isfinite(masked[0]).nonzero().flatten()
                        self.assertGreater(len(finite), 0)
                        chosen = int(torch.argmax(masked[0]).item())
                        prefix.append(chosen)
                        if chosen == tokenizer.eos_token_id:
                            break
                    self.assertEqual(tokenizer.eos_token_id, prefix[-1])
                    raw = b"".join(
                        processor.vocabulary.fragments[token_id]
                        for token_id in prefix[1:-1]
                    ).decode("utf-8")
                    validation = validate_sbn(
                        raw, SBNPolicy.for_configuration(configuration)
                    )
                    self.assertTrue(validation.valid, (raw, validation.reason))
                    if configuration == "W":
                        semantic = check_sbn_state_wellformedness(
                            validation.state
                        )
                        self.assertTrue(semantic.valid, (raw, semantic.reason))
                    stats = processor.stats()
                    self.assertEqual(0, stats.all_masked_rows)
                    self.assertEqual(stats.rows, stats.bounded_rows)
                    self.assertEqual(
                        stats.allowed_total,
                        stats.completion_certificates_accepted,
                    )

    def test_subword_witness_extends_a_complete_but_cyclic_reference(self):
        tokenizer = self.fake_tokenizer()
        text = " ".join(
            [
                "entity.n.01",
                *(f"entity.n.01 Theme -{index}" for index in range(1, 9)),
                "entity.n.01 Theme -9 Agent +1",
                "entity.n.01 Theme -10",
            ]
        )
        path = [*text.encode("utf-8"), tokenizer.eos_token_id]
        processor = SubwordSBNLogitsProcessor(
            tokenizer=tokenizer,
            prompt_token_ids=(),
            policy=SBNPolicy.for_configuration("W"),
            max_new_tokens=len(path),
            bounded_completion=True,
        )
        prefix: list[int] = []
        for token_id in path:
            self.assertTrue(
                processor.token_is_allowed_for_prefix(prefix, token_id),
                (len(prefix), text[: len(prefix)], text[len(prefix) :]),
            )
            prefix.append(token_id)

    def test_completion_cache_reuses_subword_distance_across_budgets(self):
        tokenizer = self.fake_tokenizer()
        processor = SubwordSBNLogitsProcessor(
            tokenizer=tokenizer,
            prompt_token_ids=(399,),
            retained_tokens_per_row=1,
            max_new_tokens=8,
            bounded_completion=True,
            max_completion_cache_entries=2,
        )
        state = processor.prefix_states.state_for((399, 300))
        self.assertIsNotNone(state)
        self.assertEqual(
            1,
            processor._completion_distance_within(state, 1),  # type: ignore[arg-type]
        )
        self.assertEqual(
            1,
            processor._completion_distance_within(state, 7),  # type: ignore[arg-type]
        )
        info = processor.completion_cache_info()
        self.assertEqual(1, info.hits)
        self.assertEqual(1, info.misses)
        self.assertEqual(1, info.size)

    def test_alternate_bpe_paths_share_one_completion_certificate(self):
        tokenizer = self.fake_tokenizer()
        processor = SubwordSBNLogitsProcessor(
            tokenizer=tokenizer,
            prompt_token_ids=(399,),
            retained_tokens_per_row=1,
            max_new_tokens=32,
            bounded_completion=True,
        )
        merged = processor.prefix_states.state_for((399, 300))
        bytewise = processor.prefix_states.state_for(
            (399, *b"male.n.02")
        )
        self.assertEqual(merged, bytewise)
        self.assertEqual(
            1,
            processor._completion_distance_within(merged, 30),  # type: ignore[arg-type]
        )
        self.assertEqual(
            1,
            processor._completion_distance_within(bytewise, 22),  # type: ignore[arg-type]
        )
        info = processor.completion_cache_info()
        self.assertEqual(1, info.misses)
        self.assertEqual(1, info.hits)
        self.assertEqual(1, info.size)

    def test_negative_completion_cache_bound_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "max_completion_cache_entries"
        ):
            SubwordSBNLogitsProcessor(
                tokenizer=self.fake_tokenizer(),
                prompt_token_ids=(399,),
                max_completion_cache_entries=-1,
            )


if __name__ == "__main__":
    unittest.main()
