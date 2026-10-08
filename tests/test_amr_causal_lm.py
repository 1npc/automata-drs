from __future__ import annotations

from pathlib import Path
import unittest

from automata_drs.amr_causal_lm import (
    build_amr_generation_prompt,
    build_amr_prompt,
    encode_amr_supervised_record,
    extract_first_penman_graph,
    remove_amr_output_envelope,
    iter_amr_task_records,
)
from automata_drs.amr_data import AMRCorpusRecord


class FakeTokenizer:
    eos_token_id = 2

    def __call__(self, text, *, add_special_tokens):
        ids = [ord(character) for character in text]
        if add_special_tokens:
            ids.insert(0, 1)
        return {"input_ids": ids}


class AMRCausalLMTest(unittest.TestCase):
    def test_viamr_generation_prompt_matches_published_contract(self):
        self.assertEqual(
            "<|im_start|>user\n"
            "Convert the following English sentence into its Abstract Meaning "
            "Representation (AMR):\n\n"
            "<sentence>The boy left.</sentence><|im_end|>\n"
            "<|im_start|>assistant\n",
            build_amr_generation_prompt(
                "The boy left.", template="viamr-qwen3"
            ),
        )

    def test_viamr_graph_prefill_anchors_generation_at_penman_payload(self):
        prompt = build_amr_generation_prompt(
            "The boy left.", template="viamr-qwen3-graph-prefill"
        )
        self.assertTrue(prompt.endswith("<think>\n\n</think>\n\n<amr>\n"))

    def test_viamr_envelope_removal_is_exact_and_non_repairing(self):
        self.assertEqual(
            ("(l / leave-01)", True),
            remove_amr_output_envelope(
                "(l / leave-01)\n</amr>\n", envelope="viamr-amr-tag"
            ),
        )
        self.assertEqual(
            ("prefix (l / leave-01)", False),
            remove_amr_output_envelope(
                "prefix (l / leave-01)", envelope="viamr-amr-tag"
            ),
        )

    def test_prompt_has_frozen_task_boundary(self):
        prompt = build_amr_prompt("The boy left.")
        self.assertIn("Text: The boy left.", prompt)
        self.assertTrue(prompt.endswith("AMR:\n"))

    def test_multiline_graph_is_supervised_exactly_and_gets_eos(self):
        graph = "(l / leave-01\n   :ARG0 (b / boy))"
        record = AMRCorpusRecord(7, 10, 13, "The boy left.", graph)
        encoded = encode_amr_supervised_record(
            FakeTokenizer(),
            record,
            corpus_path=Path("data/amr/train.amr"),
            max_length=512,
        )

        self.assertEqual("data/amr/train.amr:record:7", encoded["id"])
        self.assertEqual(2, encoded["input_ids"][-1])
        target_ids = [ord(character) for character in graph] + [2]
        self.assertEqual(target_ids, encoded["input_ids"][-len(target_ids) :])
        self.assertEqual(target_ids, encoded["labels"][-len(target_ids) :])
        self.assertEqual(
            [-100] * encoded["prompt_tokens"],
            encoded["labels"][: encoded["prompt_tokens"]],
        )

    def test_empty_sentence_fails_closed(self):
        record = AMRCorpusRecord(1, 1, 3, " ", "(g / graph)")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            encode_amr_supervised_record(
                FakeTokenizer(),
                record,
                corpus_path=Path("tiny.amr"),
                max_length=128,
            )

    def test_task_reader_has_stable_record_ids_and_limits(self):
        import tempfile

        corpus = "A.\n\n(a / alpha)\n\n\nB.\n\n(b / beta)\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.amr"
            path.write_text(corpus, encoding="utf-8")
            records = list(iter_amr_task_records(path, start=1, limit=1))
        self.assertEqual(1, len(records))
        self.assertTrue(records[0].identifier.endswith(":record:2"))
        self.assertEqual("B.", records[0].source)

    def test_penman_extraction_ignores_parentheses_inside_strings(self):
        text = 'Here is the graph:\n```\n(g / graph :value "a ) ( b")\n``` trailing'
        self.assertEqual(
            '(g / graph :value "a ) ( b")', extract_first_penman_graph(text)
        )
        self.assertEqual("", extract_first_penman_graph("no graph"))


if __name__ == "__main__":
    unittest.main()
