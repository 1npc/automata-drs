from __future__ import annotations

from array import array
import importlib.util
from pathlib import Path
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "train_causal_lm_lora.py"
SPEC = importlib.util.spec_from_file_location("train_causal_lm_lora_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TRAINING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINING)


class FakeTokenizer:
    eos_token_id = 2

    def __call__(self, text, *, add_special_tokens):
        ids = [ord(character) for character in text]
        if add_special_tokens:
            ids.insert(0, 1)
        return {"input_ids": ids}


class LoRATrainingDataTest(unittest.TestCase):
    def test_amr_corpus_encodes_without_flattening_or_duplicate_storage(self):
        corpus = "Sentence.\n\n\n(g / graph\n   :value 1)\n\n\n\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.amr"
            path.write_text(corpus, encoding="utf-8")
            dataset, stats = TRAINING.encode_corpus(
                FakeTokenizer(), path, task="amr", limit=-1, max_length=512
            )

        self.assertEqual(1, len(dataset))
        self.assertEqual(1, stats["records"])
        self.assertIsInstance(dataset.examples[0][0], array)
        self.assertEqual(2, len(dataset.examples[0]))
        materialized = dataset[0]
        self.assertEqual(len(materialized["input_ids"]), len(materialized["labels"]))
        self.assertEqual(2, materialized["input_ids"][-1])
        target_text = "(g / graph\n   :value 1)"
        expected_target = [ord(character) for character in target_text] + [2]
        self.assertEqual(expected_target, materialized["labels"][-len(expected_target) :])

    def test_sbn_path_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.sbn"
            path.write_text("Sentence.\tthing.n.01\n", encoding="utf-8")
            dataset, stats = TRAINING.encode_corpus(
                FakeTokenizer(), path, task="sbn", limit=-1, max_length=256
            )

        self.assertEqual(1, len(dataset))
        self.assertEqual(1, stats["records"])
        self.assertEqual(2, dataset[0]["input_ids"][-1])

    def test_amr_worst_case_selection_is_exact_and_ordered(self):
        corpus = "A.\n\n(a / alpha)\n\n\nB.\n\n(b / beta)\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.amr"
            path.write_text(corpus, encoding="utf-8")
            dataset, stats = TRAINING.encode_corpus(
                FakeTokenizer(),
                path,
                task="amr",
                limit=-1,
                max_length=512,
                record_numbers=[2],
            )
        self.assertEqual(1, len(dataset))
        self.assertEqual(1, stats["records"])
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            TRAINING.encode_corpus(
                FakeTokenizer(),
                path,
                task="amr",
                limit=1,
                max_length=512,
                record_numbers=[2],
            )


if __name__ == "__main__":
    unittest.main()
