from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import json

from automata_drs import ByteLevelVocabulary, validate_penman


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_amr_posthoc_repair.py"


class _ByteTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False):
        assert not add_special_tokens
        return {"input_ids": list(text.encode("utf-8"))}


class AMRPosthocRepairTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("amr_posthoc_repair", SCRIPT)
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)
        cls.tokenizer = _ByteTokenizer()
        cls.vocabulary = ByteLevelVocabulary(
            fragments={value: bytes((value,)) for value in range(256)},
            eos_token_ids=frozenset({256}),
            byte_token_ids={value: value for value in range(256)},
        )

    def test_budget_matched_repair_closes_without_borrowing_tokens(self):
        raw = "(a / alpha :ARG0 (b / beta"
        repair = self.module.repair_within_token_budget(
            raw,
            tokenizer=self.tokenizer,
            vocabulary=self.vocabulary,
            horizon=len(raw.encode("utf-8")),
        )
        self.assertIsNotNone(repair)
        assert repair is not None
        self.assertTrue(validate_penman(repair.surface).valid)
        self.assertLessEqual(repair.output_tokens_including_eos, repair.horizon)
        self.assertGreater(repair.dropped_tokens, 0)

    def test_valid_short_graph_is_kept_exactly(self):
        raw = "(a / alpha)"
        repair = self.module.repair_within_token_budget(
            raw,
            tokenizer=self.tokenizer,
            vocabulary=self.vocabulary,
            horizon=len(raw.encode("utf-8")) + 1,
        )
        self.assertIsNotNone(repair)
        assert repair is not None
        self.assertEqual(raw, repair.surface)
        self.assertEqual(0, repair.dropped_tokens)
        self.assertEqual(1, len(repair.completion_token_ids))

    def test_script_syntax(self):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(SCRIPT)],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_corpus_override_requires_frozen_hash_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "validation.amr"
            corpus.write_text("sentence\n\n(a / alpha)\n", encoding="utf-8")
            manifest = root / "predictions.jsonl.manifest.json"
            manifest.write_text(
                json.dumps({"source_sha256": "0" * 64}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                self.module._complete_manifest(
                    root / "predictions.jsonl",
                    manifest,
                    corpus_override=corpus,
                )

    def test_corpus_override_validates_frozen_hashes_without_rewriting_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "validation.amr"
            corpus.write_text("sentence\n\n(a / alpha)\n", encoding="utf-8")
            predictions = root / "predictions.jsonl"
            predictions.write_text('{"id":"/workspace/corpus:record:1"}\n', encoding="utf-8")
            original = {
                "path": "/workspace/missing.amr",
                "source_sha256": self.module.sha256_file(corpus),
                "output_sha256": self.module.sha256_file(predictions),
                "status": "complete",
                "rows": 1,
                "expected_rows": 1,
            }
            manifest = root / "predictions.jsonl.manifest.json"
            manifest.write_text(json.dumps(original), encoding="utf-8")
            observed = self.module._complete_manifest(
                predictions,
                manifest,
                corpus_override=corpus,
            )
            self.assertEqual(original, observed)
            self.assertEqual(original, json.loads(manifest.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
