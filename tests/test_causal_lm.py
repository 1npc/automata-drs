from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from automata_drs.causal_lm import (
    SBNRecord,
    build_sbn_prompt,
    encode_supervised_record,
    extract_one_line_sbn,
    iter_sbn_records,
    pad_supervised_batch,
    scheduled_generation_identities,
    select_sbn_records_by_line,
    validate_generation_resume_prefix,
)


class FakeTokenizer:
    eos_token_id = 2

    def __call__(self, text, *, add_special_tokens):
        ids = [ord(character) for character in text]
        if add_special_tokens:
            ids.insert(0, 1)
        return {"input_ids": ids}


class CausalLMDataTest(unittest.TestCase):
    def test_prompt_is_plain_and_has_a_hard_target_boundary(self):
        prompt = build_sbn_prompt("Tom left.")
        self.assertIn("Text: Tom left.", prompt)
        self.assertTrue(prompt.endswith("SBN:\n"))

    def test_supervision_masks_prompt_and_appends_eos(self):
        record = SBNRecord("dev:1", "Tom left.", "leave.v.01 Agent -1")
        encoded = encode_supervised_record(
            FakeTokenizer(), record, max_length=256
        )
        prompt_tokens = encoded["prompt_tokens"]
        self.assertEqual([-100] * prompt_tokens, encoded["labels"][:prompt_tokens])
        self.assertEqual(2, encoded["input_ids"][-1])
        self.assertEqual(encoded["input_ids"][-encoded["target_tokens"] :], encoded["labels"][-encoded["target_tokens"] :])

    def test_overlength_examples_fail_instead_of_truncating_gold(self):
        record = SBNRecord("dev:1", "text", "target")
        with self.assertRaisesRegex(ValueError, "exceeding max_length"):
            encode_supervised_record(FakeTokenizer(), record, max_length=2)

    def test_padding_masks_labels_and_attention(self):
        examples = [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [1], "attention_mask": [1], "labels": [-100]},
        ]
        batch = pad_supervised_batch(
            examples, pad_token_id=0, pad_to_multiple_of=4
        )
        self.assertEqual([1, 2, 0, 0], batch["input_ids"][0])
        self.assertEqual([1, 0, 0, 0], batch["attention_mask"][1])
        self.assertEqual([-100, -100, -100, -100], batch["labels"][1])

    def test_tsv_reader_keeps_stable_line_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.sbn"
            path.write_text("a\tx\n\nb\ty\n", encoding="utf-8")
            records = list(iter_sbn_records(path, start=1, limit=1))
        self.assertEqual(1, len(records))
        self.assertTrue(records[0].identifier.endswith(":3"))
        self.assertEqual("b", records[0].source)

    def test_generated_extraction_skips_blank_lines(self):
        self.assertEqual("dog.n.01", extract_one_line_sbn("\n dog.n.01 \nextra"))

    def test_physical_line_selection_preserves_requested_order(self):
        records = [
            SBNRecord("dev.sbn:2", "a", "x"),
            SBNRecord("dev.sbn:5", "b", "y"),
        ]
        selected = select_sbn_records_by_line(records, [5, 2])
        self.assertEqual(["dev.sbn:5", "dev.sbn:2"], [row.identifier for row in selected])

    def test_physical_line_selection_fails_closed(self):
        records = [SBNRecord("dev.sbn:2", "a", "x")]
        with self.assertRaisesRegex(ValueError, "not contain duplicates"):
            select_sbn_records_by_line(records, [2, 2])
        with self.assertRaisesRegex(ValueError, "not non-empty records: 3"):
            select_sbn_records_by_line(records, [3])

    def test_rotating_generation_schedule_has_a_unique_deterministic_order(self):
        records = [
            SBNRecord("1", "a", "x"),
            SBNRecord("2", "b", "y"),
            SBNRecord("3", "c", "z"),
        ]
        self.assertEqual(
            [
                ("1", "U"), ("1", "BF"), ("1", "WF"),
                ("2", "BF"), ("2", "WF"), ("2", "U"),
                ("3", "WF"), ("3", "U"), ("3", "BF"),
            ],
            scheduled_generation_identities(
                records, ["U", "BF", "WF"], schedule="rotate"
            ),
        )

    def test_resume_rows_must_be_an_exact_scheduled_prefix(self):
        expected = [("1", "U"), ("1", "BF"), ("2", "BF")]
        rows = [
            {
                "id": identifier,
                "configuration": configuration,
                "decode": {"strategy": "greedy", "num_beams": 1},
            }
            for identifier, configuration in expected[:2]
        ]
        self.assertEqual(
            2,
            validate_generation_resume_prefix(rows, expected, num_beams=1),
        )
        rows[1]["configuration"] = "WF"
        with self.assertRaisesRegex(ValueError, "out of schedule"):
            validate_generation_resume_prefix(rows, expected, num_beams=1)

    def test_resume_rows_reject_duplicates_and_decode_changes(self):
        expected = [("1", "U"), ("1", "BF")]
        duplicate = [
            {
                "id": "1",
                "configuration": "U",
                "decode": {"strategy": "greedy", "num_beams": 1},
            },
            {
                "id": "1",
                "configuration": "U",
                "decode": {"strategy": "greedy", "num_beams": 1},
            },
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_generation_resume_prefix(duplicate, expected, num_beams=1)
        changed = [dict(duplicate[0], decode={"strategy": "beam", "num_beams": 4})]
        with self.assertRaisesRegex(ValueError, "decoding metadata"):
            validate_generation_resume_prefix(changed, expected, num_beams=1)


if __name__ == "__main__":
    unittest.main()
