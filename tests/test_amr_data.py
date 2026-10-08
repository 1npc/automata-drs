from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from automata_drs.amr_data import (
    AMRCorpusFormatError,
    iter_amr_records,
    load_amr_records,
    select_amr_records_by_ordinal,
)


class AMRCorpusReaderTest(unittest.TestCase):
    def test_pairs_by_position_and_preserves_multiline_graphs(self):
        records = list(
            iter_amr_records(
                [
                    "The boy left.\n",
                    "\n",
                    "\n",
                    "(l / leave-01\n",
                    "   :ARG0 (b / boy))\n",
                    "\n",
                    "\n",
                    "\n",
                    "( End )\n",
                    "\n",
                    "\n",
                    "(e / end-01)\n",
                ]
            )
        )

        self.assertEqual(2, len(records))
        self.assertEqual("The boy left.", records[0].sentence)
        self.assertEqual("(l / leave-01\n   :ARG0 (b / boy))", records[0].graph)
        self.assertEqual(1, records[0].sentence_start_line)
        self.assertEqual(4, records[0].graph_start_line)
        self.assertEqual("( End )", records[1].sentence)
        self.assertEqual("(e / end-01)", records[1].graph)

    def test_rejects_unpaired_trailing_sentence(self):
        with self.assertRaisesRegex(AMRCorpusFormatError, "unpaired sentence"):
            list(iter_amr_records(["sentence only\n", "\n"]))

    def test_rejects_non_penman_graph_position(self):
        with self.assertRaisesRegex(AMRCorpusFormatError, "does not begin"):
            list(
                iter_amr_records(
                    ["A sentence.\n", "\n", "not a graph\n"]
                )
            )

    def test_loads_crlf_file_without_changing_internal_surface(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.amr"
            path.write_bytes(b"Sentence\r\n\r\n(g / graph\r\n  :value 1)\r\n")
            records = load_amr_records(path)

        self.assertEqual(1, len(records))
        self.assertEqual("Sentence", records[0].sentence)
        self.assertEqual("(g / graph\n  :value 1)", records[0].graph)

    def test_ordinal_selection_preserves_request_order_and_fails_closed(self):
        records = list(
            iter_amr_records(
                ["A\n", "\n", "(a / alpha)\n", "\n", "B\n", "\n", "(b / beta)\n"]
            )
        )
        self.assertEqual(
            [2, 1],
            [record.ordinal for record in select_amr_records_by_ordinal(records, [2, 1])],
        )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            select_amr_records_by_ordinal(records, [1, 1])
        with self.assertRaisesRegex(ValueError, "missing: 3"):
            select_amr_records_by_ordinal(records, [3])


if __name__ == "__main__":
    unittest.main()
