from __future__ import annotations

from collections import namedtuple
from pathlib import Path
import tempfile
import unittest

from automata_drs.penman import (
    audit_penman_corpus,
    iter_penman_records,
    sha256_file,
)


Triple = namedtuple("Triple", "source role target")


class FakeGraph:
    def __init__(self, normalized: str, edges=(), variables=()):
        self.normalized = normalized
        self._edges = tuple(edges)
        self._variables = tuple(variables)

    def edges(self):
        return self._edges

    def variables(self):
        return self._variables


class PENMANCorpusAuditTest(unittest.TestCase):
    def test_records_preserve_graph_whitespace_but_strip_transport_comments(self):
        records = list(
            iter_penman_records(
                [
                    "# ::id graph-1\n",
                    "# an ordinary comment\n",
                    "(a / alpha\n",
                    "   :ARG0 (b / beta))\n",
                    "\n",
                    "# ::id graph-2\n",
                    "(c / gamma)\n",
                ]
            )
        )
        self.assertEqual(2, len(records))
        self.assertEqual("graph-1", records[0].identifier)
        self.assertEqual(1, records[0].start_line)
        self.assertEqual("(a / alpha\n   :ARG0 (b / beta))", records[0].surface)
        self.assertEqual("graph-2", records[1].identifier)

    def test_audit_separates_raw_surface_from_oracle_normalization(self):
        corpus = """# ::id aligned
(a / alpha~e.1)

# ::id reentrant
(w / want-01 :ARG0 (b / boy) :ARG1 (g / go-01 :ARG0 b))

# ::id broken
BAD
"""

        def decode(surface: str):
            if surface == "BAD":
                raise ValueError("independent parse failure\nwith location")
            if "~e.1" in surface:
                return FakeGraph("(a / alpha)", variables=("a",))
            return FakeGraph(
                surface,
                edges=(
                    Triple("w", ":ARG0", "b"),
                    Triple("w", ":ARG1", "g"),
                    Triple("g", ":ARG0", "b"),
                ),
                variables=("w", "b", "g"),
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.txt"
            path.write_text(corpus, encoding="utf-8")
            report = audit_penman_corpus(
                path,
                decode=decode,
                encode=lambda graph: graph.normalized,
                oracle_name="fake-penman",
                oracle_version="1.0",
                source={"revision": "abc123"},
                example_limit=2,
            )

        self.assertEqual(3, report["corpus"]["records"])
        self.assertEqual(2, report["official_decode"]["accepted"])
        self.assertEqual(1, report["local_raw_on_official"]["accepted"])
        self.assertEqual(2, report["local_normalized_on_official"]["accepted"])
        self.assertEqual(
            1,
            report["feature_coverage"]["surface_alignment"][
                "officially_decoded_records"
            ],
        )
        self.assertEqual(
            1,
            report["feature_coverage"]["reentrancy"][
                "officially_decoded_records"
            ],
        )
        self.assertEqual(
            "independent parse failure",
            next(iter(report["official_decode"]["reasons"])),
        )
        self.assertEqual(
            "local_raw_on_official",
            report["interpretation"]["primary_false_rejection_measure"],
        )

    def test_hash_is_over_exact_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bytes.txt"
            path.write_bytes(b"abc")
            self.assertEqual(
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                sha256_file(path),
            )


if __name__ == "__main__":
    unittest.main()
