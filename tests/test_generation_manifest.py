from __future__ import annotations

from hashlib import sha256
import json
import tempfile
import unittest
from pathlib import Path

from automata_drs.generation_manifest import complete_generation_manifest


class GenerationManifestTest(unittest.TestCase):
    def write_fixture(self, root: Path, *, rows: int = 4) -> tuple[Path, Path]:
        corpus = root / "dev.txt"
        corpus.write_text("a\tx\nb\ty\n", encoding="utf-8")
        predictions = root / "matrix.jsonl"
        schedule = [(f"{corpus}:1", "U"), (f"{corpus}:1", "WF"),
                    (f"{corpus}:2", "WF"), (f"{corpus}:2", "U")]
        values = [
            {
                "id": identifier,
                "configuration": configuration,
                "decode": {"strategy": "greedy", "num_beams": 1},
            }
            for identifier, configuration in schedule[:rows]
        ]
        predictions.write_text(
            "".join(json.dumps(value) + "\n" for value in values),
            encoding="utf-8",
        )
        manifest = Path(str(predictions) + ".manifest.json")
        manifest.write_text(
            json.dumps(
                {
                    "path": str(corpus),
                    "start": 0,
                    "limit": -1,
                    "line_numbers": None,
                    "configurations": ["U", "WF"],
                    "schedule": "rotate",
                    "num_beams": 1,
                }
            ),
            encoding="utf-8",
        )
        return predictions, manifest

    def test_legacy_full_matrix_is_promoted_with_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions, manifest_path = self.write_fixture(Path(directory))
            completed = complete_generation_manifest(predictions)
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("complete", completed["status"])
            self.assertEqual(2, completed["examples"])
            self.assertEqual(4, completed["rows"])
            self.assertEqual(
                sha256(predictions.read_bytes()).hexdigest(),
                completed["output_sha256"],
            )
            self.assertEqual(completed, stored)

    def test_running_matrix_is_promoted_to_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions, manifest_path = self.write_fixture(Path(directory))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "running"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            completed = complete_generation_manifest(predictions)
            self.assertEqual("complete", completed["status"])

    def test_incomplete_matrix_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions, manifest_path = self.write_fixture(Path(directory), rows=3)
            before = manifest_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "incomplete: 3/4"):
                complete_generation_manifest(predictions)
            self.assertEqual(before, manifest_path.read_bytes())

    def test_completed_manifest_is_immutable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions, manifest_path = self.write_fixture(Path(directory))
            first = complete_generation_manifest(predictions)
            before = manifest_path.read_bytes()
            self.assertEqual(first, complete_generation_manifest(predictions))
            self.assertEqual(before, manifest_path.read_bytes())
            with predictions.open("a", encoding="utf-8") as sink:
                sink.write("{}\n")
            with self.assertRaisesRegex(ValueError, "only 4 are expected"):
                complete_generation_manifest(predictions)

    def test_amr_manifest_reconstructs_record_ordinals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "dev.amr"
            corpus.write_text(
                "A.\n\n(a / alpha)\n\n\nB.\n\n(b / beta)\n",
                encoding="utf-8",
            )
            predictions = root / "amr.jsonl"
            rows = [
                {
                    "id": f"{corpus}:record:{ordinal}",
                    "configuration": configuration,
                    "decode": {"strategy": "greedy", "num_beams": 1},
                }
                for ordinal in (1, 2)
                for configuration in ("U", "P")
            ]
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            manifest_path = Path(str(predictions) + ".manifest.json")
            manifest_path.write_text(
                json.dumps(
                    {
                        "task": "amr",
                        "path": str(corpus),
                        "start": 0,
                        "limit": -1,
                        "line_numbers": None,
                        "configurations": ["U", "P"],
                        "schedule": "fixed",
                        "num_beams": 1,
                    }
                ),
                encoding="utf-8",
            )
            completed = complete_generation_manifest(predictions)

        self.assertEqual(2, completed["examples"])
        self.assertEqual(4, completed["rows"])
