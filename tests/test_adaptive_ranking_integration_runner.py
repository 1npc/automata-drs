from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_adaptive_ranking_integration import complete_manifest


class AdaptiveRankingIntegrationRunnerTest(unittest.TestCase):
    @staticmethod
    def write_rows(path: Path, count: int) -> None:
        path.write_text(
            "".join(json.dumps({"row": index}) + "\n" for index in range(count)),
            encoding="utf-8",
        )

    def test_completion_is_immutable_and_reverifiable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "audit.jsonl"
            manifest_path = root / "audit.jsonl.manifest.json"
            self.write_rows(output, 2)
            manifest = {
                "schema_version": 1,
                "kind": "adaptive-ranking-real-logit-integration-audit",
                "status": "running",
                "contract": "test",
            }
            manifest_path.write_text(json.dumps(manifest) + "\n")
            complete_manifest(
                manifest_path, output, manifest, expected_rows=2
            )
            frozen = manifest_path.read_text()
            complete_manifest(
                manifest_path, output, manifest, expected_rows=2
            )
            self.assertEqual(frozen, manifest_path.read_text())

            output.write_text(output.read_text() + json.dumps({"row": 3}) + "\n")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                complete_manifest(
                    manifest_path, output, manifest, expected_rows=2
                )

    def test_completion_refuses_changed_same_length_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "audit.jsonl"
            manifest_path = root / "audit.jsonl.manifest.json"
            self.write_rows(output, 1)
            manifest = {"status": "running", "contract": "test"}
            manifest_path.write_text(json.dumps(manifest) + "\n")
            complete_manifest(
                manifest_path, output, manifest, expected_rows=1
            )
            output.write_text(json.dumps({"row": "tampered"}) + "\n")
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                complete_manifest(
                    manifest_path, output, manifest, expected_rows=1
                )


if __name__ == "__main__":
    unittest.main()
