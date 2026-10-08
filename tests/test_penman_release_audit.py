from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "audit_penman_certified_release.py"
PYTHON = REPOSITORY_ROOT / ".venv" / "bin" / "python"


class PENMANCertifiedReleaseAuditTest(unittest.TestCase):
    def test_complete_matrix_is_recomputed_and_verifiable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "amr.jsonl"
            row = {
                "id": "example-1",
                "configuration": "P",
                "raw_prediction": "(a / alpha :ARG0 (b / beta))",
                "validity": {
                    "raw_local_penman": True,
                    "raw_official_penman": True,
                },
                "generation": {
                    "ended_with_eos": True,
                    "hit_max_new_tokens": False,
                },
                "constraint": {
                    "calls": 10,
                    "rows": 10,
                    "allowed_min": 1,
                    "allowed_total": 10,
                    "all_masked_rows": 0,
                },
            }
            predictions.write_text(json.dumps(row) + "\n", encoding="utf-8")
            digest = hashlib.sha256(predictions.read_bytes()).hexdigest()
            manifest = Path(str(predictions) + ".manifest.json")
            manifest.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "task": "amr",
                        "output_sha256": digest,
                        "configurations": ["U", "P"],
                        "expected_examples": 1,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "release.json"
            created = subprocess.run(
                [str(PYTHON), str(SCRIPT), str(predictions), "--output", str(output)],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, created.returncode, created.stderr)
            verified = subprocess.run(
                [
                    str(PYTHON),
                    str(SCRIPT),
                    str(predictions),
                    "--verify-existing",
                    str(output),
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, verified.returncode, verified.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(report["gate"]["passed"])
        self.assertTrue(report["gate"]["all_attempts_releasable"])
        self.assertEqual(2, report["schema_version"])
        self.assertEqual(1, report["release"]["releasable_rows"])
        self.assertIn("not AMR task-semantic correctness", report["claim_boundary"])

    def test_refusal_is_a_reported_outcome_not_an_integrity_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "amr.jsonl"
            rows = []
            for index in (1, 2):
                rows.append(
                    {
                        "id": f"example-{index}",
                        "configuration": "P",
                        "raw_prediction": "(a / alpha)",
                        "validity": {
                            "raw_local_penman": True,
                            "raw_official_penman": True,
                        },
                        "generation": {
                            "ended_with_eos": index == 1,
                            "hit_max_new_tokens": index == 2,
                        },
                        "constraint": {
                            "calls": 10,
                            "rows": 10,
                            "allowed_min": 1,
                            "all_masked_rows": 0,
                        },
                    }
                )
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            digest = hashlib.sha256(predictions.read_bytes()).hexdigest()
            manifest = Path(str(predictions) + ".manifest.json")
            manifest.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "task": "amr",
                        "output_sha256": digest,
                        "configurations": ["U", "P"],
                        "expected_examples": 2,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "release.json"
            created = subprocess.run(
                [str(PYTHON), str(SCRIPT), str(predictions), "--output", str(output)],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, created.returncode, created.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(report["gate"]["passed"])
        self.assertFalse(report["gate"]["all_attempts_releasable"])
        self.assertEqual(1, report["release"]["releasable_rows"])
        self.assertEqual(1, report["release"]["refused_rows"])
        self.assertEqual("example-2", report["release"]["refusals"][0]["id"])


if __name__ == "__main__":
    unittest.main()
