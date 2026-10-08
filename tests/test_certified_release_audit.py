from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "audit_certified_release.py"
PYTHON = REPOSITORY_ROOT / ".venv" / "bin" / "python"


class CertifiedReleaseAuditTest(unittest.TestCase):
    @staticmethod
    def row(configuration: str) -> dict[str, object]:
        return {
            "id": "example-1",
            "configuration": configuration,
            "raw_prediction": (
                "person.n.01 EQU speaker win.v.01 Agent -1 Time +1 "
                "time.n.08 TPR now"
            ),
            "validity": {"raw_sbn": True, "raw_clf_referee": True},
            "generation": {"ended_with_eos": True, "hit_max_new_tokens": False},
            "constraint": {
                "calls": 10,
                "rows": 10,
                "allowed_min": 1,
                "allowed_total": 10,
                "all_masked_rows": 0,
                "bounded_rows": 10,
                "completion_certificates_tested": 10,
                "completion_certificates_accepted": 10,
            },
        }

    def test_complete_matrix_writes_hash_bound_release_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "matrix.jsonl"
            predictions.write_text(
                "".join(
                    json.dumps(self.row(configuration)) + "\n"
                    for configuration in ("BF", "WF")
                ),
                encoding="utf-8",
            )
            digest = hashlib.sha256(predictions.read_bytes()).hexdigest()
            Path(str(predictions) + ".manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "output_sha256": digest,
                        "configurations": ["BF", "WF"],
                        "expected_examples": 1,
                        "completion_budget_factor": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "release.json"
            result = subprocess.run(
                [str(PYTHON), str(SCRIPT), str(predictions), "--output", str(output)],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            verify = subprocess.run(
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
            self.assertEqual(0, verify.returncode, verify.stderr)
        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(2, report["release"]["releasable_rows"])
        self.assertEqual(0, report["release"]["refused_rows"])
        self.assertIn("not task-semantic correctness", report["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
