from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_wcd_smoke.py"


def constraint(**overrides: int) -> dict[str, int]:
    value = {
        "rows": 2,
        "all_masked_rows": 0,
        "prefix_rows_certified": 2,
        "prefix_rows_unavailable": 0,
        "carried_tokens": 1,
        "cached_tokens": 0,
        "replanned_tokens": 1,
        "rejected_without_certificate": 0,
        "allowed_total": 2,
    }
    value.update(overrides)
    return value


class WCDSmokeGateTest(unittest.TestCase):
    def run_gate(
        self,
        rows: list[dict[str, object]],
        *,
        with_sbn_scores: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        predictions = root / "predictions.jsonl"
        predictions.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        digest = hashlib.sha256(predictions.read_bytes()).hexdigest()
        Path(str(predictions) + ".manifest.json").write_text(
            json.dumps({
                "status": "complete",
                "output_sha256": digest,
                "rows": len(rows),
                "configurations": ["WWCD", "WCD"],
            }) + "\n",
            encoding="utf-8",
        )
        command = [
                sys.executable,
                str(SCRIPT),
                str(predictions),
                "--configurations",
                "WWCD",
                "WCD",
                "--expected-records-per-configuration",
                "1",
        ]
        if with_sbn_scores:
            scores = root / "scores.jsonl"
            scores.write_text(
                "".join(
                    json.dumps({
                        "id": row["id"],
                        "configuration": row["configuration"],
                        "semantic_scores": {"sbn_smatch": 0.75},
                    }) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            Path(str(scores) + ".manifest.json").write_text(
                json.dumps({
                    "input_sha256": digest,
                    "output_sha256": hashlib.sha256(scores.read_bytes()).hexdigest(),
                    "records": len(rows),
                    "scoring_protocol": "seeded-deduplicated-semantic-pairs-v2",
                    "seed": 42,
                    "identical_pair_score_consistency": True,
                    "scorer_sha256": "frozen-scorer",
                }) + "\n",
                encoding="utf-8",
            )
            command.extend([
                "--sbn-scores",
                str(scores),
                "--expected-sbn-scorer-sha256",
                "frozen-scorer",
            ])
        command.extend(["--output", str(root / "gate.json")])
        return subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

    @staticmethod
    def valid_rows() -> list[dict[str, object]]:
        return [
            {
                "id": "d1",
                "configuration": "WWCD",
                "validity": {"raw_sbn": True, "raw_clf_referee": True},
                "constraint": constraint(),
            },
            {
                "task": "amr",
                "id": "a1",
                "configuration": "WCD",
                "validity": {
                    "raw_local_penman": True,
                    "raw_official_penman": True,
                },
                "constraint": constraint(),
            },
        ]

    def test_cross_representation_evidence_passes(self):
        result = self.run_gate(self.valid_rows())
        self.assertEqual(0, result.returncode, result.stderr)

    def test_seeded_sbn_scores_are_bound(self):
        result = self.run_gate(self.valid_rows(), with_sbn_scores=True)
        self.assertEqual(0, result.returncode, result.stderr)

    def test_unknown_or_all_masked_prefix_fails(self):
        rows = self.valid_rows()
        rows[0]["constraint"] = constraint(
            all_masked_rows=1,
            prefix_rows_certified=1,
            prefix_rows_unavailable=1,
        )
        result = self.run_gate(rows)
        self.assertNotEqual(0, result.returncode)
        self.assertIn("all-masked", result.stderr)
        self.assertIn("prefix-unavailable", result.stderr)

    def test_incomplete_decision_accounting_fails(self):
        rows = self.valid_rows()
        rows[1]["constraint"] = constraint(allowed_total=3)
        result = self.run_gate(rows)
        self.assertNotEqual(0, result.returncode)
        self.assertIn("decision-accounting", result.stderr)


if __name__ == "__main__":
    unittest.main()
