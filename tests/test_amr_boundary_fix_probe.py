from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "check_amr_boundary_fix_probe.py"
DIVERGENCE_AUDIT = (
    REPOSITORY_ROOT / "scripts" / "audit_amr_boundary_divergence.py"
)


def _row(configuration: str, identifier: str, prediction: str) -> dict:
    return {
        "configuration": configuration,
        "id": identifier,
        "prediction": prediction,
        "validity": {"raw_official_penman": True},
        "generation": {
            "ended_with_structural_completion": configuration != "U",
            "hit_max_new_tokens": False,
        },
    }


def _score(configuration: str, f1: float) -> dict:
    return {
        "gate": {"passed": True},
        "groups": {configuration: {"raw_primary_smatch": {"f1": f1}}},
    }


class AMRBoundaryFixProbeTest(unittest.TestCase):
    def test_exact_preservation_gate_passes_and_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "u.jsonl"
            constrained = root / "p.jsonl"
            finite = root / "pf.jsonl"
            baseline.write_text(
                json.dumps(_row("U", "r1", "(a / alpha)")) + "\n",
                encoding="utf-8",
            )
            constrained.write_text(
                json.dumps(_row("P", "r1", "(a / alpha)")) + "\n",
                encoding="utf-8",
            )
            finite.write_text(
                json.dumps(_row("PF", "r1", "(a / alpha)")) + "\n",
                encoding="utf-8",
            )
            score_paths = {}
            for configuration in ("U", "P", "PF"):
                path = root / f"{configuration}.json"
                path.write_text(
                    json.dumps(_score(configuration, 0.75)), encoding="utf-8"
                )
                score_paths[configuration] = path
            output = root / "gate.json"
            command = [
                sys.executable,
                str(SCRIPT),
                "--baseline",
                str(baseline),
                "--constrained",
                str(constrained),
                "--finite-horizon",
                str(finite),
                "--baseline-score",
                str(score_paths["U"]),
                "--constrained-score",
                str(score_paths["P"]),
                "--finite-horizon-score",
                str(score_paths["PF"]),
                "--expected-records",
                "1",
                "--output",
                str(output),
            ]
            passed = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(0, passed.returncode, passed.stderr)
            self.assertTrue(json.loads(output.read_text())["gate"]["passed"])

            constrained.write_text(
                json.dumps(_row("P", "r1", "(b / beta)")) + "\n",
                encoding="utf-8",
            )
            command[-1] = str(root / "failed-gate.json")
            failed = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(1, failed.returncode, failed.stderr)

    def test_script_syntax(self):
        for script in (SCRIPT, DIVERGENCE_AUDIT):
            with self.subTest(script=script.name):
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(script)],
                    cwd=REPOSITORY_ROOT,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
