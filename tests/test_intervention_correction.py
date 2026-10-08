from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from automata_drs.intervention_correction import (
    combine_intervention_correction_rows,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def row(identifier: str, configuration: str, prediction: str) -> dict:
    return {
        "id": identifier,
        "input": f"input-{identifier}",
        "reference": f"reference-{identifier}",
        "configuration": configuration,
        "raw_prediction": prediction,
    }


class InterventionCorrectionTest(unittest.TestCase):
    def test_join_preserves_frozen_u_and_fresh_constrained_cells(self):
        baseline = [
            row("1", "U", "frozen-u-1"),
            row("2", "U", "frozen-u-2"),
            row("3", "U", "not-selected"),
        ]
        constrained = [
            row(identifier, configuration, f"fresh-{identifier}-{configuration}")
            for identifier in ("2", "1")
            for configuration in ("B", "W", "BF", "WF")
        ]
        combined, evidence = combine_intervention_correction_rows(
            baseline, constrained, selected_ids=("1", "2")
        )
        self.assertEqual(10, len(combined))
        self.assertEqual(
            ["U", "B", "W", "BF", "WF"] * 2,
            [value["configuration"] for value in combined],
        )
        self.assertEqual("frozen-u-1", combined[0]["raw_prediction"])
        self.assertEqual("frozen-u-2", combined[5]["raw_prediction"])
        self.assertEqual(
            "cell-preserving-conditioned-intervention-v2",
            evidence["derivation"],
        )

    def test_cli_verifies_sources_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "dev-u-greedy.jsonl"
            supplement = root / "dev-u-invalid-b-w-bf-wf-greedy-v2.jsonl"
            selection = root / "dev-u-invalid-selection.json"
            output = root / "dev-u-invalid-u-b-w-bf-wf-greedy-v2.jsonl"
            baseline_rows = [row("corpus:7", "U", "frozen-invalid")]
            supplement_rows = [
                row("corpus:7", name, f"fresh-{name}")
                for name in ("B", "W", "BF", "WF")
            ]
            baseline.write_text(
                "".join(json.dumps(value) + "\n" for value in baseline_rows),
                encoding="utf-8",
            )
            supplement.write_text(
                "".join(json.dumps(value) + "\n" for value in supplement_rows),
                encoding="utf-8",
            )
            Path(str(baseline) + ".manifest.json").write_text(
                json.dumps({"status": "complete"}), encoding="utf-8"
            )
            supplement_manifest = {
                "status": "complete",
                "output_sha256": sha256(supplement.read_bytes()).hexdigest(),
                "model": "org/model",
                "requested_revision": "a" * 40,
                "line_numbers": [7],
                "max_new_tokens": 256,
                "num_beams": 1,
                "batch_size": 1,
                "configurations": ["B", "W", "BF", "WF"],
                "score_model_logprob": True,
                "schedule": "rotate",
            }
            Path(str(supplement) + ".manifest.json").write_text(
                json.dumps(supplement_manifest), encoding="utf-8"
            )
            selection.write_text(
                json.dumps(
                    {
                        "input": str(baseline),
                        "input_sha256": sha256(baseline.read_bytes()).hexdigest(),
                        "selected_records": 1,
                        "selected": [{"id": "corpus:7", "line_number": 7}],
                    }
                ),
                encoding="utf-8",
            )
            command = [
                sys.executable,
                "scripts/merge_lora_intervention_correction.py",
                "--baseline",
                str(baseline),
                "--supplement",
                str(supplement),
                "--selection",
                str(selection),
                "--output",
                str(output),
            ]
            completed = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual("frozen-invalid", rows[0]["raw_prediction"])
            self.assertEqual(
                ["U", "B", "W", "BF", "WF"],
                [value["configuration"] for value in rows],
            )
            manifest = json.loads(
                Path(str(output) + ".manifest.json").read_text()
            )
            self.assertEqual(
                "derived-lora-conditioned-intervention-v2", manifest["kind"]
            )
            self.assertEqual(
                sha256(baseline.read_bytes()).hexdigest(),
                manifest["sources"]["U"]["sha256"],
            )
            repeated = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(0, repeated.returncode)
            self.assertIn("refusing to overwrite", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
