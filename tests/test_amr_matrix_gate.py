from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from automata_drs.generation_manifest import complete_generation_manifest, sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "check_amr_matrix.py"
PYTHON = REPOSITORY_ROOT / ".venv" / "bin" / "python"


class AMRMatrixGateTest(unittest.TestCase):
    @staticmethod
    def write(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def test_full_development_gate_binds_all_nonperformance_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "validation.amr"
            source.write_text(
                "A.\n\n(a / alpha)\n\n\nB.\n\n(b / beta)\n",
                encoding="utf-8",
            )
            config = root / "config.json"
            self.write(
                config,
                {
                    "data": {
                        "development": {
                            "records": 2,
                            "sha256": sha256_file(source),
                        },
                        "test": {"records": 3, "sha256": "0" * 64},
                    }
                },
            )
            predictions = root / "dev.jsonl"
            rows = []
            for ordinal in (1, 2):
                configurations = ("U", "P") if ordinal == 1 else ("P", "U")
                for configuration in configurations:
                    rows.append(
                        {
                            "id": f"{source}:record:{ordinal}",
                            "configuration": configuration,
                            "raw_prediction": "(a / alpha)",
                            "decode": {"strategy": "greedy", "num_beams": 1},
                            "validity": {
                                "raw_local_penman": True,
                                "raw_official_penman": True,
                            },
                            "score": {
                                "raw_model_logprob": -1.0,
                                "raw_model_mean_logprob": -0.1,
                            },
                            "generation": {
                                "ended_with_eos": True,
                                "hit_max_new_tokens": False,
                            },
                            "constraint": (
                                None
                                if configuration == "U"
                                else {
                                    "allowed_min": 1,
                                    "all_masked_rows": 0,
                                }
                            ),
                        }
                    )
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            manifest = Path(str(predictions) + ".manifest.json")
            self.write(
                manifest,
                {
                    "task": "amr",
                    "status": "running",
                    "path": str(source),
                    "source_sha256": sha256_file(source),
                    "start": 0,
                    "limit": -1,
                    "line_numbers": None,
                    "configurations": ["U", "P"],
                    "schedule": "rotate",
                    "num_beams": 1,
                    "expected_examples": 2,
                    "expected_rows": 4,
                },
            )
            complete_generation_manifest(predictions, manifest)
            scores = root / "scores.json"
            self.write(
                scores,
                {
                    "predictions": {"sha256": sha256_file(predictions)},
                    "groups": {"U": {"records": 2}, "P": {"records": 2}},
                    "gate": {"passed": True, "performance_threshold_used": False},
                },
            )
            paired = root / "paired.json"
            self.write(
                paired,
                {
                    "inputs": {
                        "predictions_sha256": sha256_file(predictions),
                        "scores_sha256": sha256_file(scores),
                    },
                    "population": {"records": 2, "paired": True},
                    "probability": {"paired_records": 2},
                    "search": {
                        "truncations": {"U": 0, "P": 0},
                        "p_all_masked_rows": 0,
                    },
                    "gate": {"passed": True, "performance_threshold_used": False},
                },
            )
            release = root / "release.json"
            self.write(
                release,
                {
                    "schema_version": 2,
                    "audit": "penman-certified-release",
                    "input": {"predictions_sha256": sha256_file(predictions)},
                    "population": {"examples": 2, "rows": 2},
                    "release": {
                        "releasable_rows": 2,
                        "refused_rows": 0,
                        "guarantee_counts": {
                            "final-local-and-independent-penman-membership": 2
                        },
                        "reason_counts": {},
                        "refusals": [],
                    },
                    "gate": {"passed": True, "performance_threshold_used": False},
                },
            )
            predecessor = root / "pilot-gate.json"
            self.write(predecessor, {"gate": {"passed": True}})
            output = root / "gate.json"
            command = [
                str(PYTHON),
                str(SCRIPT),
                str(predictions),
                "--scores",
                str(scores),
                "--paired",
                str(paired),
                "--release-audit",
                str(release),
                "--split",
                "dev",
                "--expected-examples",
                "2",
                "--predecessor-gates",
                str(predecessor),
                "--config",
                str(config),
            ]
            created = subprocess.run(
                [*command, "--output", str(output)],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                0, created.returncode, created.stderr + "\n" + created.stdout
            )
            verified = subprocess.run(
                [*command, "--verify-existing", str(output)],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, verified.returncode, verified.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))

            selective_release = json.loads(release.read_text(encoding="utf-8"))
            selective_release["release"] = {
                "releasable_rows": 1,
                "refused_rows": 1,
                "guarantee_counts": {
                    "final-local-and-independent-penman-membership": 1
                },
                "reason_counts": {"generation-did-not-end-with-eos": 1},
                "refusals": [
                    {
                        "id": f"{source}:record:2",
                        "reasons": ["generation-did-not-end-with-eos"],
                    }
                ],
            }
            self.write(release, selective_release)
            selective_output = root / "selective.json"
            selective = subprocess.run(
                [*command, "--output", str(selective_output)],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, selective.returncode, selective.stderr)
            selective_report = json.loads(
                selective_output.read_text(encoding="utf-8")
            )
            self.assertEqual(
                {"releasable_rows": 1, "refused_rows": 1},
                selective_report["outcomes"]["release"],
            )

            tampered_release = json.loads(release.read_text(encoding="utf-8"))
            tampered_release["release"]["releasable_rows"] = 2
            self.write(release, tampered_release)
            tampered = subprocess.run(
                [*command, "--output", str(root / "tampered.json")],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(0, tampered.returncode)

            self.write(release, selective_release)
            self.write(predecessor, {"gate": {"passed": False}})
            refused = subprocess.run(
                [*command, "--output", str(root / "refused.json")],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(0, refused.returncode)
        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(2, report["schema_version"])
        self.assertEqual(13, sum(report["gate"]["checks"].values()))


if __name__ == "__main__":
    unittest.main()
