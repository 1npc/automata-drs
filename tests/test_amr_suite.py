from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from automata_drs.generation_manifest import sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "summarize_amr_suite.py"
PYTHON = Path(sys.executable)


class AMRSuiteTest(unittest.TestCase):
    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def write_matrix(
        self,
        run_root: Path,
        stem: str,
        split: str,
        records: int,
        config: Path,
        predecessors: list[Path],
        *,
        refused: int = 0,
    ) -> Path:
        predictions = run_root / f"{stem}.jsonl"
        predictions.write_text(f"{split}-predictions\n", encoding="utf-8")
        prediction_sha = sha256_file(predictions)
        scores = run_root / f"{stem}.smatch.json"
        group = lambda valid, f1: {
            "records": records,
            "raw_official_penman": valid,
            "raw_primary_smatch": {"f1": f1},
        }
        self.write_json(
            scores,
            {
                "audit": "amr-generation-smatchpp",
                "predictions": {"sha256": prediction_sha},
                "groups": {
                    "U": group(records - 1, 0.70),
                    "P": group(records, 0.71),
                },
                "gate": {
                    "passed": True,
                    "performance_threshold_used": False,
                },
            },
        )
        paired = run_root / f"{stem}.paired.json"
        self.write_json(
            paired,
            {
                "audit": "amr-paired-u-p-analysis",
                "inputs": {
                    "predictions_sha256": prediction_sha,
                    "scores_sha256": sha256_file(scores),
                },
                "population": {"records": records},
                "smatch": {
                    "u_corpus_f1": 0.70,
                    "p_corpus_f1": 0.71,
                    "delta_p_minus_u": 0.01,
                    "delta_ci_lower": 0.0,
                    "delta_ci_upper": 0.02,
                },
                "gate": {
                    "passed": True,
                    "performance_threshold_used": False,
                },
            },
        )
        release = run_root / f"{stem}.p-certified-release.json"
        released = records - refused
        refusals = [
            {
                "id": f"{split}-refusal-{index}",
                "reasons": ["generation-did-not-end-with-eos"],
            }
            for index in range(refused)
        ]
        self.write_json(
            release,
            {
                "schema_version": 2,
                "audit": "penman-certified-release",
                "input": {"predictions_sha256": prediction_sha},
                "population": {"examples": records, "rows": records},
                "release": {
                    "releasable_rows": released,
                    "refused_rows": refused,
                    "guarantee_counts": (
                        {
                            "final-local-and-independent-penman-membership": released
                        }
                        if released
                        else {}
                    ),
                    "reason_counts": (
                        {"generation-did-not-end-with-eos": refused}
                        if refused
                        else {}
                    ),
                    "refusals": refusals,
                },
                "gate": {
                    "passed": True,
                    "performance_threshold_used": False,
                },
            },
        )
        gate = run_root / f"{stem}.gate.json"
        self.write_json(
            gate,
            {
                "schema_version": 2,
                "audit": "amr-u-p-matrix-gate",
                "split": split,
                "inputs": {
                    "config_sha256": sha256_file(config),
                    "predictions_sha256": prediction_sha,
                    "scores_sha256": sha256_file(scores),
                    "paired_sha256": sha256_file(paired),
                    "release_audit_sha256": sha256_file(release),
                    "predecessor_gates": [
                        {"path": str(path), "sha256": sha256_file(path)}
                        for path in predecessors
                    ],
                },
                "population": {"examples": records},
                "outcomes": {
                    "release": {
                        "releasable_rows": released,
                        "refused_rows": refused,
                    }
                },
                "gate": {
                    "passed": True,
                    "performance_threshold_used": False,
                },
            },
        )
        return gate

    def test_complete_suite_is_verifiable_and_fails_closed_on_chain_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "run"
            run_root.mkdir()
            config = root / "config.json"
            self.write_json(
                config,
                {
                    "host": {"repo_id": "example/amr-host"},
                    "development_pilot": {"records": 1},
                    "data": {
                        "development": {"records": 2},
                        "test": {"records": 3},
                    },
                },
            )
            plan = root / "plan.json"
            self.write_json(
                plan,
                {
                    "audit": "amr-full-run-resource-plan",
                    "inputs": {"config_sha256": sha256_file(config)},
                    "gate": {
                        "passed": True,
                        "performance_threshold_used": False,
                    },
                },
            )
            training_gate = run_root / "full-training-gate.json"
            self.write_json(
                training_gate,
                {
                    "audit": "amr-full-lora-training-gate",
                    "inputs": {
                        "config_sha256": sha256_file(config),
                        "resource_plan_sha256": sha256_file(plan),
                    },
                    "observed": {"train_records": 10},
                    "gate": {
                        "passed": True,
                        "performance_threshold_used": False,
                    },
                },
            )
            pilot_gate = self.write_matrix(
                run_root,
                "dev1-u-p-greedy",
                "pilot",
                1,
                config,
                [training_gate],
            )
            dev_gate = self.write_matrix(
                run_root,
                "dev2-u-p-greedy",
                "dev",
                2,
                config,
                [pilot_gate],
                refused=1,
            )

            beam_matrix = run_root / "dev2-u-postp-p-beam4.jsonl"
            beam_matrix.write_text("beam-matrix\n", encoding="utf-8")
            beam_online = run_root / "dev2-p-beam4.jsonl"
            beam_online.write_text("beam-online\n", encoding="utf-8")
            beam_scores = run_root / "dev2-u-postp-p-beam4.smatch.json"
            self.write_json(
                beam_scores,
                {
                    "audit": "amr-generation-smatchpp",
                    "predictions": {"sha256": sha256_file(beam_matrix)},
                    "gate": {
                        "passed": True,
                        "performance_threshold_used": False,
                    },
                },
            )
            comparison = {
                "corpus_smatch_delta": 0.01,
                "delta_ci_lower": 0.0,
                "delta_ci_upper": 0.02,
                "repairs": 1,
                "regressions": 0,
                "exact_mcnemar_p": 1.0,
                "holm_adjusted_mcnemar_p": 1.0,
            }
            beam_report = run_root / "dev2-u-postp-p-beam4.analysis.json"
            self.write_json(
                beam_report,
                {
                    "schema_version": 2,
                    "audit": "amr-beam-u-postp-p-analysis",
                    "population": {"records": 2},
                    "inputs": {
                        "predictions_sha256": sha256_file(beam_matrix),
                        "scores_sha256": sha256_file(beam_scores),
                        "predecessor_gate_sha256": sha256_file(dev_gate),
                    },
                    "groups": {"U": {}, "PostP": {}, "P": {}},
                    "comparisons": {
                        "postp_minus_u": comparison,
                        "p_minus_u": comparison,
                        "p_minus_postp": comparison,
                    },
                    "selection": {"fallbacks": 0},
                    "search": {
                        "online_p_all_masked_rows": 1,
                        "truncations": {"U": 0, "PostP": 0, "P": 1},
                        "online_p_accepting_rows": 1,
                        "online_p_nonaccepting_rows": 1,
                    },
                    "gate": {
                        "passed": True,
                        "performance_threshold_used": False,
                    },
                },
            )
            beam_release = run_root / "dev2-p-beam4.p-certified-release.json"
            self.write_json(
                beam_release,
                {
                    "schema_version": 2,
                    "audit": "penman-certified-release",
                    "input": {"predictions_sha256": sha256_file(beam_online)},
                    "population": {"examples": 2, "rows": 2},
                    "release": {
                        "releasable_rows": 1,
                        "refused_rows": 1,
                        "guarantee_counts": {
                            "final-local-and-independent-penman-membership": 1
                        },
                        "reason_counts": {"generation-did-not-end-with-eos": 1},
                        "refusals": [
                            {
                                "id": "beam-refusal-1",
                                "reasons": ["generation-did-not-end-with-eos"],
                            }
                        ],
                    },
                    "gate": {
                        "passed": True,
                        "performance_threshold_used": False,
                    },
                },
            )
            test_gate = self.write_matrix(
                run_root,
                "test3-u-p-greedy",
                "test",
                3,
                config,
                [dev_gate, beam_report, beam_release],
            )
            output_json = root / "suite.json"
            output_markdown = root / "suite.md"
            command = [
                str(PYTHON),
                str(SCRIPT),
                str(run_root),
                "--resource-plan",
                str(plan),
                "--config",
                str(config),
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_markdown),
            ]
            created = subprocess.run(
                command, cwd=REPOSITORY_ROOT, text=True, capture_output=True
            )
            self.assertEqual(0, created.returncode, created.stderr)
            verified = subprocess.run(
                [*command, "--verify-existing"],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, verified.returncode, verified.stderr)
            report = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(2, report["schema_version"])
            self.assertTrue(report["certification"]["orthogonal_to_deterministic_search"])
            self.assertEqual(["pilot", "dev", "test"], [
                row["split"] for row in report["matrices"]
            ])
            self.assertIn(
                "policy-relative invariant", output_markdown.read_text(encoding="utf-8")
            )
            markdown = output_markdown.read_text(encoding="utf-8")
            self.assertIn("Release coverage is an empirical availability result", markdown)
            self.assertIn(
                "| dev | 2 | 1/2 | 2/2 | 0.7000 | 0.7100 | +0.0100 "
                "[+0.0000, +0.0200] | 1/2 (50.00%) | 100.00% |",
                markdown,
            )

            stored_beam_report = json.loads(
                beam_report.read_text(encoding="utf-8")
            )
            legacy_beam_report = dict(stored_beam_report)
            legacy_beam_report["schema_version"] = 1
            self.write_json(beam_report, legacy_beam_report)
            refused = subprocess.run(
                [*command, "--verify-existing"],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(0, refused.returncode)
            self.assertIn("beam report does not bind", refused.stderr)
            self.write_json(beam_report, stored_beam_report)

            stored_dev_gate = json.loads(dev_gate.read_text(encoding="utf-8"))
            legacy_dev_gate = dict(stored_dev_gate)
            legacy_dev_gate["schema_version"] = 1
            self.write_json(dev_gate, legacy_dev_gate)
            refused = subprocess.run(
                [*command, "--verify-existing"],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(0, refused.returncode)
            self.assertIn("matrix gate is not schema 2", refused.stderr)

            self.write_json(dev_gate, stored_dev_gate)
            drifted_dev_gate = json.loads(dev_gate.read_text(encoding="utf-8"))
            drifted_dev_gate["outcomes"]["release"]["releasable_rows"] = 2
            drifted_dev_gate["outcomes"]["release"]["refused_rows"] = 0
            self.write_json(dev_gate, drifted_dev_gate)
            refused = subprocess.run(
                [*command, "--verify-existing"],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(0, refused.returncode)
            self.assertIn("release-bound", refused.stderr)

            self.write_json(dev_gate, stored_dev_gate)
            stored_gate = json.loads(test_gate.read_text(encoding="utf-8"))
            stored_gate["inputs"]["predecessor_gates"].reverse()
            self.write_json(test_gate, stored_gate)
            refused = subprocess.run(
                [*command, "--verify-existing"],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(0, refused.returncode)
            self.assertIn("predecessor chain drifted", refused.stderr)


if __name__ == "__main__":
    unittest.main()
