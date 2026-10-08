from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_drs_horizon_generation.py"
CONFIG = ROOT / "configs" / "wcd_drs_horizon_test128_v1.json"
CONFIG_SURFACE_V2 = ROOT / "configs/wcd_drs_horizon_test128_surface_v2.json"
RUNNER = ROOT / "scripts" / "run_colossus_drs_horizon_test128_v1.sh"
RUNNER_SURFACE_V2 = (
    ROOT / "scripts/run_colossus_drs_horizon_test128_surface_v2.sh"
)
SPEC = importlib.util.spec_from_file_location("check_drs_horizon_generation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DRSStressGenerationGateTest(unittest.TestCase):
    def test_registered_runner_is_resource_bounded_and_hash_bound(self):
        contract = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(48, contract["generation"]["max_new_tokens"])
        self.assertEqual(128, contract["data"]["limit"])
        self.assertFalse(contract["data"]["selection_performance_conditioned"])
        self.assertFalse(
            contract["budget_selection"][
                "heldout_outcomes_observed_before_registration"
            ]
        )
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn(hashlib.sha256(CONFIG.read_bytes()).hexdigest(), source)
        self.assertIn("--limit 128", source)
        self.assertIn("--max-new-tokens 48", source)
        self.assertIn("--configurations W WF WWCD", source)
        self.assertIn("OMP_NUM_THREADS=2", source)
        self.assertIn("automata_gpu_select_idle", source)
        self.assertIn("required_headroom_bytes", source)
        self.assertIn("automata_scratch_available_bytes", source)
        self.assertNotIn("rm -rf", source)
        completed = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual("", completed.stderr)
        self.assertEqual(0, completed.returncode)

    def test_surface_v2_uses_a_fixed_unconditioned_offset_window(self):
        contract = json.loads(CONFIG_SURFACE_V2.read_text(encoding="utf-8"))
        self.assertEqual(
            "fixed-128-source-order-window", contract["data"]["selection"]
        )
        self.assertEqual(128, contract["data"]["start"])
        self.assertEqual(128, contract["data"]["limit"])
        self.assertFalse(contract["data"]["selection_performance_conditioned"])
        self.assertFalse(
            contract["budget_selection"][
                "heldout_window_outcomes_observed_before_registration"
            ]
        )
        self.assertEqual(
            "fa57a6516378f2b48a4d0fe1f04c6529e447e80079fd4cd71742e493df826d76",
            contract["method"]["implementation_sha256"],
        )
        self.assertFalse(contract["method"]["trusted_verifier_changed"])

        source = RUNNER_SURFACE_V2.read_text(encoding="utf-8")
        self.assertIn(hashlib.sha256(CONFIG_SURFACE_V2.read_bytes()).hexdigest(), source)
        self.assertIn("--start 128 --limit 128", source)
        self.assertIn("--max-new-tokens 48", source)
        self.assertIn("--configurations W WF WWCD", source)
        self.assertIn("OMP_NUM_THREADS=2", source)
        self.assertIn("automata_gpu_select_idle", source)
        self.assertIn("automata_scratch_available_bytes", source)
        self.assertNotIn("rm -rf", source)
        completed = subprocess.run(
            ["bash", "-n", str(RUNNER_SURFACE_V2)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual("", completed.stderr)
        self.assertEqual(0, completed.returncode)

    def test_complete_paired_matrix_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            rows = []
            for ordinal in range(128):
                for configuration in ("W", "WF", "WWCD"):
                    constraint = {
                        "rows": 1,
                        "all_masked_rows": 0,
                    }
                    if configuration == "WF":
                        constraint.update(
                            completion_certificates_tested=1,
                            completion_certificates_accepted=1,
                        )
                    if configuration == "WWCD":
                        constraint.update(
                            prefix_rows_certified=1,
                            prefix_rows_unavailable=0,
                            carried_tokens=0,
                            cached_tokens=0,
                            replanned_tokens=1,
                            rejected_without_certificate=0,
                            allowed_total=1,
                        )
                    rows.append(
                        {
                            "id": f"test:{ordinal + 1}",
                            "configuration": configuration,
                            "prediction": "male.n.02",
                            "validity": {
                                "raw_sbn": True,
                                "raw_clf_referee": True,
                            },
                            "generation": {
                                "ended_with_eos": True,
                                "hit_max_new_tokens": False,
                                "output_tokens_including_eos": 10,
                            },
                            "timing": {
                                "generation_ms": 10.0,
                                "constraint_ms": 1.0,
                            },
                            "constraint": constraint,
                        }
                    )
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            contract = json.loads(CONFIG.read_text(encoding="utf-8"))
            manifest = {
                "status": "complete",
                "model": contract["model"]["repo_id"],
                "requested_revision": contract["model"]["revision"],
                "resolved_revision": contract["model"]["revision"],
                "start": 0,
                "limit": 128,
                "max_input_tokens": 512,
                "max_new_tokens": 48,
                "num_beams": 1,
                "batch_size": 1,
                "configurations": ["W", "WF", "WWCD"],
                "schedule": "rotate",
                "expected_examples": 128,
                "expected_rows": 384,
                "examples": 128,
                "rows": 384,
                "output_sha256": sha(predictions),
                "physical_gpu_index": "0",
            }
            Path(str(predictions) + ".manifest.json").write_text(
                json.dumps(manifest) + "\n", encoding="utf-8"
            )

            scores = root / "scores.jsonl"
            score_rows = [
                {
                    "id": row["id"],
                    "configuration": row["configuration"],
                    "semantic_scores": {"sbn_smatch": 1.0},
                }
                for row in rows
            ]
            scores.write_text(
                "".join(json.dumps(row) + "\n" for row in score_rows),
                encoding="utf-8",
            )
            score_manifest = {
                "input_sha256": sha(predictions),
                "output_sha256": sha(scores),
                "records": 384,
                "scoring_protocol": contract["scoring"]["protocol"],
                "seed": 42,
                "scorer_sha256": contract["scoring"]["scorer_sha256"],
                "identical_pair_score_consistency": True,
            }
            Path(str(scores) + ".manifest.json").write_text(
                json.dumps(score_manifest) + "\n", encoding="utf-8"
            )
            report = MODULE.build_report(
                predictions, config_path=CONFIG, score_path=scores
            )
            self.assertTrue(report["gate"]["passed"])
            self.assertEqual(
                128, report["paired"]["WWCD-WF"]["prediction_exact"]
            )
            self.assertEqual(
                128, report["summaries"]["WWCD"]["prefix_rows_certified"]
            )


if __name__ == "__main__":
    unittest.main()
