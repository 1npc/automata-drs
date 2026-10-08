from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "benchmark_adaptive_ranking.py"


class AdaptiveRankingBenchmarkTest(unittest.TestCase):
    def test_cpu_report_separates_materialization_from_host_transfer(self):
        try:
            import torch  # noqa: F401
        except ImportError as error:
            self.skipTest(f"optional Torch runtime is unavailable: {error}")

        environment = os.environ.copy()
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        environment.update(
            {
                "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--vocabulary-size",
                "64",
                "--mapped-token-count",
                "61",
                "--required",
                "2",
                "--initial-k",
                "4",
                "--certificate-modulus",
                "2",
                "--certificate-residues",
                "1",
                "--repeats",
                "1",
                "--warmup",
                "0",
                "--source-model",
                "example/model",
                "--source-revision",
                "a" * 40,
                "--source-config-sha256",
                "b" * 64,
                "--source-tokenizer-sha256",
                "c" * 64,
                "--source-tokenizer-config-sha256",
                "d" * 64,
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        report = json.loads(completed.stdout)
        full = report["full_ranking"]
        adaptive = report["adaptive"]
        comparison = report["comparison"]

        self.assertEqual(4, report["schema_version"])
        self.assertEqual(0, report["correctness"]["mismatches"])
        self.assertIsNone(report["environment"]["device_name"])
        self.assertIsNone(report["environment"]["device_total_memory_bytes"])
        self.assertIsNone(report["environment"]["device_compute_capability"])
        self.assertEqual(
            {
                "model": "example/model",
                "revision": "a" * 40,
                "config_sha256": "b" * 64,
                "tokenizer_sha256": "c" * 64,
                "tokenizer_config_sha256": "d" * 64,
            },
            report["source_contract"],
        )
        self.assertEqual(61, report["parameters"]["mapped_token_count"])
        self.assertEqual(3, report["parameters"]["unmapped_logit_positions"])
        self.assertTrue(
            all(
                token_id < 61
                for token_id in report["correctness"]["selected_token_ids"]
            )
        )
        self.assertIsNone(report["environment"]["cuda_visible_devices"])
        self.assertEqual(
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            report["environment"]["cpu_thread_limits"],
        )
        self.assertEqual(64, full["python_scalar_values_materialized_per_repeat"])
        self.assertEqual(
            2 * adaptive["cumulative_candidate_pool_median"],
            adaptive["python_scalar_values_materialized_median"],
        )
        self.assertFalse(comparison["device_to_host_transfer_applies"])
        self.assertIsNone(
            comparison["device_to_host_scalar_transfer_reduction"]
        )
        expected_reduction = (
            1.0
            - adaptive["python_scalar_values_materialized_median"] / 64
        )
        self.assertAlmostEqual(
            expected_reduction,
            comparison["python_scalar_materialization_reduction"],
        )
        self.assertFalse(
            report["scope"]["end_to_end_generation_speed_claim"]
        )
        self.assertEqual(
            {
                "primary": "descending_score",
                "tie_break": "ascending_token_id",
                "full_reference_argsort_stable": True,
                "frozen_publication_argsort_stable": False,
                "frozen_publication_equivalence_claim": False,
            },
            report["ranking_order"],
        )


if __name__ == "__main__":
    unittest.main()
