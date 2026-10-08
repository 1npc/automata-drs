from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from automata_drs.ranking_gate import (
    COMMON_PARAMETERS,
    SOURCE_CONTRACT,
    evaluate_adaptive_ranking_benchmarks,
    verify_frozen_adaptive_ranking_benchmarks,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = REPOSITORY_ROOT / "scripts" / "check_adaptive_ranking_benchmarks.py"


class AdaptiveRankingGateTest(unittest.TestCase):
    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def report(self, required: int) -> dict[str, object]:
        vocabulary_size = COMMON_PARAMETERS["vocabulary_size"]
        peak = 128.0 if required == 1 else 2048.0
        cumulative = peak + 64.0
        materialized = 2 * cumulative
        full_seconds = 0.012 + required / 10_000
        adaptive_seconds = 0.008 + required / 10_000
        return {
            "schema_version": 4,
            "benchmark": "tie-closed-adaptive-token-ranking",
            "source_contract": SOURCE_CONTRACT,
            "scope": {
                "synthetic_scores_and_certificate_predicate": True,
                "end_to_end_generation_speed_claim": False,
                "integrated_into_frozen_publication_generator": False,
            },
            "ranking_order": {
                "primary": "descending_score",
                "tie_break": "ascending_token_id",
                "full_reference_argsort_stable": True,
                "frozen_publication_argsort_stable": False,
                "frozen_publication_equivalence_claim": False,
            },
            "parameters": {**COMMON_PARAMETERS, "required": required},
            "environment": {
                "torch": "test",
                "torch_cuda_runtime": "test",
                "device_name": "NVIDIA A40",
                "device_total_memory_bytes": 48 * 1024**3,
                "device_compute_capability": [8, 6],
                "cuda_visible_devices": "0",
                "cpu_thread_limits": {
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
            },
            "correctness": {
                "mismatches": 0,
                "examples": [],
                "selected_token_ids": list(range(required)),
            },
            "full_ranking": {
                "python_scalar_values_materialized_per_repeat": vocabulary_size,
                "materialization_contents": "token_ids",
                "certificate_tests_median": 100,
                "seconds_median": full_seconds,
                "seconds_p95": full_seconds,
            },
            "adaptive": {
                "peak_candidate_pool_median": peak,
                "peak_candidate_pool_max": int(peak),
                "cumulative_candidate_pool_median": cumulative,
                "python_scalar_values_materialized_median": materialized,
                "materialization_contents": "candidate_ids_and_scores_each_round",
                "certificate_tests_median": 100,
                "topk_rounds_median": 2,
                "boundary_ties_added_max": 0,
                "full_vocabulary_fallbacks": 0,
                "seconds_median": adaptive_seconds,
                "seconds_p95": adaptive_seconds,
            },
            "comparison": {
                "median_wall_time_speedup": full_seconds / adaptive_seconds,
                "candidate_id_transfer_reduction": (
                    1.0 - cumulative / vocabulary_size
                ),
                "python_scalar_materialization_reduction": (
                    1.0 - materialized / vocabulary_size
                ),
                "device_to_host_scalar_transfer_reduction": (
                    1.0 - materialized / vocabulary_size
                ),
                "device_to_host_transfer_applies": True,
            },
            "implementation": {
                "ranking_sha256": self.sha256(
                    REPOSITORY_ROOT / "src" / "automata_drs" / "ranking.py"
                ),
                "benchmark_sha256": self.sha256(
                    REPOSITORY_ROOT
                    / "scripts"
                    / "benchmark_adaptive_ranking.py"
                ),
                "publication_processor_sha256": self.sha256(
                    REPOSITORY_ROOT / "src" / "automata_drs" / "subword.py"
                ),
                "publication_generator_sha256": self.sha256(
                    REPOSITORY_ROOT
                    / "scripts"
                    / "generate_causal_lm_baseline.py"
                ),
            },
        }

    @staticmethod
    def write(path: Path, value: object) -> None:
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def evaluate(self, root: Path) -> dict[str, object]:
        greedy = root / "greedy.json"
        beam = root / "beam.json"
        self.write(greedy, self.report(1))
        self.write(beam, self.report(8))
        return evaluate_adaptive_ranking_benchmarks(
            (greedy, beam), repository_root=REPOSITORY_ROOT
        )

    def test_complete_cuda_cells_pass_without_performance_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.evaluate(Path(directory))
        self.assertTrue(report["passed"])
        self.assertFalse(report["performance_threshold_used"])
        self.assertEqual([1, 8], [cell["required"] for cell in report["cells"]])
        self.assertTrue(
            all(
                cell["performance_threshold_used"] is False
                for cell in report["cells"]
            )
        )

    def test_tampered_derived_metric_and_unmapped_selection_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            greedy = root / "greedy.json"
            beam = root / "beam.json"
            greedy_report = self.report(1)
            greedy_report["comparison"]["median_wall_time_speedup"] = 99.0  # type: ignore[index]
            beam_report = self.report(8)
            beam_report["correctness"]["selected_token_ids"][-1] = 248_077  # type: ignore[index]
            self.write(greedy, greedy_report)
            self.write(beam, beam_report)
            report = evaluate_adaptive_ranking_benchmarks(
                (greedy, beam), repository_root=REPOSITORY_ROOT
            )
        self.assertFalse(report["passed"])
        failed = {item["name"] for item in report["checks"] if not item["passed"]}
        self.assertIn("required1_derived_comparison", failed)
        self.assertIn("required8_ranking_equivalence", failed)

    def test_invalid_count_and_timing_quantile_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            greedy = root / "greedy.json"
            beam = root / "beam.json"
            greedy_report = self.report(1)
            greedy_report["adaptive"]["topk_rounds_median"] = 99  # type: ignore[index]
            beam_report = self.report(8)
            beam_report["adaptive"]["seconds_p95"] = 0.0  # type: ignore[index]
            self.write(greedy, greedy_report)
            self.write(beam, beam_report)
            report = evaluate_adaptive_ranking_benchmarks(
                (greedy, beam), repository_root=REPOSITORY_ROOT
            )
        self.assertFalse(report["passed"])
        failed = {item["name"] for item in report["checks"] if not item["passed"]}
        self.assertIn("required1_measurement_counts", failed)
        self.assertIn("required8_measurement_counts", failed)

    def test_frozen_publication_equivalence_cannot_be_claimed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            greedy = root / "greedy.json"
            beam = root / "beam.json"
            greedy_report = self.report(1)
            greedy_report["ranking_order"][  # type: ignore[index]
                "frozen_publication_equivalence_claim"
            ] = True
            self.write(greedy, greedy_report)
            self.write(beam, self.report(8))
            report = evaluate_adaptive_ranking_benchmarks(
                (greedy, beam), repository_root=REPOSITORY_ROOT
            )
        self.assertFalse(report["passed"])
        failed = {item["name"] for item in report["checks"] if not item["passed"]}
        self.assertIn("required1_ranking_order_boundary", failed)

    def test_frozen_gate_must_match_recomputed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.evaluate(Path(directory))
        stored = dict(report)
        stored["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        verify_frozen_adaptive_ranking_benchmarks(report, stored)
        stored["performance_threshold_used"] = True
        with self.assertRaisesRegex(ValueError, "differs"):
            verify_frozen_adaptive_ranking_benchmarks(report, stored)

    def test_cli_writes_once_and_reverifies_frozen_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            greedy = root / "greedy.json"
            beam = root / "beam.json"
            gate = root / "gate.json"
            self.write(greedy, self.report(1))
            self.write(beam, self.report(8))
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
            base = [
                sys.executable,
                str(CHECK_SCRIPT),
                "--greedy",
                str(greedy),
                "--beam",
                str(beam),
            ]
            written = subprocess.run(
                [*base, "--output", str(gate)],
                cwd=REPOSITORY_ROOT,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, written.returncode, written.stderr)
            frozen = json.loads(gate.read_text(encoding="utf-8"))
            self.assertTrue(frozen["passed"])
            self.assertIsInstance(frozen["checked_at_utc"], str)
            verified = subprocess.run(
                [*base, "--verify-existing", str(gate)],
                cwd=REPOSITORY_ROOT,
                env=environment,
                text=True,
                capture_output=True,
            )
        self.assertEqual(0, verified.returncode, verified.stderr)
        self.assertIn("verified adaptive-ranking", verified.stdout)

    def test_cli_failure_does_not_write_partial_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            greedy = root / "greedy.json"
            beam = root / "beam.json"
            gate = root / "gate.json"
            self.write(greedy, self.report(1))
            invalid = self.report(8)
            invalid["correctness"]["mismatches"] = 1  # type: ignore[index]
            self.write(beam, invalid)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(CHECK_SCRIPT),
                    "--greedy",
                    str(greedy),
                    "--beam",
                    str(beam),
                    "--output",
                    str(gate),
                ],
                cwd=REPOSITORY_ROOT,
                env=environment,
                text=True,
                capture_output=True,
            )
            gate_exists = gate.exists()
        self.assertNotEqual(0, completed.returncode)
        self.assertIn("ranking benchmark checks failed", completed.stderr)
        self.assertFalse(gate_exists)


if __name__ == "__main__":
    unittest.main()
