from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from automata_drs.causal_lm import (
    iter_sbn_records,
    scheduled_generation_identities,
    select_sbn_records_by_line,
)
from automata_drs.ranking_integration_gate import (
    MODEL,
    REVISION,
    THREAD_LIMITS,
    evaluate_adaptive_ranking_integration,
    freeze_adaptive_ranking_integration,
    verify_frozen_adaptive_ranking_integration,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SELECTION = (
    REPOSITORY_ROOT
    / "runs"
    / "lora"
    / "qwen35-4b-base-r16-seed42"
    / "dev-adaptive-ranking-integration-selection.json"
)


class RankingIntegrationGateTest(unittest.TestCase):
    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def implementation(self) -> dict[str, str]:
        return {
            "audit_runner_sha256": self.sha256(
                REPOSITORY_ROOT
                / "scripts"
                / "audit_adaptive_ranking_integration.py"
            ),
            "adaptive_subword_sha256": self.sha256(
                REPOSITORY_ROOT
                / "src"
                / "automata_drs"
                / "adaptive_subword.py"
            ),
            "ranking_sha256": self.sha256(
                REPOSITORY_ROOT / "src" / "automata_drs" / "ranking.py"
            ),
            "frozen_subword_sha256": self.sha256(
                REPOSITORY_ROOT / "src" / "automata_drs" / "subword.py"
            ),
        }

    def make_cell(
        self,
        root: Path,
        adapter: Path,
        *,
        beams: int,
        tamper_frozen_tie: bool = False,
    ) -> Path:
        plan = json.loads(SELECTION.read_text(encoding="utf-8"))
        data = Path(plan["data"])
        line_numbers = [item["line_number"] for item in plan["selected"]]
        records = select_sbn_records_by_line(
            iter_sbn_records(data), line_numbers
        )
        expected = scheduled_generation_identities(
            records, ("BF", "WF"), schedule="rotate"
        )
        rows = []
        for index, (identifier, configuration) in enumerate(expected):
            processor = {
                "calls": 1,
                "rows": 1,
                "allowed_total": 1,
                "allowed_min": 1,
                "allowed_max": 1,
                "all_masked_rows": 0,
                "candidates_tested": 5,
            }
            rows.append(
                {
                    "id": identifier,
                    "configuration": configuration,
                    "decode": {
                        "strategy": "greedy" if beams == 1 else "beam",
                        "num_beams": beams,
                    },
                    "raw_prediction": "male.n.02",
                    "surface_token_ids": [300, 400],
                    "validity": {"sbn": True, "clf_referee": True},
                    "equivalence": {
                        "calls": 1,
                        "rows": 1,
                        "frozen_stable_mismatches": (
                            1 if tamper_frozen_tie and index == 0 else 0
                        ),
                        "stable_adaptive_mismatches": 0,
                        "frozen_stable_examples": (
                            [{"call_index": 0}] if tamper_frozen_tie and index == 0 else []
                        ),
                    },
                    "processors": {
                        "frozen": processor,
                        "stable": processor,
                        "adaptive": processor,
                        "adaptive_ranking": {
                            "ranked_rows": 1,
                            "cumulative_candidate_pool": 8,
                            "python_scalar_values_materialized": 16,
                            "certificate_tests": 5,
                        },
                    },
                    "runtime": {
                        "cuda_device_name": "NVIDIA A40",
                        "cuda_device_total_bytes": 48 * 1024**3,
                    },
                }
            )
        output = root / ("greedy.jsonl" if beams == 1 else "beam.jsonl")
        output.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        adapter_hashes = {
            path.name: self.sha256(path)
            for path in adapter.iterdir()
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "kind": "adaptive-ranking-real-logit-integration-audit",
            "status": "complete",
            "selection": str(SELECTION),
            "selection_sha256": self.sha256(SELECTION),
            "data": plan["data"],
            "data_sha256": plan["data_sha256"],
            "model": MODEL,
            "revision": REVISION,
            "model_loader": "multimodal-text-only",
            "adapter": str(adapter),
            "adapter_hashes": adapter_hashes,
            "num_beams": beams,
            "configurations": ["BF", "WF"],
            "schedule": "rotate",
            "max_input_tokens": 512,
            "max_new_tokens": 256,
            "completion_budget_factor": 1.0,
            "adaptive_initial_k": 32,
            "expected_examples": 32,
            "expected_rows": 64,
            "same_logits_three_way_shadow": True,
            "returned_generation_path": "frozen_unstable_full_sort",
            "performance_threshold_used": False,
            "physical_gpu_index": "0",
            "cpu_thread_limits": THREAD_LIMITS,
            "implementation": self.implementation(),
            "rows": 64,
            "output_sha256": self.sha256(output),
            "completed_at_utc": "2026-08-26T00:00:00+00:00",
        }
        Path(str(output) + ".manifest.json").write_text(
            json.dumps(manifest) + "\n", encoding="utf-8"
        )
        return output

    def evaluate(self, root: Path, *, tamper: bool = False):
        adapter = root / "adapter"
        adapter.mkdir()
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            (adapter / name).write_text(name + "\n", encoding="utf-8")
        greedy = self.make_cell(
            root, adapter, beams=1, tamper_frozen_tie=tamper
        )
        beam = self.make_cell(root, adapter, beams=4)
        return evaluate_adaptive_ranking_integration(
            (greedy, beam),
            selection_path=SELECTION,
            repository_root=REPOSITORY_ROOT,
        )

    def test_complete_two_cell_audit_supports_drop_in_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.evaluate(Path(directory))
        self.assertTrue(report["passed"])
        self.assertTrue(report["drop_in_optimization_supported"])
        self.assertFalse(report["performance_threshold_used"])

    def test_one_frozen_stable_tie_difference_fails_drop_in_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.evaluate(Path(directory), tamper=True)
        self.assertFalse(report["passed"])
        self.assertFalse(report["drop_in_optimization_supported"])
        failed = {item["name"] for item in report["checks"] if not item["passed"]}
        self.assertIn("greedy_same_logit_equivalence", failed)

    def test_frozen_report_is_reverified_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.evaluate(Path(directory))
        stored = freeze_adaptive_ranking_integration(report)
        verify_frozen_adaptive_ranking_integration(report, stored)
        stored["drop_in_optimization_supported"] = False
        with self.assertRaisesRegex(ValueError, "differs"):
            verify_frozen_adaptive_ranking_integration(report, stored)


if __name__ == "__main__":
    unittest.main()
