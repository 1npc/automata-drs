from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.plan_adaptive_ranking_integration import build_plan


class AdaptiveRankingIntegrationPlanTest(unittest.TestCase):
    @staticmethod
    def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def fixtures(self, root: Path):
        baseline = root / "baseline.jsonl"
        failures = root / "failures.json"
        data = root / "data.sbn"
        rows = []
        for number in range(1, 13):
            failed = number in {3, 9}
            rows.append(
                {
                    "id": f"data.sbn:{number}",
                    "configuration": "U",
                    "validity": {
                        "raw_sbn": not failed,
                        "raw_clf_referee": None if failed else True,
                    },
                    "generation": {
                        "output_tokens_including_eos": number,
                        "ended_with_eos": True,
                        "hit_max_new_tokens": False,
                    },
                }
            )
        self.write_jsonl(baseline, rows)
        data.write_text("\n".join(f"row {i}" for i in range(12)) + "\n")
        failures.write_text(
            json.dumps(
                {
                    "input_sha256": self.sha256(baseline),
                    "selected": [
                        {"line_number": 3},
                        {"line_number": 9},
                    ],
                }
            )
            + "\n"
        )
        return baseline, failures, data

    def test_plan_keeps_all_failures_and_length_spread_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline, failures, data = self.fixtures(Path(directory))
            plan = build_plan(
                baseline, failures, data, controls=4
            )
        self.assertEqual(
            {
                "baseline_rows": 12,
                "u_failure_pressure": 2,
                "length_stratified_valid_controls": 4,
                "selected": 6,
                "generation_cells": 24,
            },
            plan["counts"],
        )
        selected = plan["selected"]
        failure_lines = {
            row["line_number"]
            for row in selected
            if row["group"] == "u_failure_pressure"
        }
        self.assertEqual({3, 9}, failure_lines)
        self.assertEqual(
            [1, 5, 8, 12],
            [
                row["u_output_tokens_including_eos"]
                for row in selected
                if row["group"] == "length_stratified_valid_control"
            ],
        )
        self.assertFalse(plan["selection_rule"]["randomness_used"])
        self.assertEqual(0, plan["registered_audit"]["stable_adaptive_mismatches_required"])

    def test_plan_rejects_failure_selection_from_another_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline, failures, data = self.fixtures(Path(directory))
            value = json.loads(failures.read_text())
            value["input_sha256"] = "0" * 64
            failures.write_text(json.dumps(value) + "\n")
            with self.assertRaisesRegex(ValueError, "not bound"):
                build_plan(baseline, failures, data, controls=2)


if __name__ == "__main__":
    unittest.main()
