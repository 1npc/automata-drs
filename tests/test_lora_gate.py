from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from automata_drs.lora_gate import SMOKE_CONFIGURATIONS, evaluate_lora_smoke_gate


class LoRASmokeGateTest(unittest.TestCase):
    model_row = {
        "key": "model",
        "repo_id": "example/model",
        "revision": "abc123",
    }

    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def make_run(self, directory: str) -> Path:
        root = Path(directory) / "model-smoke"
        root.mkdir()
        self.write_json(
            Path(str(root) + ".tokenizer-audit.json"),
            {
                "requested_revision": "abc123",
                "target_roundtrip_failure_count": 0,
                "lengths": {"total_max": 100},
            },
        )
        self.write_json(
            root / "train_manifest.json",
            {
                "status": "complete",
                "requested_revision": "abc123",
                "recipe": {"max_steps": 8},
            },
        )
        self.write_json(
            root / "train_metrics.json",
            {"train": {"train_loss": 1.0}, "eval": {"eval_loss": 2.0}},
        )
        adapter = root / "adapter" / "adapter_model.safetensors"
        adapter.parent.mkdir()
        adapter.write_bytes(b"adapter")
        prediction = root / "dev20-u-b-w-bf-wf-greedy.jsonl"
        rows = []
        for identifier in range(20):
            for configuration in SMOKE_CONFIGURATIONS:
                rows.append(
                    {
                        "id": str(identifier),
                        "configuration": configuration,
                        "validity": {
                            "raw_sbn": True,
                            "clf_referee": configuration != "BF",
                        },
                        "generation": {
                            "hit_max_new_tokens": False,
                            "canonical_tokenization": True,
                        },
                        "constraint": (
                            None
                            if configuration == "U"
                            else {"all_masked_rows": 0}
                        ),
                    }
                )
        prediction.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        self.write_json(
            Path(str(prediction) + ".manifest.json"),
            {"requested_revision": "abc123"},
        )
        return root

    def test_complete_smoke_run_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = self.make_run(directory)
            report = evaluate_lora_smoke_gate(
                self.model_row, run_root, max_length=1024
            )
        self.assertTrue(report["passed"])
        self.assertEqual(1.0, report["diagnostics"]["canonical_tokenization_rate"])

    def test_roundtrip_failure_and_all_masked_row_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = self.make_run(directory)
            audit_path = Path(str(run_root) + ".tokenizer-audit.json")
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["target_roundtrip_failure_count"] = 1
            self.write_json(audit_path, audit)
            prediction = run_root / "dev20-u-b-w-bf-wf-greedy.jsonl"
            rows = [json.loads(line) for line in prediction.read_text().splitlines()]
            rows[1]["constraint"]["all_masked_rows"] = 1
            prediction.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = evaluate_lora_smoke_gate(
                self.model_row, run_root, max_length=1024
            )
        self.assertFalse(report["passed"])
        failed = {item["name"] for item in report["checks"] if not item["passed"]}
        self.assertEqual({"target_roundtrip_failures", "all_masked_rows"}, failed)

    def test_bounded_wf_gate_uses_raw_referee_before_postprocessed_value(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = self.make_run(directory)
            prediction = run_root / "dev20-u-b-w-bf-wf-greedy.jsonl"
            rows = [json.loads(line) for line in prediction.read_text().splitlines()]
            wf_row = next(row for row in rows if row["configuration"] == "WF")
            wf_row["validity"]["raw_clf_referee"] = False
            wf_row["validity"]["clf_referee"] = True
            prediction.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = evaluate_lora_smoke_gate(
                self.model_row, run_root, max_length=1024
            )
        failed = {item["name"] for item in report["checks"] if not item["passed"]}
        self.assertIn("bounded_outputs_valid", failed)

    def test_missing_artifact_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "missing"
            report = evaluate_lora_smoke_gate(
                self.model_row, run_root, max_length=1024
            )
        self.assertFalse(report["passed"])
        self.assertTrue(report["missing_artifacts"])

    def test_smaller_model_specific_smoke_matrix_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = self.make_run(directory)
            prediction = run_root / "dev20-u-b-w-bf-wf-greedy.jsonl"
            rows = [json.loads(line) for line in prediction.read_text().splitlines()]
            kept_ids = {"0", "1", "2", "3"}
            rows = [row for row in rows if row["id"] in kept_ids]
            prediction.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            model_row = {**self.model_row, "smoke_records": 4}
            report = evaluate_lora_smoke_gate(
                model_row, run_root, max_length=1024
            )
        self.assertTrue(report["passed"])
        self.assertEqual(4, report["diagnostics"]["smoke_records"])


if __name__ == "__main__":
    unittest.main()
