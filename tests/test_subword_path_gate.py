from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from automata_drs.subword_path_gate import (
    evaluate_selected_subword_path_audit,
    verify_frozen_selected_subword_path_audit,
)


class SelectedSubwordPathGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "host_selection": {"required_records": 2},
            "method_experiment": {
                "pilot": {"max_new_tokens": 8},
                "publication_matrices": [
                    {"split": "test", "records": 2}
                ],
            },
        }
        self.model = {"key": "model", "repo_id": "org/model", "revision": "r"}

    @staticmethod
    def audit() -> dict[str, object]:
        def report(configuration: str) -> dict[str, object]:
            return {
                "configuration": configuration,
                "records": 2,
                "eligible": 2,
                "accepted": 2,
                "false_rejection_count": 0,
                "false_rejection_examples": [],
                "peak_prefix_cache_entries": 5,
                "transition_cache": {"size": 20},
                "completion_cache": (
                    {"size": 0, "misses": 0}
                    if configuration == "W"
                    else {"size": 12, "misses": 13}
                ),
            }

        return {
            "path": "dev.sbn",
            "tokenizer": "runs/model-r16-seed42/adapter",
            "max_new_tokens": 8,
            "expected_records": 2,
            "raw_byte_fallbacks": 256,
            "reports": [report("W"), report("WF")],
            "gate": {"require_zero_false_rejections": True, "passed": True},
        }

    def make_report(self, root: Path) -> tuple[dict[str, object], Path]:
        corpus = root / "dev.sbn"
        corpus.write_text("a\tb\n", encoding="utf-8")
        audit_path = root / "audit.json"
        audit = self.audit()
        import hashlib

        audit["path"] = str(corpus)
        audit["input_sha256"] = hashlib.sha256(corpus.read_bytes()).hexdigest()
        audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")
        report = evaluate_selected_subword_path_audit(
            self.config,
            self.model,
            split="dev",
            corpus_path=corpus,
            audit_path=audit_path,
            runs_root=Path("runs"),
        )
        return report, audit_path

    def test_complete_zero_rejection_audit_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _ = self.make_report(Path(directory))
        self.assertTrue(report["passed"])

    def test_wf_without_certificate_activity_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, audit_path = self.make_report(root)
            self.assertTrue(report["passed"])
            audit = json.loads(audit_path.read_text())
            audit["reports"][1]["completion_cache"] = {"size": 0, "misses": 0}
            audit_path.write_text(json.dumps(audit) + "\n")
            report = evaluate_selected_subword_path_audit(
                self.config,
                self.model,
                split="dev",
                corpus_path=root / "dev.sbn",
                audit_path=audit_path,
                runs_root=Path("runs"),
            )
        check = next(item for item in report["checks"] if item["name"] == "wf_certificate_planner_activated")
        self.assertFalse(check["passed"])

    def test_test_split_uses_publication_population_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "test.sbn"
            corpus.write_text("a\tb\n", encoding="utf-8")
            audit_path = root / "audit.json"
            audit = self.audit()
            import hashlib

            audit["path"] = str(corpus)
            audit["input_sha256"] = hashlib.sha256(corpus.read_bytes()).hexdigest()
            audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")
            report = evaluate_selected_subword_path_audit(
                self.config,
                self.model,
                split="test",
                corpus_path=corpus,
                audit_path=audit_path,
                runs_root=Path("runs"),
            )
        self.assertTrue(report["passed"])
        self.assertEqual("test", report["split"])

    def test_train_split_uses_training_manifest_population_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "train.sbn"
            corpus.write_text("a\tb\n", encoding="utf-8")
            runs_root = root / "runs"
            run_root = runs_root / "model-r16-seed42"
            run_root.mkdir(parents=True)
            training_manifest = {
                "status": "complete",
                "train_path": str(corpus),
                "train": {"records": 2},
            }
            manifest_path = run_root / "train_manifest.json"
            manifest_path.write_text(
                json.dumps(training_manifest) + "\n", encoding="utf-8"
            )
            audit_path = root / "audit.json"
            audit = self.audit()
            import hashlib

            audit["path"] = str(corpus)
            audit["input_sha256"] = hashlib.sha256(corpus.read_bytes()).hexdigest()
            audit["tokenizer"] = str(run_root / "adapter")
            audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")
            report = evaluate_selected_subword_path_audit(
                self.config,
                self.model,
                split="train",
                corpus_path=corpus,
                audit_path=audit_path,
                runs_root=runs_root,
            )
            self.assertTrue(report["passed"])
            self.assertEqual("train", report["split"])
            self.assertEqual(
                str(manifest_path),
                report["evidence"]["training_manifest_path"],
            )
            self.assertEqual(
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                report["evidence"]["training_manifest_sha256"],
            )

    def test_stored_gate_detects_audit_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _ = self.make_report(Path(directory))
            stored = json.loads(json.dumps(report))
            stored.update(
                {
                    "checked_at_utc": "2026-08-26T00:00:00+00:00",
                    "config_sha256": "c" * 64,
                    "selection_sha256": "s" * 64,
                }
            )
            verify_frozen_selected_subword_path_audit(
                report,
                stored,
                config_sha256="c" * 64,
                selection_sha256="s" * 64,
            )
            stored["evidence"]["audit_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "differs from current evidence"):
                verify_frozen_selected_subword_path_audit(
                    report,
                    stored,
                    config_sha256="c" * 64,
                    selection_sha256="s" * 64,
                )


if __name__ == "__main__":
    unittest.main()
