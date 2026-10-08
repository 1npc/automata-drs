from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from automata_drs.penman import sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "check_amr_preflight.py"
PYTHON = REPOSITORY_ROOT / ".venv" / "bin" / "python"
CONFIG = REPOSITORY_ROOT / "configs" / "amr_transfer_experiment.json"


class AMRPreflightGateTest(unittest.TestCase):
    def make_evidence(self, root: Path) -> tuple[Path, Path]:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        train_expected = config["data"]["train"]
        dev_expected = config["data"]["development"]
        reader_hash = sha256_file(
            REPOSITORY_ROOT / "src" / "automata_drs" / "amr_data.py"
        )
        state_hash = sha256_file(
            REPOSITORY_ROOT / "src" / "automata_drs" / "penman" / "state.py"
        )
        train_audit = {
            "corpus": {
                "sha256": train_expected["sha256"],
                "records": train_expected["records"],
            },
            "official_decode": {"acceptance_rate": 1.0},
            "local_raw": {"acceptance_rate": 1.0},
            "implementation": {
                "reader_sha256": reader_hash,
                "state_sha256": state_hash,
                "script_sha256": sha256_file(
                    REPOSITORY_ROOT / "scripts" / "audit_local_amr_corpus.py"
                ),
            },
            "gate": {"passed": True},
        }
        train_path = root / "train-audit.json"
        train_path.write_text(json.dumps(train_audit), encoding="utf-8")

        implementation = {
            "reader_sha256": reader_hash,
            "prompt_sha256": sha256_file(
                REPOSITORY_ROOT / "src" / "automata_drs" / "amr_causal_lm.py"
            ),
            "lifting_sha256": sha256_file(
                REPOSITORY_ROOT / "src" / "automata_drs" / "lifting.py"
            ),
            "penman_state_sha256": state_hash,
            "trainer_sha256": sha256_file(
                REPOSITORY_ROOT / "scripts" / "train_causal_lm_lora.py"
            ),
            "script_sha256": sha256_file(
                REPOSITORY_ROOT / "scripts" / "audit_amr_tokenization.py"
            ),
        }

        def split(expected: dict, *, longest: bool) -> dict:
            value = {
                "sha256": expected["sha256"],
                "records": expected["records"],
                "checks": {
                    "hash_matches": True,
                    "record_count_matches": True,
                    "exact_target_byte_roundtrip": True,
                    "accepted_with_eos": True,
                },
                "coverage": {
                    "exact_target_byte_roundtrip": expected["records"],
                    "accepted_with_eos": expected["records"],
                    "acceptance_rate": 1.0,
                },
                "lengths": {
                    "longest_records": [],
                    "prompt": {"maximum": 120},
                    "target_with_eos": {"maximum": 1500},
                },
            }
            if longest:
                value["lengths"]["longest_records"] = [
                    {"ordinal": index, "sequence_tokens": 1800 - index}
                    for index in range(1, 17)
                ]
            return value

        token_audit = {
            "config": {"sha256": sha256_file(CONFIG)},
            "tokenizer": {
                "model": config["host"]["repo_id"],
                "requested_revision": config["host"]["revision"],
                "resolved_revision": config["host"]["revision"],
                "model_max_length": 32768,
            },
            "splits": {
                "train": split(train_expected, longest=True),
                "development": split(dev_expected, longest=False),
            },
            "training_length_decision": {
                "minimum_observed_no_truncation": 1799,
                "rounded_up_multiple_of_128": 1920,
            },
            "host_memory_estimate": {"approximate_compact_bytes": 200_000_000},
            "controller": {
                "max_cache_entries": 20_000,
                "final_cache": {"entries": 20_000},
            },
            "implementation": implementation,
            "gate": {"passed": True},
        }
        token_path = root / "token-audit.json"
        token_path.write_text(json.dumps(token_audit), encoding="utf-8")
        return train_path, token_path

    def run_gate(
        self, train: Path, token: Path, output: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(PYTHON),
                str(SCRIPT),
                "--config",
                str(CONFIG),
                "--train-audit",
                str(train),
                "--token-audit",
                str(token),
                "--output",
                str(output),
            ],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
        )

    def test_passes_and_recommends_worst_case_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, token = self.make_evidence(root)
            output = root / "gate.json"
            result = self.run_gate(train, token, output)
            self.assertEqual(0, result.returncode, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(1920, report["recommendation"]["max_length"])
        self.assertEqual(16, len(report["recommendation"]["smoke_train_record_ordinals"]))

    def test_recomputes_coverage_instead_of_trusting_inner_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, token = self.make_evidence(root)
            value = json.loads(token.read_text(encoding="utf-8"))
            value["splits"]["train"]["coverage"]["accepted_with_eos"] -= 1
            token.write_text(json.dumps(value), encoding="utf-8")
            output = root / "gate.json"
            result = self.run_gate(train, token, output)
            self.assertEqual(1, result.returncode, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertFalse(report["gate"]["passed"])
        self.assertFalse(report["gate"]["checks"]["token_train_coverage"])


if __name__ == "__main__":
    unittest.main()
