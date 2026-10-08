from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY_ROOT / "scripts" / "run_colossus_amr_preflight.sh"
AUDIT_SCRIPT = REPOSITORY_ROOT / "scripts" / "audit_amr_tokenization.py"


class AMRPreflightTest(unittest.TestCase):
    @staticmethod
    def load_audit_module():
        spec = importlib.util.spec_from_file_location(
            "audit_amr_tokenization_test", AUDIT_SCRIPT
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_runner_is_cpu_only_private_and_refuses_drs_overlap(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("automata_scratch_require_private", source)
        self.assertIn("AUTOMATA_DRS_AMR_DATA_ROOT", source)
        self.assertIn("AUTOMATA_DRS_ANALYSIS_PYTHON", source)
        self.assertIn("AUTOMATA_DRS_ALLOW_CPU_PREFLIGHT_OVERLAP", source)
        self.assertIn("the frozen DRS selected-host suite is still active", source)
        self.assertIn("OMP_NUM_THREADS=1", source)
        self.assertIn("TOKENIZERS_PARALLELISM=false", source)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", source)
        self.assertNotIn("automata_gpu_claim", source)
        self.assertLess(source.index("pgrep -f"), source.index("automata_scratch_mktemp"))
        syntax = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, syntax.returncode, syntax.stderr)

    def test_token_audit_length_summary_is_exact(self):
        module = self.load_audit_module()
        self.assertEqual(
            {"minimum": 1, "median": 2, "p95": 9, "p99": 9, "maximum": 9},
            module._length_summary([1, 2, 9]),
        )

    def test_token_audit_resolves_revision_from_hub_snapshot_path(self):
        module = self.load_audit_module()
        revision = "1" * 40
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "models--org--model"
                / "snapshots"
                / revision
                / "tokenizer_config.json"
            )
            path.parent.mkdir(parents=True)
            path.touch()
            tokenizer = type("Tokenizer", (), {"init_kwargs": {}})()
            self.assertEqual(
                revision,
                module._resolved_huggingface_revision(tokenizer, path),
            )

    def test_token_audit_resolves_revision_before_following_snapshot_symlink(self):
        module = self.load_audit_module()
        revision = "2" * 40
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "models--org--model"
            blob = root / "blobs" / "tokenizer-config-blob"
            blob.parent.mkdir(parents=True)
            blob.touch()
            path = root / "snapshots" / revision / "tokenizer_config.json"
            path.parent.mkdir(parents=True)
            path.symlink_to(blob)
            tokenizer = type("Tokenizer", (), {"init_kwargs": {}})()
            self.assertEqual(
                revision,
                module._resolved_huggingface_revision(tokenizer, path),
            )

    def test_token_audit_revision_resolution_is_fail_closed(self):
        module = self.load_audit_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer_config.json"
            path.touch()
            tokenizer = type("Tokenizer", (), {"init_kwargs": {}})()
            self.assertIsNone(
                module._resolved_huggingface_revision(tokenizer, path)
            )

    def test_token_audit_cannot_use_heldout_test_for_length_selection(self):
        source = AUDIT_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('choices=("train", "development")', source)
        self.assertNotIn('choices=("train", "development", "test")', source)


if __name__ == "__main__":
    unittest.main()
