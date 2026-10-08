from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from automata_drs import gpt2_byte_decoder


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/audit_penman_completion_paths.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("penman_pf_audit_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PENMANCompletionAuditTest(unittest.TestCase):
    @staticmethod
    def fake_transformers():
        inverse = gpt2_byte_decoder()
        encoder = {byte: character for character, byte in inverse.items()}

        class ByteLevel:
            pass

        class Backend:
            decoder = ByteLevel()

        class FakeTokenizer:
            backend_tokenizer = Backend()
            all_special_ids = [256]
            eos_token_id = 256
            init_kwargs = {"_commit_hash": "requested-revision"}

            @staticmethod
            def get_vocab():
                result = {encoder[byte]: byte for byte in range(256)}
                result["<eos>"] = 256
                return result

            @staticmethod
            def __call__(text, *, add_special_tokens):
                assert add_special_tokens is False
                return {"input_ids": list(text.encode("utf-8"))}

            @staticmethod
            def __len__():
                return 257

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(model, **kwargs):
                if model == "example/model":
                    assert kwargs["revision"] == "requested-revision"
                else:
                    assert Path(model).name == "saved-tokenizer"
                    assert "revision" not in kwargs
                return FakeTokenizer()

        module = types.ModuleType("transformers")
        module.__version__ = "test-version"
        module.AutoTokenizer = AutoTokenizer
        return module

    @staticmethod
    def invoke(module, arguments: list[str]) -> int:
        with (
            patch.object(sys, "argv", [str(SCRIPT), *arguments]),
            patch.dict(sys.modules, {"transformers": PENMANCompletionAuditTest.fake_transformers()}),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            return module.main()

    def test_amr_paired_gold_paths_write_verify_and_bind_horizon(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "validation.amr"
            corpus.write_text(
                "A person wants.\n\n"
                "(w / want-01 :ARG0 (p / person))\n\n"
                "Her name is Zhang.\n\n"
                '(p / person :name (n / name :op1 "张"))\n',
                encoding="utf-8",
            )
            output = root / "pf.json"
            contract = root / "resource-plan.json"
            contract.write_text('{"max_new_tokens": 256}\n', encoding="utf-8")
            tokenizer_source = root / "saved-tokenizer"
            tokenizer_source.mkdir()
            (tokenizer_source / "tokenizer.json").write_text(
                '{"version": "test"}\n', encoding="utf-8"
            )
            (tokenizer_source / "tokenizer_config.json").write_text(
                '{"eos_token": "<eos>"}\n', encoding="utf-8"
            )
            common = [
                str(corpus),
                "--input-format",
                "amr-paired",
                "--model",
                "example/model",
                "--revision",
                "requested-revision",
                "--tokenizer-source",
                str(tokenizer_source),
                "--max-new-tokens",
                "256",
                "--expect-records",
                "2",
                "--contract-file",
                str(contract),
                "--require-zero-false-rejections",
            ]
            self.assertEqual(0, self.invoke(module, [*common, "--output", str(output)]))
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(2, report["coverage"]["certified_records"])
            self.assertEqual(
                report["coverage"]["gold_token_successors_tested"],
                report["coverage"]["gold_token_successors_certified"],
            )
            self.assertGreater(report["witnesses"]["tokens_maximum"], 0)
            self.assertTrue(report["gate"]["passed"])
            self.assertFalse(report["contract"]["changes_frozen_amr_u_p_suite"])
            self.assertEqual(
                str(contract), report["contract"]["bound_contract_file"]["path"]
            )
            self.assertEqual(2, report["schema_version"])
            self.assertTrue(report["tokenizer"]["revision_bound"])
            self.assertEqual(
                str(tokenizer_source), report["tokenizer"]["load_source"]
            )
            self.assertEqual(
                64,
                len(report["tokenizer"]["local_assets"]["aggregate_sha256"]),
            )
            self.assertEqual(
                64, len(report["tokenizer"]["byte_vocabulary_sha256"])
            )
            self.assertEqual(64, len(report["source"]["gold_token_paths_sha256"]))
            self.assertLessEqual(
                report["controller"]["final_cache"]["entries"],
                report["controller"]["max_cache_entries"],
            )
            self.assertEqual(
                0,
                self.invoke(module, [*common, "--verify-existing", str(output)]),
            )

            tokenizer_config = tokenizer_source / "tokenizer_config.json"
            original_tokenizer_config = tokenizer_config.read_text(encoding="utf-8")
            tokenizer_config.write_text(
                '{"eos_token": "changed"}\n', encoding="utf-8"
            )
            with self.assertRaises(SystemExit) as raised:
                self.invoke(module, [*common, "--verify-existing", str(output)])
            self.assertEqual(2, raised.exception.code)
            tokenizer_config.write_text(
                original_tokenizer_config, encoding="utf-8"
            )

            report["contract"]["max_new_tokens"] = 255
            output.write_text(json.dumps(report) + "\n", encoding="utf-8")
            with self.assertRaises(SystemExit) as raised:
                self.invoke(module, [*common, "--verify-existing", str(output)])
            self.assertEqual(2, raised.exception.code)

    def test_too_short_horizon_is_reported_not_silently_skipped(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.txt"
            corpus.write_text("# ::id one\n(a / alpha)\n", encoding="utf-8")
            output = root / "pf.json"
            self.assertEqual(
                1,
                self.invoke(
                    module,
                    [
                        str(corpus),
                        "--input-format",
                        "penman",
                        "--model",
                        "example/model",
                        "--revision",
                        "requested-revision",
                        "--max-new-tokens",
                        "2",
                        "--output",
                        str(output),
                        "--require-zero-false-rejections",
                    ],
                ),
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(1, report["coverage"]["over_budget_records"])
            self.assertFalse(report["gate"]["passed"])

    def test_expected_source_and_config_hashes_fail_before_tokenizer_loading(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.txt"
            corpus.write_text("# ::id one\n(a / alpha)\n", encoding="utf-8")
            output = root / "pf.json"
            for option in ("--expect-corpus-sha256", "--expect-config-sha256"):
                with self.assertRaises(SystemExit) as raised:
                    self.invoke(
                        module,
                        [
                            str(corpus),
                            "--input-format",
                            "penman",
                            "--model",
                            "example/model",
                            "--revision",
                            "requested-revision",
                            "--max-new-tokens",
                            "256",
                            option,
                            "0" * 64,
                            "--output",
                            str(output),
                        ],
                    )
                self.assertEqual(2, raised.exception.code)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
