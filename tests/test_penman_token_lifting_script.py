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
SCRIPT = REPOSITORY_ROOT / "scripts" / "audit_penman_token_lifting.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location(
        "audit_penman_token_lifting_test_module", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PENMANTokenLiftingScriptTest(unittest.TestCase):
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
            init_kwargs = {"_commit_hash": "resolved-revision"}

            @staticmethod
            def get_vocab():
                vocabulary = {encoder[byte]: byte for byte in range(256)}
                vocabulary["<eos>"] = 256
                return vocabulary

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
                assert model == "example/model"
                assert kwargs["revision"] == "requested-revision"
                return FakeTokenizer()

        module = types.ModuleType("transformers")
        module.__version__ = "test-version"
        module.AutoTokenizer = AutoTokenizer
        return module

    def invoke(self, module, arguments: list[str]) -> int:
        with (
            patch.object(sys, "argv", [str(SCRIPT), *arguments]),
            patch.dict(
                sys.modules,
                {"transformers": self.fake_transformers()},
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            return module.main()

    def test_write_verify_and_detect_tampering(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.txt"
            corpus.write_text(
                "# ::id one\n(a / alpha)\n\n"
                '# ::id two\n(p / person :name (n / name :op1 "张"))\n',
                encoding="utf-8",
            )
            output = root / "audit.json"
            common = [
                str(corpus),
                "--model",
                "example/model",
                "--revision",
                "requested-revision",
                "--expect-records",
                "2",
                "--source-revision",
                "source-revision",
            ]
            self.assertEqual(
                0,
                self.invoke(module, [*common, "--output", str(output)]),
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(2, report["coverage"]["exact_byte_roundtrip"])
            self.assertEqual(2, report["coverage"]["accepted_with_eos"])
            self.assertTrue(report["gate"]["passed"])

            self.assertEqual(
                0,
                self.invoke(
                    module,
                    [*common, "--verify-existing", str(output)],
                ),
            )

            report["coverage"]["accepted_with_eos"] = 1
            output.write_text(json.dumps(report) + "\n", encoding="utf-8")
            with self.assertRaises(SystemExit) as raised:
                self.invoke(
                    module,
                    [*common, "--verify-existing", str(output)],
                )
            self.assertEqual(2, raised.exception.code)


if __name__ == "__main__":
    unittest.main()
