from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPOSITORY_ROOT / "scripts" / "check_lora_suite_v3_promotion.py"
RUNNER = REPOSITORY_ROOT / "scripts" / "promote_colossus_lora_suite_v3.sh"
SPEC = importlib.util.spec_from_file_location("lora_suite_v3_promotion", CHECKER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LoRASuiteV3PromotionTest(unittest.TestCase):
    @staticmethod
    def write(path: Path, value: object) -> None:
        if isinstance(value, str):
            path.write_text(value, encoding="utf-8")
        else:
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        v2_json = root / "v2.json"
        v2_markdown = root / "v2.md"
        v3_json = root / "v3.json"
        v3_markdown = root / "v3.md"
        selection = {"status": "selected", "selected_key": "m"}
        self.write(
            v2_json,
            {
                "schema_version": 6,
                "scope": "full",
                "host_selection": selection,
                "evidence": {"config_sha256": "a" * 64},
            },
        )
        input_hashes = {
            f"runs/lora/{evaluation}.summary.json": "b" * 64
            for evaluation in MODULE.EXPECTED_EVALUATIONS
        }
        self.write(
            v3_json,
            {
                "schema_version": 7,
                "scope": "full",
                "host_selection": selection,
                "models": [
                    {
                        "key": "m",
                        "method_baselines": [
                            {
                                "evaluation": evaluation,
                                "configuration": "U",
                                "strategy": (
                                    "beam4"
                                    if evaluation == "dev400-beam4"
                                    else "greedy"
                                ),
                                "path": f"runs/lora/{evaluation}.summary.json",
                                "group": {"total": 2},
                            }
                            for evaluation in MODULE.EXPECTED_EVALUATIONS
                        ],
                    }
                ],
                "evidence": {
                    "config_sha256": "a" * 64,
                    "input_sha256": input_hashes,
                },
            },
        )
        self.write(v2_markdown, "legacy\n")
        self.write(v3_markdown, "current\n")
        return v2_json, v2_markdown, v3_json, v3_markdown

    def test_gate_binds_immutable_v2_and_complete_v3_u_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            report = MODULE.build_report(*paths)
            self.assertTrue(report["gate"]["passed"])
            self.assertEqual(6, len(report["target_v3"]["evaluation_specific_u_rows"]))
            self.write(paths[1], "changed legacy\n")
            changed = MODULE.build_report(*paths)
        self.assertNotEqual(
            report["source_v2"]["markdown_sha256"],
            changed["source_v2"]["markdown_sha256"],
        )

    def test_gate_refuses_missing_u_and_runner_is_cpu_only(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            v3 = json.loads(paths[2].read_text(encoding="utf-8"))
            v3["models"][0]["method_baselines"].pop()
            self.write(paths[2], v3)
            with self.assertRaisesRegex(ValueError, "evaluation_specific_u"):
                MODULE.build_report(*paths)

        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("recent-model-and-method-results-v2.json", source)
        self.assertIn("recent-model-and-method-results-v3.json", source)
        self.assertIn("check_lora_suite_v3_promotion.py", source)
        self.assertIn("--verify-existing", source)
        self.assertIn("OMP_NUM_THREADS=1", source)
        self.assertIn("nice -n 15", source)
        self.assertIn("ulimit -f", source)
        self.assertIn("pgrep -f", source)
        self.assertIn("run_colossus_lora_interventions_v2", source)
        self.assertIn("sha256sum", source)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", source)
        self.assertNotIn("nvidia-smi", source)
        syntax = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, syntax.returncode, syntax.stderr)


if __name__ == "__main__":
    unittest.main()
