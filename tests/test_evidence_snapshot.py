from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "verify_evidence_snapshot.py"
SPEC = importlib.util.spec_from_file_location("verify_evidence_snapshot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EvidenceSnapshotTest(unittest.TestCase):
    @staticmethod
    def write(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, str):
            path.write_text(value, encoding="utf-8")
        else:
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    @staticmethod
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def ready_lock(_drs: Path, _amr: Path, _package: Path) -> dict[str, object]:
        return {
            "state": "all-suite-evidence-and-result-package-ready-for-integration",
            "drs": {"ready": True},
            "amr": {"ready": True},
            "result_package": {"ready": True},
        }

    def build_fixture(self, root: Path) -> tuple[Path, Path]:
        drs_input = root / "runs" / "lora" / "gate.json"
        amr_input = root / "runs" / "amr" / "gate.json"
        self.write(drs_input, {"passed": True})
        self.write(amr_input, {"passed": True})
        drs_suite = root / str(MODULE.DRS_SUITE)
        amr_suite = root / str(MODULE.AMR_SUITE)
        result_package = root / str(MODULE.RESULT_PACKAGE)
        self.write(
            drs_suite,
            {
                "evidence": {
                    "input_sha256": {
                        "runs/lora/gate.json": self.digest(drs_input)
                    }
                }
            },
        )
        self.write(
            amr_suite,
            {"evidence": {"runs/amr/gate.json": self.digest(amr_input)}},
        )
        self.write(result_package, {"audit": "manuscript-result-package"})
        self.write(root / str(MODULE.DRS_MARKDOWN), "DRS rendered\n")
        self.write(root / str(MODULE.AMR_MARKDOWN), "AMR rendered\n")
        self.write(
            root / str(MODULE.RESULT_PACKAGE_MARKDOWN), "Results rendered\n"
        )
        return drs_input, amr_input

    def test_manifest_binds_canonical_suites_and_compact_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drs_input, _ = self.build_fixture(root)
            report = MODULE.build_manifest(
                root,
                lock_builder=self.ready_lock,
                drs_renderer=lambda _report: "DRS rendered\n",
                amr_renderer=lambda _report: "AMR rendered\n",
                result_package_renderer=lambda _report: "Results rendered\n",
            )
            self.assertTrue(report["gate"]["passed"])
            self.assertEqual(
                "all-suite-evidence-and-result-package-ready-for-integration",
                report["result_lock"]["state"],
            )
            self.assertEqual(
                1,
                report["canonical_suites"]["drs"]["references"][
                    "required_compact_files"
                ],
            )
            self.write(drs_input, {"passed": False})
            with self.assertRaisesRegex(ValueError, "hash drifted"):
                MODULE.build_manifest(
                    root,
                    lock_builder=self.ready_lock,
                    drs_renderer=lambda _report: "DRS rendered\n",
                    amr_renderer=lambda _report: "AMR rendered\n",
                    result_package_renderer=lambda _report: "Results rendered\n",
                )

    def test_manifest_refuses_incomplete_lock_or_unregistered_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_fixture(root)

            def incomplete_lock(
                _drs: Path, _amr: Path, _package: Path
            ) -> dict[str, object]:
                return {
                    "state": "partial-suite-evidence-ready",
                    "drs": {"ready": True},
                    "amr": {"ready": False},
                    "gate": {"passed": True},
                }

            with self.assertRaisesRegex(ValueError, "do not pass the result lock"):
                MODULE.build_manifest(
                    root,
                    lock_builder=incomplete_lock,
                    drs_renderer=lambda _report: "DRS rendered\n",
                    amr_renderer=lambda _report: "AMR rendered\n",
                    result_package_renderer=lambda _report: "Results rendered\n",
                )
            self.write(root / "runs" / "lora" / "predictions.jsonl", "row\n")
            with self.assertRaisesRegex(ValueError, "unregistered file"):
                MODULE.build_manifest(
                    root,
                    lock_builder=self.ready_lock,
                    drs_renderer=lambda _report: "DRS rendered\n",
                    amr_renderer=lambda _report: "AMR rendered\n",
                    result_package_renderer=lambda _report: "Results rendered\n",
                )


if __name__ == "__main__":
    unittest.main()
