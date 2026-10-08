from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "audit_hf_cache_partials.py"


class HuggingFaceCachePartialAuditTest(unittest.TestCase):
    def test_reports_only_proven_shadowed_bytes_as_reclaimable(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            cache = scratch / "automata-drs-lora.resumev2"
            blobs = cache / "huggingface" / "hub" / "models--Qwen--Model" / "blobs"
            blobs.mkdir(parents=True)
            (blobs / "aaa").write_bytes(b"complete")
            (blobs / "aaa.1234abcd.incomplete").write_bytes(b"old")
            (blobs / "bbb.deadbeef.incomplete").write_bytes(b"active")
            (blobs / "legacy.incomplete").write_bytes(b"unknown")

            completed = subprocess.run(
                [
                    str(SCRIPT),
                    str(cache),
                    "--scratch-root",
                    str(scratch),
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertFalse(report["deletes_files"])
            self.assertEqual(3, report["partial_files"])
            self.assertEqual(16, report["partial_bytes"])
            self.assertEqual(1, report["proven_reclaimable_files_after_process_exit"])
            self.assertEqual(3, report["proven_reclaimable_bytes_after_process_exit"])
            by_name = {Path(row["path"]).name: row for row in report["partials"]}
            self.assertTrue(
                by_name["aaa.1234abcd.incomplete"][
                    "safe_to_remove_after_process_exit"
                ]
            )
            self.assertFalse(
                by_name["bbb.deadbeef.incomplete"][
                    "safe_to_remove_after_process_exit"
                ]
            )
            self.assertEqual(
                "unrecognized-incomplete-file",
                by_name["legacy.incomplete"]["kind"],
            )

    def test_refuses_cache_outside_exact_owned_scratch_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scratch = root / "scratch"
            cache = root / "automata-drs-lora.resumev2"
            (cache / "huggingface" / "hub").mkdir(parents=True)
            scratch.mkdir()
            completed = subprocess.run(
                [
                    str(SCRIPT),
                    str(cache),
                    "--scratch-root",
                    str(scratch),
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(0, completed.returncode)
            self.assertIn("outside the exact scratch root", completed.stderr)

    def test_output_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            cache = scratch / "automata-drs-lora.audit"
            (cache / "huggingface" / "hub").mkdir(parents=True)
            output = scratch / "audit.json"
            command = [
                str(SCRIPT),
                str(cache),
                "--scratch-root",
                str(scratch),
                "--output",
                str(output),
            ]
            first = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True)
            second = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True)
            self.assertEqual(0, first.returncode, first.stderr)
            self.assertNotEqual(0, second.returncode)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
