import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from arr1012_review_recovery import load_reusable_rows


class RecoveryTest(unittest.TestCase):
    def fixture(self, directory):
        path = Path(directory)/"old.jsonl"
        rows = [{"id": "x:1", "configuration": name, "decode": {"strategy": "greedy", "num_beams": 1},
                 "generation": {"native_token_ids": [1, 2]},
                 "constraint": {"all_masked_rows": masked}, "raw_prediction": "person.n.01"}
                for name, masked in (("WF", 1), ("WWCD", 0))]
        path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
        Path(str(path)+".manifest.json").write_text("{}", encoding="utf-8")
        return path, {"review_reuse_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, [("x:1", "WF"),("x:1", "WWCD")]

    def test_failed_rows_rerun_successful_rows_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path, manifest, ids = self.fixture(folder)
            rows = load_reusable_rows(path, manifest, ids)
            self.assertEqual(list(rows), [("x:1", "WWCD")])
            self.assertEqual(rows[("x:1", "WWCD")]["review_reuse"]["line"], 2)

    def test_contract_change_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, manifest, ids = self.fixture(folder)
            manifest["max_new_tokens"] = 78
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                load_reusable_rows(path, manifest, ids)

    def test_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, manifest, ids = self.fixture(folder)
            manifest["review_reuse_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "hash changed"):
                load_reusable_rows(path, manifest, ids)

    def test_truncated_final_write_not_reused_and_never_edited(self):
        with tempfile.TemporaryDirectory() as folder:
            path, manifest, ids = self.fixture(folder)
            with path.open("ab") as f:
                f.write(b'{"id":')
            raw = path.read_bytes()
            manifest["review_reuse_sha256"] = hashlib.sha256(raw).hexdigest()
            self.assertEqual(len(load_reusable_rows(path, manifest, ids)), 1)
            self.assertEqual(path.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
