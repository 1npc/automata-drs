from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "derive_amr_nfc_corpus.py"


class AMRNFCDerivationTest(unittest.TestCase):
    def test_derives_canonically_equivalent_corpus_and_verifies_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.amr"
            output = root / "train.nfc.amr"
            audit = root / "audit.json"
            source.write_text(
                'Cafe\u0301\n\n(c / cafe :name "Canape\u0301")\n\n',
                encoding="utf-8",
            )
            command = [
                "python3",
                str(SCRIPT),
                str(source),
                "--output",
                str(output),
                "--audit",
                str(audit),
            ]
            created = subprocess.run(
                command, cwd=REPOSITORY_ROOT, text=True, capture_output=True
            )
            self.assertEqual(0, created.returncode, created.stderr)
            self.assertEqual(
                'Café\n\n(c / cafe :name "Canapé")\n\n',
                output.read_text(encoding="utf-8"),
            )
            report = json.loads(audit.read_text(encoding="utf-8"))
            self.assertTrue(report["gate"]["passed"])
            self.assertEqual([1], report["changes"]["changed_records"])
            self.assertEqual([1], report["changes"]["changed_sentence_records"])
            self.assertEqual([1], report["changes"]["changed_graph_records"])
            verified = subprocess.run(
                [*command, "--verify-existing"],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, verified.returncode, verified.stderr)


if __name__ == "__main__":
    unittest.main()
