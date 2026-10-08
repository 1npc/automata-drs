from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCORE_SCRIPT = REPOSITORY_ROOT / "scripts" / "score_sbn_smatch.py"


class ScoreScriptTest(unittest.TestCase):
    @staticmethod
    def write_predictions(path: Path) -> None:
        rows = [
            {
                "id": str(index),
                "reference": concept,
                "prediction": concept,
                "configuration": "U",
                "decode": {"strategy": "greedy", "num_beams": 1},
            }
            for index, concept in enumerate(("dog.n.01", "cat.n.01"))
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    @staticmethod
    def write_duplicate_predictions(path: Path) -> None:
        rows = [
            {
                "id": "1",
                "reference": "dog.n.01",
                "prediction": "dog.n.01",
                "configuration": configuration,
                "decode": {"strategy": "greedy", "num_beams": 1},
            }
            for configuration in ("U", "BF")
        ]
        rows.append(
            {
                "id": "2",
                "reference": "cat.n.01",
                "prediction": "cat.n.01",
                "configuration": "W",
                "decode": {"strategy": "greedy", "num_beams": 1},
            }
        )
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def test_scores_and_writes_provenance_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            scorer = root / "fake_scorer.py"
            output = root / "scores.jsonl"
            self.write_predictions(predictions)
            scorer.write_text(
                "print('0:1.0')\n"
                "print('1:0.25')\n"
                "print('average f1 smatch score: 0.625')\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCORE_SCRIPT),
                    str(predictions),
                    "--scorer",
                    str(scorer),
                    "--output",
                    str(output),
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            scores = [json.loads(line) for line in output.read_text().splitlines()]
            manifest = json.loads(
                Path(str(output) + ".manifest.json").read_text(encoding="utf-8")
            )
            scorer_hash = hashlib.sha256(scorer.read_bytes()).hexdigest()

        self.assertEqual([1.0, 0.25], [row["semantic_scores"]["sbn_smatch"] for row in scores])
        self.assertEqual(2, manifest["records"])
        self.assertEqual(scorer_hash, manifest["scorer_sha256"])
        self.assertIsNone(manifest["scorer_git_revision"])
        self.assertEqual("seeded-deduplicated-semantic-pairs-v2", manifest["scoring_protocol"])
        self.assertEqual(2, manifest["unique_semantic_pairs"])
        self.assertEqual(0, manifest["reused_score_records"])
        self.assertEqual(42, manifest["seed"])

    def test_identical_semantic_pairs_share_one_seeded_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            scorer = root / "fake_scorer.py"
            output = root / "scores.jsonl"
            self.write_duplicate_predictions(predictions)
            scorer.write_text(
                "import sys\n"
                "lines = open(sys.argv[2], encoding='utf-8').read().splitlines()\n"
                "for index, _ in enumerate(lines):\n"
                "    print(f'{index}:{0.5 + index * 0.25}')\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCORE_SCRIPT),
                    str(predictions),
                    "--scorer",
                    str(scorer),
                    "--output",
                    str(output),
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            scores = [json.loads(line) for line in output.read_text().splitlines()]
            manifest = json.loads(
                Path(str(output) + ".manifest.json").read_text(encoding="utf-8")
            )

        values = [row["semantic_scores"]["sbn_smatch"] for row in scores]
        self.assertEqual([0.5, 0.5, 0.75], values)
        self.assertEqual(2, manifest["unique_semantic_pairs"])
        self.assertEqual(1, manifest["reused_score_records"])
        self.assertTrue(manifest["identical_pair_score_consistency"])
        self.assertEqual(1, manifest["scorer_invocations"])

    def test_seed_makes_external_scorer_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            scorer = root / "random_scorer.py"
            self.write_predictions(predictions)
            scorer.write_text(
                "import random, sys\n"
                "lines = open(sys.argv[2], encoding='utf-8').read().splitlines()\n"
                "for index, _ in enumerate(lines):\n"
                "    print(f'{index}:{random.random()}')\n",
                encoding="utf-8",
            )
            outputs = []
            for name in ("scores-a.jsonl", "scores-b.jsonl"):
                output = root / name
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(SCORE_SCRIPT),
                        str(predictions),
                        "--scorer",
                        str(scorer),
                        "--output",
                        str(output),
                        "--seed",
                        "17",
                    ],
                    cwd=REPOSITORY_ROOT,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(0, completed.returncode, completed.stderr)
                outputs.append(output.read_text(encoding="utf-8"))

        self.assertEqual(outputs[0], outputs[1])

    def test_input_change_during_scoring_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            scorer = root / "mutating_scorer.py"
            output = root / "scores.jsonl"
            self.write_predictions(predictions)
            scorer.write_text(
                "from pathlib import Path\n"
                f"path = Path({str(predictions)!r})\n"
                "path.write_text(path.read_text() + '\\n', encoding='utf-8')\n"
                "print('0:1.0')\n"
                "print('1:1.0')\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCORE_SCRIPT),
                    str(predictions),
                    "--scorer",
                    str(scorer),
                    "--output",
                    str(output),
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            output_exists = output.exists()

        self.assertNotEqual(0, completed.returncode)
        self.assertIn("changed while SBN Smatch was running", completed.stderr)
        self.assertFalse(output_exists)


if __name__ == "__main__":
    unittest.main()
