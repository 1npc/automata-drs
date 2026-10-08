from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from automata_drs.interventions import build_intervention_plan


def row(
    identifier: str,
    prediction: str,
    *,
    raw_sbn: bool,
    referee: bool | None,
    raw_referee: bool | None = None,
    configuration: str = "U",
    canonical: bool = True,
) -> dict[str, object]:
    return {
        "id": identifier,
        "input": "example",
        "reference": "dog.n.01",
        "prediction": prediction,
        "raw_prediction": prediction,
        "configuration": configuration,
        "validity": {
            "raw_sbn": raw_sbn,
            "sbn": raw_sbn,
            **(
                {}
                if raw_referee is None
                else {"raw_clf_referee": raw_referee}
            ),
            "clf_referee": referee,
        },
        "generation": {
            "canonical_tokenization": canonical,
            "hit_max_new_tokens": False,
        },
    }


class InterventionPlanTest(unittest.TestCase):
    digest = "a" * 64

    def test_selects_structural_and_referee_only_failures(self):
        records = [
            row("dev:1", "dog.n.01", raw_sbn=True, referee=True),
            row(
                "dev:2",
                "dog.n.01 Theme +1",
                raw_sbn=False,
                referee=None,
                canonical=False,
            ),
            row(
                "dev:3",
                "person.n.01 CONTINUATION <0 person.n.01",
                raw_sbn=True,
                referee=False,
            ),
            row(
                "dev:1",
                "dog.n.01",
                raw_sbn=True,
                referee=True,
                configuration="BF",
            ),
        ]
        plan = build_intervention_plan(
            records, input_path="dev.jsonl", input_sha256=self.digest
        )
        self.assertEqual(3, plan["u_records"])
        self.assertEqual(2, plan["selected_records"])
        self.assertEqual(1, plan["structural_invalid"])
        self.assertEqual(1, plan["referee_only_invalid"])
        self.assertEqual(1, plan["canonical_noncanonical_total"])
        self.assertEqual([2, 3], [item["line_number"] for item in plan["selected"]])
        self.assertEqual(
            "reference-existence", plan["selected"][0]["failure_family"]
        )
        self.assertEqual(
            "semantic-wellformedness", plan["selected"][1]["failure_family"]
        )
        self.assertTrue(plan["selected"][1]["b_valid"])
        self.assertFalse(plan["selected"][1]["w_valid"])

    def test_stored_validity_is_independently_replayed(self):
        with self.assertRaisesRegex(ValueError, "stored/raw B validity mismatch"):
            build_intervention_plan(
                [row("dev:1", "dog.n.01 Theme +1", raw_sbn=True, referee=True)],
                input_path="dev.jsonl",
                input_sha256=self.digest,
            )
        with self.assertRaisesRegex(ValueError, "stored/raw W validity mismatch"):
            build_intervention_plan(
                [
                    row(
                        "dev:1",
                        "person.n.01 CONTINUATION <0 person.n.01",
                        raw_sbn=True,
                        referee=True,
                    )
                ],
                input_path="dev.jsonl",
                input_sha256=self.digest,
            )

    def test_raw_referee_takes_precedence_over_postprocessed_fallback(self):
        raw_valid = row(
            "dev:1",
            "dog.n.01",
            raw_sbn=True,
            raw_referee=True,
            referee=False,
        )
        plan = build_intervention_plan(
            [raw_valid], input_path="dev.jsonl", input_sha256=self.digest
        )
        self.assertEqual(0, plan["selected_records"])

        raw_invalid = row(
            "dev:2",
            "person.n.01 CONTINUATION <0 person.n.01",
            raw_sbn=True,
            raw_referee=False,
            referee=True,
        )
        plan = build_intervention_plan(
            [raw_invalid], input_path="dev.jsonl", input_sha256=self.digest
        )
        self.assertEqual(1, plan["selected_records"])
        self.assertIs(
            False,
            plan["selected"][0]["stored_validity"]["raw_clf_referee"],
        )

    def test_duplicate_u_identifiers_and_bad_digest_fail_closed(self):
        valid = row("dev:1", "dog.n.01", raw_sbn=True, referee=True)
        with self.assertRaisesRegex(ValueError, "duplicate U identifiers"):
            build_intervention_plan(
                [valid, valid],
                input_path="dev.jsonl",
                input_sha256=self.digest,
            )
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            build_intervention_plan(
                [valid], input_path="dev.jsonl", input_sha256="not-a-digest"
            )

    def test_planning_script_freezes_source_hash_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "predictions.jsonl"
            output = root / "selection.json"
            source_bytes = (
                json.dumps(
                    row("dev:7", "dog.n.01 Theme +1", raw_sbn=False, referee=None)
                )
                + "\n"
            ).encode()
            source.write_bytes(source_bytes)
            command = [
                sys.executable,
                "scripts/plan_lora_interventions.py",
                str(source),
                "--output",
                str(output),
            ]
            first = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(0, first.returncode, first.stderr)
            plan = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(hashlib.sha256(source_bytes).hexdigest(), plan["input_sha256"])
            self.assertEqual([7], [item["line_number"] for item in plan["selected"]])
            second = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(0, second.returncode)
            self.assertIn("refusing to overwrite", second.stderr)


if __name__ == "__main__":
    unittest.main()
