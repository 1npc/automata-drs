from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from automata_drs.lora_layer_ablation import (
    combine_fine_grained_ablation_rows,
    combine_layer_ablation_rows,
)


def row(identifier: str, configuration: str, *, source: str = "text"):
    return {
        "id": identifier,
        "input": source,
        "reference": "male.n.02",
        "configuration": configuration,
        "payload": f"{identifier}:{configuration}",
    }


class LoraLayerAblationTest(unittest.TestCase):
    def test_fine_merge_cli_marks_derived_manifest_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layer = root / "layer.jsonl"
            supplement = root / "fine.jsonl"
            output = root / "combined-fine.jsonl"
            config = root / "config.json"
            layer.write_text(
                "".join(
                    json.dumps(row("1", name)) + "\n"
                    for name in ("U", "B", "W", "BF", "WF")
                ),
                encoding="utf-8",
            )
            supplement.write_text(
                "".join(
                    json.dumps(row("1", name)) + "\n"
                    for name in ("S", "R", "G")
                ),
                encoding="utf-8",
            )
            for path in (layer, supplement):
                Path(str(path) + ".manifest.json").write_text(
                    "{}\n", encoding="utf-8"
                )
            config.write_text(
                json.dumps(
                    {
                        "method_experiment": {
                            "fine_grained_ablation": {
                                "records": 1,
                                "generated_configurations": ["S", "R", "G"],
                                "source_ladder": ["U", "B", "W", "BF", "WF"],
                                "derived_full_ladder": [
                                    "U", "S", "R", "G", "B", "W", "BF", "WF"
                                ],
                            }
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    "scripts/merge_lora_fine_ablation.py",
                    "--layer",
                    str(layer),
                    "--supplement",
                    str(supplement),
                    "--output",
                    str(output),
                    "--config",
                    str(config),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(
                Path(str(output) + ".manifest.json").read_text(encoding="utf-8")
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual("complete", manifest["status"])
        self.assertEqual(
            ["U", "S", "R", "G", "B", "W", "BF", "WF"],
            [item["configuration"] for item in rows],
        )

    def test_fine_grained_join_preserves_all_existing_layer_cells(self):
        layer = [
            row(identifier, configuration)
            for identifier in ("2", "1")
            for configuration in ("U", "B", "W", "BF", "WF")
        ]
        supplement = [
            row(identifier, configuration)
            for identifier in ("1", "2")
            for configuration in ("S", "R", "G")
        ]
        combined, evidence = combine_fine_grained_ablation_rows(
            layer, supplement, expected_records=2
        )
        self.assertEqual(
            [
                (identifier, configuration)
                for identifier in ("2", "1")
                for configuration in ("U", "S", "R", "G", "B", "W", "BF", "WF")
            ],
            [(item["id"], item["configuration"]) for item in combined],
        )
        originals = {item["payload"]: item for item in [*layer, *supplement]}
        self.assertTrue(all(originals[item["payload"]] is item for item in combined))
        self.assertEqual(
            "cell-preserving-fine-grained-join-by-id", evidence["derivation"]
        )

    def test_merge_cli_marks_derived_manifest_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot = root / "pilot.jsonl"
            supplement = root / "supplement.jsonl"
            output = root / "combined.jsonl"
            config = root / "config.json"
            pilot.write_text(
                "".join(
                    json.dumps(row("1", name)) + "\n"
                    for name in ("U", "BF", "WF")
                ),
                encoding="utf-8",
            )
            supplement.write_text(
                "".join(
                    json.dumps(row("1", name)) + "\n"
                    for name in ("B", "W")
                ),
                encoding="utf-8",
            )
            Path(str(pilot) + ".manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            Path(str(supplement) + ".manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            config.write_text(
                json.dumps(
                    {
                        "method_experiment": {
                            "layer_ablation": {
                                "records": 1,
                                "generated_configurations": ["B", "W"],
                                "derived_full_ladder": [
                                    "U", "B", "W", "BF", "WF"
                                ],
                            }
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    "scripts/merge_lora_layer_ablation.py",
                    "--pilot",
                    str(pilot),
                    "--supplement",
                    str(supplement),
                    "--output",
                    str(output),
                    "--config",
                    str(config),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(
                Path(str(output) + ".manifest.json").read_text(encoding="utf-8")
            )
        self.assertEqual("complete", manifest["status"])

    def test_cell_preserving_join_builds_full_ladder_in_pilot_id_order(self):
        pilot = [
            row("2", configuration)
            for configuration in ("BF", "WF", "U")
        ] + [
            row("1", configuration)
            for configuration in ("WF", "U", "BF")
        ]
        supplement = [
            row(identifier, configuration)
            for identifier in ("1", "2")
            for configuration in ("B", "W")
        ]
        combined, evidence = combine_layer_ablation_rows(
            pilot, supplement, expected_records=2
        )
        self.assertEqual(
            [
                (identifier, configuration)
                for identifier in ("2", "1")
                for configuration in ("U", "B", "W", "BF", "WF")
            ],
            [(item["id"], item["configuration"]) for item in combined],
        )
        original = {
            item["payload"]: item for item in [*pilot, *supplement]
        }
        self.assertTrue(all(original[item["payload"]] is item for item in combined))
        self.assertEqual("cell-preserving-join-by-id", evidence["derivation"])

    def test_population_mismatch_and_duplicate_cells_fail_closed(self):
        pilot = [row("1", configuration) for configuration in ("U", "BF", "WF")]
        supplement = [row("2", configuration) for configuration in ("B", "W")]
        with self.assertRaisesRegex(ValueError, "populations differ"):
            combine_layer_ablation_rows(pilot, supplement, expected_records=1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            combine_layer_ablation_rows(
                pilot,
                [row("1", "B"), row("1", "B")],
                expected_records=1,
            )

    def test_source_or_reference_mismatch_fails_closed(self):
        pilot = [row("1", configuration) for configuration in ("U", "BF", "WF")]
        supplement = [row("1", "B"), row("1", "W", source="other")]
        with self.assertRaisesRegex(ValueError, "source/reference mismatch"):
            combine_layer_ablation_rows(pilot, supplement, expected_records=1)


if __name__ == "__main__":
    unittest.main()
