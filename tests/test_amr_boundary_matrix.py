from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_amr_boundary_matrix.py"
RUNNER = ROOT / "scripts" / "run_colossus_external_amr_boundary32_h288.sh"
CONFIG = ROOT / "configs" / "wcd_amr_boundary_dev32_h288_v1.json"
SPEC = importlib.util.spec_from_file_location("check_amr_boundary_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


LENGTHS = [
    12, 12, 75, 76, 81, 82, 95, 97, 97, 124, 133, 140, 159, 160, 171,
    177, 184, 188, 203, 217, 227, 228, 231, 246, 258, 259, 261, 266,
    279, 298, 363, 512,
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def generation_row(
    configuration: str,
    ordinal: int,
    *,
    invalid_u: bool = False,
) -> dict[str, object]:
    valid = not (configuration == "U" and invalid_u)
    prediction = f"(x{ordinal} / example)" if valid else "broken"
    constraint = None
    if configuration == "P":
        constraint = {"rows": 1, "all_masked_rows": 0, "allowed_total": 1}
    elif configuration == "PF":
        constraint = {
            "rows": 1,
            "all_masked_rows": 0,
            "allowed_total": 1,
            "completion_certificates_tested": 1,
            "completion_certificates_accepted": 1,
        }
    elif configuration == "WCD":
        constraint = {
            "rows": 1,
            "all_masked_rows": 0,
            "allowed_total": 1,
            "prefix_rows_certified": 1,
            "prefix_rows_unavailable": 0,
            "carried_tokens": 0,
            "cached_tokens": 0,
            "replanned_tokens": 1,
            "rejected_without_certificate": 0,
        }
    constrained = configuration != "U"
    return {
        "task": "amr",
        "id": f"validation.amr:record:{ordinal}",
        "input": f"Example {ordinal}.",
        "reference": f"(x{ordinal} / example)",
        "prediction": prediction,
        "raw_prediction": prediction,
        "configuration": configuration,
        "validity": {"raw_official_penman": valid},
        "generation": {
            "ended_with_eos": not constrained and valid,
            "ended_with_structural_completion": constrained,
            "hit_max_new_tokens": not valid,
            "canonical_tokenization": True,
            "output_tokens_including_eos": 10,
        },
        "timing": {"generation_ms": 10.0, "constraint_ms": float(constrained)},
        "constraint": constraint,
    }


def score_report(
    predictions: Path, rows: list[dict[str, object]], configurations: list[str]
) -> dict[str, object]:
    paired_rows = []
    groups = {}
    for configuration in configurations:
        selected = [row for row in rows if row["configuration"] == configuration]
        matches = sum(row["validity"]["raw_official_penman"] for row in selected)  # type: ignore[index]
        f1 = matches / len(selected)
        groups[configuration] = {
            "raw_primary_smatch": {
                "f1": f1,
                "matches": matches,
                "predicted_triples": matches,
                "gold_triples": len(selected),
            }
        }
        for row in selected:
            valid = bool(row["validity"]["raw_official_penman"])  # type: ignore[index]
            paired_rows.append({
                "configuration": configuration,
                "id": row["id"],
                "raw_official_penman": valid,
                "raw_smatchpp_scoreable": valid,
                "raw_smatch": {
                    "f1": 1.0 if valid else 0.0,
                    "matches": 1 if valid else 0,
                    "predicted_triples": 1 if valid else 0,
                    "gold_triples": 1,
                },
            })
    return {
        "scorer": {
            "name": "smatchpp",
            "version": "1.8.0",
            "solver": "ilp",
            "optimality_required": True,
        },
        "gate": {
            "passed": True,
            "all_alignments_optimal": True,
            "complete_identity_pairing": True,
        },
        "predictions": {"sha256": sha(predictions), "rows": len(rows)},
        "groups": groups,
        "paired_rows": paired_rows,
    }


class AMRBoundaryMatrixTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> Path:
        contract = json.loads(CONFIG.read_text(encoding="utf-8"))
        corpus = root / "validation.amr"
        corpus.write_text("synthetic frozen corpus\n", encoding="utf-8")
        contract["data"]["path"] = str(corpus)
        contract["data"]["sha256"] = sha(corpus)

        predecessor = root / "predecessor.jsonl"
        predecessor_rows = [
            {
                "configuration": "U",
                "id": f"validation.amr:record:{index}",
                "generation": {"output_tokens_including_eos": length},
            }
            for index, length in enumerate(LENGTHS, start=1)
        ]
        write_jsonl(predecessor, predecessor_rows)
        predecessor_manifest = root / "predecessor.manifest.json"
        predecessor_scores = root / "predecessor.scores.json"
        write_json(predecessor_manifest, {"status": "complete"})
        write_json(predecessor_scores, {"gate": {"passed": True}})
        contract["horizon_selection"].update({
            "predecessor_predictions": str(predecessor),
            "predecessor_predictions_sha256": sha(predecessor),
            "predecessor_manifest": str(predecessor_manifest),
            "predecessor_manifest_sha256": sha(predecessor_manifest),
            "predecessor_scores": str(predecessor_scores),
            "predecessor_scores_sha256": sha(predecessor_scores),
        })
        for predecessor_name in ("boundary_fix4_gate", "wcd4_gate"):
            path = root / f"{predecessor_name}.json"
            write_json(path, {"gate": {"passed": True}})
            contract["promotion_predecessors"][predecessor_name] = {
                "path": str(path), "sha256": sha(path)
            }

        files = {
            "up": ("up_predictions", "up_scores", ["U", "P"]),
            "pf": ("pf_predictions", "pf_scores", ["PF"]),
            "wcd": ("wcd_predictions", "wcd_scores", ["WCD"]),
        }
        for label, (prediction_key, score_key, configurations) in files.items():
            predictions = root / f"{label}.jsonl"
            rows = [
                generation_row(
                    configuration,
                    ordinal,
                    invalid_u=(ordinal == 32),
                )
                for ordinal in range(1, 33)
                for configuration in configurations
            ]
            write_jsonl(predictions, rows)
            expected_rows = 32 * len(configurations)
            manifest = {
                "status": "complete",
                "task": "amr",
                "model": contract["model"]["repo_id"],
                "model_loader": contract["model"]["model_loader"],
                "requested_revision": contract["model"]["revision"],
                "resolved_revision": contract["model"]["revision"],
                "adapter": None,
                "path": str(corpus),
                "start": 0,
                "limit": 32,
                "max_input_tokens": 512,
                "max_new_tokens": 288,
                "num_beams": 1,
                "batch_size": 1,
                "configurations": configurations,
                "score_model_logprob": False,
                "prompt_template": "viamr-qwen3-graph-prefill",
                "output_envelope": "viamr-amr-tag",
                "source_sha256": sha(corpus),
                "expected_examples": 32,
                "expected_rows": expected_rows,
                "examples": 32,
                "rows": expected_rows,
                "schedule": "rotate" if len(configurations) > 1 else "fixed",
                "stop_on_structural_completion": True,
                "output_sha256": sha(predictions),
                "physical_gpu_index": "1",
            }
            write_json(Path(str(predictions) + ".manifest.json"), manifest)
            scores = root / f"{label}.scores.json"
            write_json(scores, score_report(predictions, rows, configurations))
            contract["generation"][prediction_key] = str(predictions)
            contract["generation"][score_key] = str(scores)
        config = root / "config.json"
        write_json(config, contract)
        return config

    def test_complete_matrix_passes_and_reports_a_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.make_fixture(Path(temporary))
            report = MODULE.build_report(config)
        self.assertTrue(report["gate"]["passed"], report["gate"]["failures"])
        self.assertEqual(1, report["paired"]["WCD-U"]["validity"]["repairs"])
        self.assertEqual(32, report["equivalence"]["raw_prediction_exact"])
        self.assertFalse(report["gate"]["performance_threshold_used"])

    def test_pf_wcd_divergence_fails_the_invariant_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.make_fixture(root)
            contract = json.loads(config.read_text(encoding="utf-8"))
            predictions = Path(contract["generation"]["wcd_predictions"])
            rows = [json.loads(line) for line in predictions.read_text().splitlines()]
            rows[0]["prediction"] = rows[0]["raw_prediction"] = "(z / changed)"
            write_jsonl(predictions, rows)
            manifest = Path(str(predictions) + ".manifest.json")
            manifest_value = json.loads(manifest.read_text())
            manifest_value["output_sha256"] = sha(predictions)
            write_json(manifest, manifest_value)
            scores = Path(contract["generation"]["wcd_scores"])
            write_json(scores, score_report(predictions, rows, ["WCD"]))
            report = MODULE.build_report(config)
        self.assertFalse(report["gate"]["passed"])
        self.assertIn("WCD:PF-output-equivalence", report["gate"]["failures"])

    def test_registration_and_runner_are_frozen_and_resource_bounded(self):
        contract = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(288, contract["generation"]["max_new_tokens"])
        self.assertEqual([0, 1], contract["resources"]["allowed_physical_gpus"])
        self.assertFalse(
            contract["horizon_selection"][
                "boundary_outcomes_observed_before_registration"
            ]
        )
        self.assertEqual(32, contract["confirmation_window"]["start"])
        self.assertEqual(96, contract["confirmation_window"]["limit"])
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn(sha(CONFIG), source)
        self.assertIn("--start 0 --limit 32", source)
        self.assertIn("--max-new-tokens 288", source)
        self.assertIn("--configurations U P", source)
        self.assertIn("--method PF", source)
        self.assertIn("--method WCD", source)
        self.assertIn("OMP_NUM_THREADS=2", source)
        self.assertIn("automata_gpu_select_idle", source)
        self.assertNotIn("rm -rf", source)
        syntax = subprocess.run(
            ["bash", "-n", str(RUNNER)], text=True, capture_output=True
        )
        self.assertEqual(0, syntax.returncode, syntax.stderr)

    def test_gate_script_syntax(self):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(SCRIPT)],
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
