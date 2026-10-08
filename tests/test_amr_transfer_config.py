from __future__ import annotations

import json
from pathlib import Path
import unittest

from automata_drs.penman import sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class AMRTransferConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (REPOSITORY_ROOT / "configs" / "amr_transfer_experiment.json").read_text(
                encoding="utf-8"
            )
        )

    def test_data_and_audit_hashes_are_bound(self):
        data = self.config["data"]
        manifest = REPOSITORY_ROOT / data["manifest"]
        self.assertEqual(data["manifest_sha256"], sha256_file(manifest))
        manifest_value = json.loads(manifest.read_text(encoding="utf-8"))

        for split in ("train", "development", "test"):
            row = data[split]
            corpus = REPOSITORY_ROOT / row["path"]
            self.assertEqual(row["sha256"], sha256_file(corpus))
            file_name = corpus.name
            self.assertEqual(row["records"], manifest_value["files"][file_name]["examples"])
            if "audit" not in row:
                continue
            audit = REPOSITORY_ROOT / row["audit"]
            self.assertEqual(row["audit_sha256"], sha256_file(audit))
            audit_value = json.loads(audit.read_text(encoding="utf-8"))
            self.assertTrue(audit_value["gate"]["passed"])
            self.assertEqual(row["records"], audit_value["corpus"]["records"])
            self.assertEqual(1.0, audit_value["official_decode"]["acceptance_rate"])
            self.assertEqual(1.0, audit_value["local_raw"]["acceptance_rate"])

        normalization = data["train"]["normalization"]
        derivation = REPOSITORY_ROOT / normalization["derivation_audit"]
        self.assertEqual(
            normalization["derivation_audit_sha256"], sha256_file(derivation)
        )
        derivation_value = json.loads(derivation.read_text(encoding="utf-8"))
        self.assertTrue(derivation_value["gate"]["passed"])
        self.assertEqual("NFC", normalization["form"])
        self.assertEqual([50632], normalization["changed_records"])
        self.assertEqual(
            data["train"]["sha256"], derivation_value["derived"]["sha256"]
        )

    def test_host_matches_frozen_recent_model_registry(self):
        registry = json.loads(
            (REPOSITORY_ROOT / "configs" / "lora_baselines.json").read_text(
                encoding="utf-8"
            )
        )
        expected = next(
            model
            for model in registry["models"]
            if model["key"] == self.config["host"]["key"]
        )
        for key in ("repo_id", "revision", "model_loader"):
            self.assertEqual(expected[key], self.config["host"][key])

    def test_first_pilot_does_not_claim_unimplemented_finite_planning(self):
        self.assertEqual(256, self.config["development_pilot"]["records"])
        self.assertEqual(["U", "P"], self.config["development_pilot"]["conditions"])
        self.assertIn("PF", self.config["conditions"])
        self.assertNotIn("PF", self.config["development_pilot"]["conditions"])
        order = self.config["execution_order"]
        self.assertLess(order.index("full-development-u-p"), order.index("heldout-test-u-p"))
        self.assertEqual("prepared-not-launched", self.config["status"])

    def test_smatch_is_raw_primary_and_never_drops_invalid_rows(self):
        contract = self.config["evaluation"]["smatch_contract"]
        self.assertEqual("raw_prediction", contract["primary_surface"])
        self.assertIn("zero matches", contract["invalid_surface_policy"])
        self.assertIn("no example is dropped", contract["invalid_surface_policy"])
        self.assertIn("no sentinel graph is scored", contract["invalid_surface_policy"])

    def test_beam_diagnostic_forbids_unbounded_step_score_history(self):
        beam = self.config["evaluation"]["beam_diagnostic"]
        resources = self.config["resource_contract"]
        self.assertEqual(4, beam["num_beams"])
        self.assertEqual(4, beam["num_return_sequences"])
        self.assertEqual(1.0, beam["length_penalty"])
        self.assertFalse(beam["candidate_score_history"])
        self.assertEqual(64, beam["score_chunk_tokens"])
        self.assertEqual(
            "forbidden", resources["beam_stepwise_full_vocabulary_score_history"]
        )
        self.assertEqual(2, resources["beam_likelihood_candidates_per_example_max"])
        artifacts = self.config["implementation_artifacts"]
        for name in (
            "beam_candidates",
            "beam_selection",
            "beam_merge",
            "beam_analysis",
            "suite_summary",
        ):
            self.assertTrue((REPOSITORY_ROOT / artifacts[name]).is_file())


if __name__ == "__main__":
    unittest.main()
