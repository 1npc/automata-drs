from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from automata_drs.subword import ByteLevelVocabulary, gpt2_byte_decoder


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_drs_horizon_false_negatives.py"
CONFIG = ROOT / "configs" / "wcd_drs_horizon_test128_false_negative_audit_v1.json"
RUNNER = ROOT / "scripts/run_colossus_drs_horizon_false_negative_audit_v1.sh"
SPEC = importlib.util.spec_from_file_location(
    "audit_drs_horizon_false_negatives", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fake_tokenizer():
    inverse = gpt2_byte_decoder()
    encoder = {byte: character for character, byte in inverse.items()}

    class ByteLevel:
        pass

    class Backend:
        decoder = ByteLevel()

    class Tokenizer:
        backend_tokenizer = Backend()
        all_special_ids = [400]
        eos_token_id = 400

        @staticmethod
        def get_vocab():
            vocabulary = {encoder[byte]: byte for byte in range(256)}
            vocabulary["<eos>"] = 400
            return vocabulary

        @staticmethod
        def __call__(text, *, add_special_tokens):
            del add_special_tokens
            return {"input_ids": list(text.encode("utf-8"))}

        @staticmethod
        def decode(token_ids, **kwargs):
            del kwargs
            return bytes(
                token_id for token_id in token_ids if 0 <= token_id < 256
            ).decode("utf-8", errors="replace")

    return Tokenizer()


class DRSFalseNegativeAuditTest(unittest.TestCase):
    def test_contract_is_full_population_posthoc_cpu_only(self):
        contract = MODULE.load_contract(CONFIG)
        self.assertEqual("registered-posthoc-diagnostic", contract["status"])
        self.assertEqual(128, contract["source"]["records"])
        self.assertEqual(48, contract["source"]["max_new_tokens"])
        self.assertEqual(
            "all-128-W-rows-in-source-order", contract["audit"]["population"]
        )
        self.assertFalse(contract["audit"]["performance_conditioned_selection"])
        self.assertFalse(contract["resource_contract"]["model_weights_loaded"])
        self.assertFalse(contract["resource_contract"]["gpu_used"])
        self.assertEqual(1, contract["resource_contract"]["cpu_math_threads"])

        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn(hashlib.sha256(CONFIG.read_bytes()).hexdigest(), source)
        self.assertIn("OMP_NUM_THREADS=1", source)
        self.assertIn("TOKENIZERS_PARALLELISM=false", source)
        self.assertIn("nice -n 15", source)
        self.assertIn("ulimit -f 1024", source)
        self.assertIn("--verify-existing", source)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", source)
        self.assertNotIn("nvidia-smi", source)
        self.assertNotIn("rm -rf", source)
        completed = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual("", completed.stderr)
        self.assertEqual(0, completed.returncode)

    def test_observed_accepting_suffix_proves_planner_false_negative(self):
        tokenizer = fake_tokenizer()
        vocabulary = ByteLevelVocabulary.from_tokenizer(tokenizer)
        prediction = "person.n.01 Role +1 freak.n.01"
        tokens = len(prediction.encode("utf-8")) + 1
        row = {
            "id": "toy:false-negative",
            "prediction": prediction,
            "validity": {"raw_sbn": True, "raw_clf_referee": True},
            "generation": {
                "output_tokens_including_eos": tokens,
                "canonical_tokenization": True,
                "ended_with_eos": True,
                "hit_max_new_tokens": False,
            },
        }
        result = MODULE.audit_w_path(
            row,
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            horizon=tokens,
        )
        self.assertTrue(result["auditable"])
        self.assertTrue(result["source_path_replay_accepting"])
        self.assertFalse(result["planner_accepts_entire_path"])
        self.assertGreaterEqual(len(result["false_negative_events"]), 1)
        self.assertEqual([], result["other_rejection_events"])
        first = result["false_negative_events"][0]
        self.assertEqual("1", first["token_fragment"]["utf8"])
        self.assertEqual("unknown", first["planner_status"])
        self.assertTrue(first["observed_residual_verifier_accepted"])
        self.assertEqual(
            first["remaining_budget_after_token"],
            first["observed_residual_tokens"],
        )

    def test_non_boundary_path_is_admitted(self):
        tokenizer = fake_tokenizer()
        vocabulary = ByteLevelVocabulary.from_tokenizer(tokenizer)
        prediction = "male.n.02"
        tokens = len(prediction.encode("utf-8")) + 1
        row = {
            "id": "toy:accepted",
            "prediction": prediction,
            "validity": {"raw_sbn": True, "raw_clf_referee": True},
            "generation": {
                "output_tokens_including_eos": tokens,
                "canonical_tokenization": True,
                "ended_with_eos": True,
                "hit_max_new_tokens": False,
            },
        }
        result = MODULE.audit_w_path(
            row,
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            horizon=tokens,
        )
        self.assertTrue(result["planner_accepts_entire_path"])
        self.assertEqual([], result["false_negative_events"])
        self.assertEqual([], result["other_rejection_events"])


if __name__ == "__main__":
    unittest.main()
