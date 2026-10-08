from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

from automata_drs import SBNPolicy
from automata_drs.subword import (
    ByteLevelVocabulary,
    SubwordSBNLogitsProcessor,
    gpt2_byte_decoder,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "audit_subword_horizon_stress.py"
RUNNER = REPOSITORY_ROOT / "scripts" / "run_colossus_qwen_drs_horizon_stress.sh"
CONFIG = REPOSITORY_ROOT / "configs" / "qwen_drs_horizon_stress.json"
RUNNER_V2 = (
    REPOSITORY_ROOT / "scripts" / "run_colossus_qwen_drs_horizon_stress_v2.sh"
)
RUNNER_CLOSURE_BRIDGE = REPOSITORY_ROOT / (
    "scripts/run_colossus_qwen_drs_horizon_stress_closure_bridge.sh"
)
CONFIG_V2 = REPOSITORY_ROOT / "configs" / "qwen_drs_horizon_stress_v2.json"
SPEC = importlib.util.spec_from_file_location("audit_subword_horizon_stress", SCRIPT)
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


class QwenHorizonStressTest(unittest.TestCase):
    def make_processor(self, horizon: int):
        tokenizer = fake_tokenizer()
        vocabulary = ByteLevelVocabulary.from_tokenizer(tokenizer)
        processor = SubwordSBNLogitsProcessor(
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            prompt_token_ids=(),
            policy=SBNPolicy.for_configuration("B"),
            max_new_tokens=horizon,
            bounded_completion=True,
            max_completion_cache_entries=100,
        )
        return tokenizer, vocabulary, processor

    def test_locally_live_path_is_rejected_only_when_horizon_is_too_short(self):
        row = {
            "id": "toy:1",
            "configuration": "B",
            "prediction": "male.n.02",
            "generation": {"ended_with_eos": True, "hit_max_new_tokens": False},
        }
        tokenizer, vocabulary, processor = self.make_processor(5)
        short = MODULE.audit_selected_path(
            row=row,
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            processor=processor,
            horizon=5,
        )
        self.assertFalse(short["path_within_horizon"])
        self.assertFalse(short["accepted"])
        self.assertTrue(short["first_rejection"]["unbounded_transition_live"])
        self.assertIsNone(short["first_rejection"]["bounded_witness_tokens"])

        tokenizer, vocabulary, processor = self.make_processor(10)
        fitting = MODULE.audit_selected_path(
            row=row,
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            processor=processor,
            horizon=10,
        )
        self.assertTrue(fitting["path_within_horizon"])
        self.assertTrue(fitting["accepted"])
        self.assertIsNone(fitting["first_rejection"])

    def test_summary_retains_source_failures_and_negative_results(self):
        outcomes = [
            {
                "id": "valid",
                "path_tokens_including_eos": 10,
                "path_within_horizon": True,
                "source_policy_valid": True,
                "source_policy_reason": None,
                "source_execution_complete": True,
                "accepted": True,
                "first_rejection": None,
            },
            {
                "id": "invalid",
                "path_tokens_including_eos": 4,
                "path_within_horizon": True,
                "source_policy_valid": False,
                "source_policy_reason": "invalid",
                "source_execution_complete": True,
                "accepted": False,
                "first_rejection": None,
            },
            {
                "id": "truncated",
                "path_tokens_including_eos": 12,
                "path_within_horizon": False,
                "source_policy_valid": True,
                "source_policy_reason": None,
                "source_execution_complete": False,
                "accepted": False,
                "first_rejection": None,
            },
        ]
        summary = MODULE.summarize_cell(
            outcomes,
            horizon=10,
            source_configuration="B",
            bounded_configuration="BF",
            max_examples=32,
        )
        self.assertEqual(3, summary["attempted"])
        self.assertEqual(1, summary["source_policy_invalid"])
        self.assertEqual(1, summary["source_execution_incomplete"])
        self.assertEqual(1, summary["accepted"])
        self.assertEqual(2, len(summary["examples"]))

    def test_contract_and_runner_are_frozen_cpu_only_post_primary(self):
        contract = MODULE.load_contract(CONFIG)
        self.assertEqual([64, 128, 192, 256], contract["audit"]["counterfactual_horizons"])
        self.assertFalse(contract["audit"]["performance_threshold_used"])
        self.assertFalse(contract["resource_contract"]["model_weights_loaded"])
        self.assertFalse(contract["resource_contract"]["gpu_used"])
        self.assertEqual(1, contract["resource_contract"]["cpu_math_threads"])

        source = RUNNER.read_text(encoding="utf-8")
        config_sha = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
        self.assertIn(config_sha, source)
        self.assertIn("recent-model-and-method-results-v3.promotion.json", source)
        self.assertIn("--scope full", source)
        self.assertIn("--verify-existing", source)
        self.assertIn("TOKENIZERS_PARALLELISM=false", source)
        self.assertIn("OMP_NUM_THREADS=1", source)
        self.assertIn("nice -n 15", source)
        self.assertIn("ulimit -f 8192", source)
        self.assertIn("run_colossus_lora_interventions_v2", source)
        self.assertIn("generate_causal_lm_baseline.py", source)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", source)
        self.assertNotIn("nvidia-smi", source)
        self.assertNotIn("rm -rf", source)

    def test_runner_has_valid_bash_syntax(self):
        completed = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual("", completed.stderr)
        self.assertEqual(0, completed.returncode)

    def test_v2_contract_is_fixed_first_128_and_closure_bound(self):
        contract = MODULE.load_contract(CONFIG_V2)
        self.assertEqual(2, contract["schema_version"])
        self.assertEqual(
            [64, 128, 192, 256], contract["audit"]["counterfactual_horizons"]
        )
        self.assertEqual(
            {
                "rule": "first-n-paired-source-order",
                "records": 128,
                "performance_conditioned": False,
                "reason": (
                    "Bound wall time while retaining a deterministic paired "
                    "prefix fixed before the stress report."
                ),
            },
            contract["source"]["selection"],
        )
        self.assertEqual(
            "6d6a549a4a63fd61f6b0d3a19e72ad2bc146098d0e690172c5af67668b0f1cb6",
            contract["prerequisites"]["closure_sha256"],
        )
        self.assertFalse(contract["audit"]["performance_threshold_used"])
        self.assertFalse(contract["resource_contract"]["model_weights_loaded"])
        self.assertFalse(contract["resource_contract"]["gpu_used"])
        self.assertEqual(1, contract["resource_contract"]["cpu_math_threads"])

        source = RUNNER_V2.read_text(encoding="utf-8")
        config_sha = hashlib.sha256(CONFIG_V2.read_bytes()).hexdigest()
        self.assertIn(config_sha, source)
        self.assertIn("pre-wcd-experiments-v1.json", source)
        self.assertIn("verify_pre_wcd_experiment_closure.py", source)
        self.assertIn("TOKENIZERS_PARALLELISM=false", source)
        self.assertIn("OMP_NUM_THREADS=1", source)
        self.assertIn("nice -n 15", source)
        self.assertIn("ulimit -f 4096", source)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", source)
        self.assertNotIn("nvidia-smi", source)
        self.assertNotIn("rm -rf", source)

    def test_v2_runner_has_valid_bash_syntax(self):
        completed = subprocess.run(
            ["bash", "-n", str(RUNNER_V2)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual("", completed.stderr)
        self.assertEqual(0, completed.returncode)

    def test_closure_bridge_preserves_original_registered_contract(self):
        source = RUNNER_CLOSURE_BRIDGE.read_text(encoding="utf-8")
        config_sha = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
        self.assertIn(config_sha, source)
        self.assertIn("configs/qwen_drs_horizon_stress.json", source)
        self.assertIn("dev400-b-w-horizon-stress.json", source)
        self.assertIn("pre-wcd-experiments-v1.json", source)
        self.assertIn("verify_pre_wcd_experiment_closure.py", source)
        self.assertIn("OMP_NUM_THREADS=1", source)
        self.assertIn("nice -n 15", source)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", source)
        self.assertNotIn("nvidia-smi", source)
        self.assertNotIn("rm -rf", source)

        completed = subprocess.run(
            ["bash", "-n", str(RUNNER_CLOSURE_BRIDGE)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual("", completed.stderr)
        self.assertEqual(0, completed.returncode)


if __name__ == "__main__":
    unittest.main()
