from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPOSITORY_ROOT / "scripts" / "generate_amr_beam_candidates.py"
MERGER = REPOSITORY_ROOT / "scripts" / "merge_amr_beam_selection.py"
SCORER = REPOSITORY_ROOT / "scripts" / "score_amr_predictions.py"
ANALYZER = REPOSITORY_ROOT / "scripts" / "analyze_amr_beam.py"
PYTHON = REPOSITORY_ROOT / ".venv" / "bin" / "python"


def _load_generator():
    spec = importlib.util.spec_from_file_location("amr_beam_generator_test", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AMRBeamScriptTest(unittest.TestCase):
    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_chunked_score_matches_full_teacher_forcing(self):
        torch.manual_seed(7)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=19,
                n_positions=32,
                n_ctx=32,
                n_embd=16,
                n_layer=1,
                n_head=2,
                use_cache=True,
                bos_token_id=1,
                eos_token_id=2,
            )
        ).eval()
        prompt = torch.tensor([[1, 2, 3, 4]])
        suffix = torch.tensor([5, 6, 7, 8, 9])
        full = torch.cat((prompt, suffix.unsqueeze(0)), dim=-1)
        with torch.inference_mode():
            logits = model(full).logits[:, :-1, :]
        targets = full[:, 1:]
        token_logprobs = torch.log_softmax(logits.float(), dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        expected = float(token_logprobs[:, prompt.shape[-1] - 1 :].sum().item())

        module = _load_generator()
        observed, mean = module._stream_score_suffix(
            model, torch, prompt, suffix, chunk_tokens=2
        )
        self.assertAlmostEqual(expected, observed, places=5)
        self.assertAlmostEqual(expected / len(suffix), mean, places=5)

    def test_generator_disables_score_history_and_bounds_rescoring(self):
        source = GENERATOR.read_text(encoding="utf-8")
        self.assertIn("output_scores=False", source)
        self.assertIn("return_dict_in_generate=False", source)
        self.assertIn("score-chunk-tokens", source)
        self.assertIn("required_score_ranks = {0, selected_rank}", source)
        self.assertNotIn("generated.sequences_scores", source)

    def test_transformers_beam_order_is_unchanged_without_score_history(self):
        torch.manual_seed(11)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=19,
                n_positions=32,
                n_ctx=32,
                n_embd=16,
                n_layer=1,
                n_head=2,
                use_cache=True,
                bos_token_id=1,
                eos_token_id=None,
                pad_token_id=0,
            )
        ).eval()
        prompt = torch.tensor([[1, 2, 3]])
        plain = model.generate(
            prompt,
            do_sample=False,
            num_beams=3,
            num_return_sequences=3,
            max_new_tokens=4,
            length_penalty=1.0,
            return_dict_in_generate=False,
            output_scores=False,
            pad_token_id=0,
        )
        rich = model.generate(
            prompt,
            do_sample=False,
            num_beams=3,
            num_return_sequences=3,
            max_new_tokens=4,
            length_penalty=1.0,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=0,
        )
        self.assertTrue(torch.equal(plain, rich.sequences))
        self.assertIsNotNone(rich.sequences_scores)
        self.assertTrue(
            bool(torch.all(rich.sequences_scores[:-1] >= rich.sequences_scores[1:]))
        )

    def test_scripts_compile(self):
        for script in (GENERATOR, MERGER, ANALYZER):
            result = subprocess.run(
                [str(PYTHON), "-m", "py_compile", str(script)],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)

    def test_three_condition_analysis_is_paired_and_raw_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "dev.amr"
            corpus.write_text(
                "A.\n\n(a / alpha)\n\n\nB.\n\n(b / beta)\n", encoding="utf-8"
            )
            predictions = root / "beam.jsonl"
            rows = []
            for ordinal, graph in ((1, "(a / alpha)"), (2, "(b / beta)")):
                for condition in ("U", "PostP", "P"):
                    selective_refusal = ordinal == 2 and condition == "P"
                    raw = (
                        "invalid"
                        if ordinal == 2 and condition in {"U", "P"}
                        else graph
                    )
                    row = {
                        "task": "amr",
                        "id": f"{corpus}:record:{ordinal}",
                        "input": "A." if ordinal == 1 else "B.",
                        "reference": graph,
                        "prediction": raw if raw.startswith("(") else "",
                        "raw_prediction": raw,
                        "configuration": condition,
                        "decode": {"strategy": "beam", "num_beams": 4},
                        "validity": {
                            "raw_local_penman": raw.startswith("("),
                            "raw_official_penman": raw.startswith("("),
                        },
                        "timing": {"generation_ms": 10.0, "constraint_ms": 0.0},
                        "score": {
                            "raw_model_logprob": -10.0,
                            "raw_model_mean_logprob": -1.0,
                        },
                        "generation": {
                            "ended_with_eos": not selective_refusal,
                            "hit_max_new_tokens": selective_refusal,
                        },
                        "constraint": (
                            {"all_masked_rows": int(selective_refusal)}
                            if condition == "P"
                            else None
                        ),
                    }
                    if condition == "PostP":
                        selected_rank = 0 if ordinal == 1 else 1
                        row["beam_selection"] = {
                            "method": "first-raw-penman-valid-in-final-beam-order",
                            "ranking_method": (
                                "transformers-final-beam-order-length-penalty-1"
                            ),
                            "selected_candidate_rank": selected_rank,
                            "fallback_to_u_top1": False,
                            "candidate_count": 4,
                            "raw_penman_valid_candidate_count": 2,
                            "raw_penman_valid_candidate_ranks": (
                                [0, 1] if ordinal == 1 else [1, 2]
                            ),
                        }
                    rows.append(row)
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            Path(str(predictions) + ".manifest.json").write_text(
                json.dumps(
                    {
                        "task": "amr",
                        "status": "running",
                        "path": str(corpus),
                        "start": 0,
                        "limit": -1,
                        "line_numbers": None,
                        "configurations": ["U", "PostP", "P"],
                        "schedule": "fixed",
                        "num_beams": 4,
                        "derivation": (
                            "u-top1-and-first-raw-penman-valid-u-beam-plus-online-p"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            scores = root / "scores.json"
            predecessor = root / "dev-gate.json"
            predecessor.write_text(
                json.dumps({"gate": {"passed": True}}), encoding="utf-8"
            )
            scoring = subprocess.run(
                [str(PYTHON), str(SCORER), str(predictions), "--output", str(scores)],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, scoring.returncode, scoring.stderr)
            report_path = root / "analysis.json"
            analysis = subprocess.run(
                [
                    str(PYTHON),
                    str(ANALYZER),
                    str(predictions),
                    "--scores",
                    str(scores),
                    "--predecessor-gate",
                    str(predecessor),
                    "--output",
                    str(report_path),
                    "--resamples",
                    "100",
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, analysis.returncode, analysis.stderr)
            verified = subprocess.run(
                [
                    str(PYTHON),
                    str(ANALYZER),
                    str(predictions),
                    "--scores",
                    str(scores),
                    "--predecessor-gate",
                    str(predecessor),
                    "--verify-existing",
                    str(report_path),
                    "--resamples",
                    "100",
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, verified.returncode, verified.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(2, report["schema_version"])
        self.assertEqual(1, report["search"]["online_p_nonaccepting_rows"])
        self.assertEqual(1, report["search"]["online_p_all_masked_rows"])
        self.assertEqual(1, report["search"]["truncations"]["P"])
        self.assertGreater(
            report["comparisons"]["postp_minus_u"]["corpus_smatch_delta"], 0.0
        )
        self.assertEqual(0, report["selection"]["fallbacks"])
        self.assertFalse(report["selection"]["numeric_step_score_history_retained"])

    def test_merger_binds_candidate_and_online_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "dev.amr"
            corpus.write_text("A.\n\n(a / alpha)\n", encoding="utf-8")
            config = root / "config.json"
            config.write_text(
                json.dumps({"host": {"repo_id": "model", "revision": "rev"}}),
                encoding="utf-8",
            )
            identifier = f"{corpus}:record:1"
            candidates = root / "candidates.jsonl"
            candidate_rows = []
            for rank in range(4):
                valid = rank in {1, 3}
                scored = rank in {0, 1}
                raw = "(a / alpha)" if valid else f"invalid-{rank}"
                candidate_rows.append(
                    {
                        "task": "amr",
                        "id": identifier,
                        "input": "A.",
                        "reference": "(a / alpha)",
                        "prediction": raw if valid else "",
                        "raw_prediction": raw,
                        "configuration": "U4",
                        "decode": {
                            "strategy": "beam",
                            "num_beams": 4,
                            "num_return_sequences": 4,
                            "output_scores": False,
                            "length_penalty": 1.0,
                        },
                        "beam_candidate": {
                            "rank": rank,
                            "ranking_method": (
                                "transformers-final-beam-order-length-penalty-1"
                            ),
                            "sequence_score": None,
                            "validation_ms": 0.1,
                        },
                        "validity": {
                            "raw_local_penman": valid,
                            "raw_official_penman": valid,
                        },
                        "timing": {"generation_ms": 1.0, "constraint_ms": 0.0},
                        "score": {
                            "raw_model_logprob": -2.0 if scored else None,
                            "raw_model_mean_logprob": -1.0 if scored else None,
                            "scored_tokens": 2 if scored else None,
                        },
                        "generation": {
                            "ended_with_eos": True,
                            "hit_max_new_tokens": False,
                        },
                        "constraint": None,
                    }
                )
            candidates.write_text(
                "".join(json.dumps(row) + "\n" for row in candidate_rows),
                encoding="utf-8",
            )
            common = {
                "model": "model",
                "model_loader": "causal-lm",
                "requested_revision": "rev",
                "resolved_revision": "rev",
                "adapter": "adapter",
                "path": str(corpus),
                "start": 0,
                "limit": -1,
                "max_input_tokens": 32,
                "max_new_tokens": 16,
                "source_sha256": self._sha256(corpus),
                "experiment_config_sha256": self._sha256(config),
            }
            Path(str(candidates) + ".manifest.json").write_text(
                json.dumps(
                    {
                        **common,
                        "status": "complete",
                        "kind": "amr-unconstrained-beam-candidates",
                        "task": "amr",
                        "configuration": "U4",
                        "num_beams": 4,
                        "num_return_sequences": 4,
                        "output_scores": False,
                        "score_policy": (
                            "chunked-teacher-forcing-for-u-top1-and-postp-only"
                        ),
                        "expected_examples": 1,
                        "output_sha256": self._sha256(candidates),
                    }
                ),
                encoding="utf-8",
            )
            online = root / "online.jsonl"
            online_row = {
                "task": "amr",
                "id": identifier,
                "input": "A.",
                "reference": "(a / alpha)",
                "prediction": "(a / alpha)",
                "raw_prediction": "(a / alpha)",
                "configuration": "P",
                "decode": {"strategy": "beam", "num_beams": 4},
                "validity": {
                    "raw_local_penman": True,
                    "raw_official_penman": True,
                },
                "timing": {"generation_ms": 2.0, "constraint_ms": 0.5},
                "score": {
                    "raw_model_logprob": -2.1,
                    "raw_model_mean_logprob": -1.05,
                },
                "generation": {
                    "ended_with_eos": True,
                    "hit_max_new_tokens": False,
                },
                "constraint": {"all_masked_rows": 0},
            }
            online.write_text(json.dumps(online_row) + "\n", encoding="utf-8")
            Path(str(online) + ".manifest.json").write_text(
                json.dumps(
                    {
                        **common,
                        "task": "amr",
                        "status": "running",
                        "line_numbers": None,
                        "configurations": ["P"],
                        "schedule": "fixed",
                        "num_beams": 4,
                        "score_model_logprob": True,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "matrix.jsonl"
            result = subprocess.run(
                [
                    str(PYTHON),
                    str(MERGER),
                    "--candidates",
                    str(candidates),
                    "--online-p",
                    str(online),
                    "--output",
                    str(output),
                    "--config",
                    str(config),
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            verified = subprocess.run(
                [
                    str(PYTHON),
                    str(MERGER),
                    "--candidates",
                    str(candidates),
                    "--online-p",
                    str(online),
                    "--verify-existing",
                    str(output),
                    "--config",
                    str(config),
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, verified.returncode, verified.stderr)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            manifest = json.loads(
                Path(str(output) + ".manifest.json").read_text(encoding="utf-8")
            )
            output_sha256 = self._sha256(output)

        self.assertEqual(["U", "PostP", "P"], [row["configuration"] for row in rows])
        self.assertEqual(1, rows[1]["beam_selection"]["selected_candidate_rank"])
        self.assertEqual("complete", manifest["status"])
        self.assertEqual(output_sha256, manifest["output_sha256"])


if __name__ == "__main__":
    unittest.main()
