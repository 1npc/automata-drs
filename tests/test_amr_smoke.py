from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from automata_drs.generation_manifest import complete_generation_manifest, sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY_ROOT / "scripts" / "run_colossus_amr_smoke.sh"
CHECKER = REPOSITORY_ROOT / "scripts" / "check_amr_smoke.py"
PYTHON = REPOSITORY_ROOT / ".venv" / "bin" / "python"


class AMRSmokeTest(unittest.TestCase):
    def test_runner_revalidates_preflight_before_gpu_claim(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("--verify-existing", source)
        self.assertIn("automata_gpu_claim", source)
        self.assertIn("AUTOMATA_DRS_GPU_INDEX:-0", source)
        self.assertIn("AUTOMATA_DRS_ANALYSIS_PYTHON", source)
        self.assertIn("AUTOMATA_DRS_ALLOW_DISJOINT_GPU_OVERLAP", source)
        self.assertIn("AUTOMATA_DRS_LORA_CACHE", source)
        self.assertIn("--train-record-numbers", source)
        self.assertIn("--task amr", source)
        self.assertIn("--configurations U P", source)
        self.assertIn("audit_penman_certified_release.py", source)
        self.assertIn("OMP_NUM_THREADS=1", source)
        self.assertLess(source.index("--verify-existing"), source.index("automata_gpu_claim"))
        self.assertLess(source.index("pgrep -f"), source.index("automata_gpu_claim"))
        result = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_trainer_persists_peak_cuda_memory_for_full_run_planning(self):
        source = (
            REPOSITORY_ROOT / "scripts" / "train_causal_lm_lora.py"
        ).read_text(encoding="utf-8")
        self.assertIn("torch.cuda.reset_peak_memory_stats()", source)
        self.assertIn("torch.cuda.max_memory_allocated()", source)
        self.assertIn("torch.cuda.max_memory_reserved()", source)
        self.assertIn('"cuda_memory": cuda_memory', source)

    def make_fixture(
        self, root: Path, *, refuse_p: bool = False
    ) -> tuple[Path, Path]:
        config = json.loads(
            (REPOSITORY_ROOT / "configs" / "amr_transfer_experiment.json").read_text(
                encoding="utf-8"
            )
        )
        registry = json.loads(
            (REPOSITORY_ROOT / "configs" / "lora_baselines.json").read_text(
                encoding="utf-8"
            )
        )
        model = next(row for row in registry["models"] if row["key"] == config["host"]["key"])
        ordinals = list(range(1, 17))
        preflight = {
            "inputs": {
                "config_sha256": sha256_file(
                    REPOSITORY_ROOT / "configs" / "amr_transfer_experiment.json"
                )
            },
            "recommendation": {
                "max_length": 2048,
                "max_input_tokens": 256,
                "max_new_tokens": 1536,
                "smoke_train_record_ordinals": ordinals,
                "smoke_per_device_batch_size": 1,
                "smoke_gradient_accumulation_steps": 16,
                "smoke_max_steps": 8,
            },
            "gate": {"passed": True},
        }
        preflight_path = root / "preflight.json"
        preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

        run_root = root / "smoke"
        adapter = run_root / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        train_manifest = {
            "status": "complete",
            "task": "amr",
            "model": config["host"]["repo_id"],
            "model_loader": config["host"]["model_loader"],
            "requested_revision": config["host"]["revision"],
            "resolved_revision": config["host"]["revision"],
            "train_source_sha256": config["data"]["train"]["sha256"],
            "eval_source_sha256": config["data"]["development"]["sha256"],
            "train_record_numbers": ordinals,
            "train": {"records": 16, "max_sequence_tokens": 1900},
            "eval": {"records": 16, "max_sequence_tokens": 1800},
            "recipe": {
                "max_length": 2048,
                "max_steps": 8,
                "batch_size": 1,
                "gradient_accumulation": 16,
                "target_modules": model["training_target_modules_manifest"],
                "learning_rate": config["training"]["learning_rate"],
                "warmup_ratio": config["training"]["warmup_ratio"],
                "lora_rank": config["training"]["lora_rank"],
                "lora_alpha": config["training"]["lora_alpha"],
                "lora_dropout": config["training"]["lora_dropout"],
                "seed": config["training"]["seed"],
                "precision": "bf16",
                "torch_threads": 2,
                "dataloader_num_workers": 0,
                "dataset_storage": "compact-uint32-input-ids; masks-and-labels-materialized-per-item",
            },
        }
        (run_root / "train_manifest.json").write_text(
            json.dumps(train_manifest), encoding="utf-8"
        )
        (run_root / "train_metrics.json").write_text(
            json.dumps(
                {
                    "train": {"train_loss": 0.2},
                    "eval": {"eval_loss": 0.1},
                    "wall_seconds": 12.0,
                    "cuda_memory": {
                        "peak_allocated_bytes": 20_000_000_000,
                        "peak_reserved_bytes": 22_000_000_000,
                        "device_total_bytes": 48_000_000_000,
                    },
                }
            ),
            encoding="utf-8",
        )

        corpus = Path("data/amr/validation.amr")
        predictions = run_root / "dev-smoke-u-p-greedy.jsonl"
        rows = []
        for ordinal in range(1, 5):
            configurations = ("U", "P") if (ordinal - 1) % 2 == 0 else ("P", "U")
            for configuration in configurations:
                refused = refuse_p and configuration == "P"
                rows.append(
                    {
                        "id": f"{corpus}:record:{ordinal}",
                        "reference": "(a / alpha)",
                        "raw_prediction": "(a" if refused else "(a / alpha)",
                        "prediction": "" if refused else "(a / alpha)",
                        "configuration": configuration,
                        "decode": {"strategy": "greedy", "num_beams": 1},
                        "validity": {
                            "raw_local_penman": not refused,
                            "raw_official_penman": not refused,
                        },
                        "timing": {"generation_ms": 10.0, "constraint_ms": 1.0},
                        "score": {"raw_model_logprob": -1.0},
                        "generation": {
                            "hit_max_new_tokens": refused,
                            "ended_with_eos": not refused,
                            "peak_gpu_bytes": 1_000_000,
                        },
                        "constraint": (
                            None
                            if configuration == "U"
                            else {
                                "calls": 10,
                                "rows": 10,
                                "allowed_min": 1,
                                "allowed_total": 10,
                                "all_masked_rows": 0,
                            }
                        ),
                    }
                )
        predictions.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        generation_manifest_path = Path(str(predictions) + ".manifest.json")
        generation_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "causal-lm-generation-matrix",
                    "task": "amr",
                    "status": "running",
                    "model": config["host"]["repo_id"],
                    "model_loader": config["host"]["model_loader"],
                    "requested_revision": config["host"]["revision"],
                    "adapter": str(run_root / "adapter"),
                    "path": str(corpus),
                    "start": 0,
                    "limit": 4,
                    "line_numbers": None,
                    "configurations": ["U", "P"],
                    "schedule": "rotate",
                    "num_beams": 1,
                    "batch_size": 1,
                    "score_model_logprob": True,
                    "max_input_tokens": 256,
                    "max_new_tokens": 1536,
                    "experiment_config_sha256": sha256_file(
                        REPOSITORY_ROOT / "configs" / "amr_transfer_experiment.json"
                    ),
                    "cuda_device_total_bytes": 48_000_000_000,
                }
            ),
            encoding="utf-8",
        )
        complete_generation_manifest(predictions, generation_manifest_path)
        scores = {
            "predictions": {"sha256": sha256_file(predictions)},
            "groups": {"U": {"records": 4}, "P": {"records": 4}},
            "gate": {"passed": True},
        }
        (run_root / "dev-smoke-u-p-greedy.smatch.json").write_text(
            json.dumps(scores), encoding="utf-8"
        )
        paired = {
            "population": {"records": 4, "paired": True},
            "search": {
                "p_all_masked_rows": 0,
                "truncations": {"P": 4 if refuse_p else 0, "U": 0},
            },
            "gate": {"passed": True},
        }
        (run_root / "dev-smoke-u-p-greedy.paired.json").write_text(
            json.dumps(paired), encoding="utf-8"
        )
        release_audit = {
            "schema_version": 2,
            "audit": "penman-certified-release",
            "input": {"predictions_sha256": sha256_file(predictions)},
            "population": {"examples": 4, "rows": 4},
            "release": {
                "releasable_rows": 0 if refuse_p else 4,
                "refused_rows": 4 if refuse_p else 0,
                "guarantee_counts": (
                    {}
                    if refuse_p
                    else {"final-local-and-independent-penman-membership": 4}
                ),
                "reason_counts": (
                    {"generation-hit-hard-horizon": 4} if refuse_p else {}
                ),
                "refusals": (
                    [
                        {
                            "id": f"{corpus}:record:{ordinal}",
                            "reasons": ["generation-hit-hard-horizon"],
                        }
                        for ordinal in range(1, 5)
                    ]
                    if refuse_p
                    else []
                ),
            },
            "gate": {"passed": True, "performance_threshold_used": False},
        }
        (run_root / "dev-smoke-p-certified-release.json").write_text(
            json.dumps(release_audit), encoding="utf-8"
        )
        return run_root, preflight_path

    def run_checker(self, run_root: Path, preflight: Path, output: Path):
        return subprocess.run(
            [
                str(PYTHON),
                str(CHECKER),
                str(run_root),
                "--preflight-gate",
                str(preflight),
                "--output",
                str(output),
            ],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
        )

    def test_smoke_checker_recomputes_promotion_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root, preflight = self.make_fixture(root)
            output = root / "smoke-gate.json"
            result = self.run_checker(run_root, preflight, output)
            self.assertEqual(0, result.returncode, result.stderr)
            verified = subprocess.run(
                [
                    str(PYTHON),
                    str(CHECKER),
                    str(run_root),
                    "--preflight-gate",
                    str(preflight),
                    "--verify-existing",
                    str(output),
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, verified.returncode, verified.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(report["gate"]["passed"])

    def test_smoke_checker_allows_fully_accounted_selective_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root, preflight = self.make_fixture(root, refuse_p=True)
            output = root / "smoke-gate.json"
            result = self.run_checker(run_root, preflight, output)
            self.assertEqual(0, result.returncode, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(report["gate"]["passed"])
        self.assertFalse(report["gate"]["performance_threshold_used"])
        self.assertEqual(0, report["reported_outcomes"]["p_releasable_rows"])
        self.assertEqual(4, report["reported_outcomes"]["p_refused_rows"])
        self.assertEqual(4, report["reported_outcomes"]["truncations"]["P"])

    def test_smoke_checker_rejects_tampered_completed_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root, preflight = self.make_fixture(root)
            predictions = run_root / "dev-smoke-u-p-greedy.jsonl"
            rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
            p_row = next(row for row in rows if row["configuration"] == "P")
            p_row["validity"]["raw_official_penman"] = False
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            output = root / "smoke-gate.json"
            result = self.run_checker(run_root, preflight, output)
            self.assertNotEqual(0, result.returncode)


if __name__ == "__main__":
    unittest.main()
