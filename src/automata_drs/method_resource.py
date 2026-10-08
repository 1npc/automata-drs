"""Fail-closed resource plan for the selected-host method experiment."""

from __future__ import annotations

from typing import Any


MAX_GENERATED_SURFACES = 30_000
MAX_THEORETICAL_OUTPUT_TOKENS = 8_000_000
SCRATCH_MARGIN_BYTES = 6 * 1024 * 1024 * 1024
MAX_COMPLETION_CACHE_ENTRIES_PER_EXAMPLE = 100_000


def _unique_matrix(method: dict[str, Any], split: str) -> dict[str, Any]:
    matches = [
        row
        for row in method.get("publication_matrices", [])
        if row.get("split") == split
    ]
    if len(matches) != 1:
        raise ValueError(f"publication matrix is missing or duplicated: {split}")
    return matches[0]


def build_method_resource_plan(
    config: dict[str, Any],
    selection: dict[str, Any],
    *,
    config_sha256: str,
    selection_sha256: str,
) -> dict[str, Any]:
    """Compute a conservative, configuration-bound method-suite upper bound."""

    selected_key = str(selection.get("selected_key", ""))
    models = [row for row in config.get("models", []) if row.get("key") == selected_key]
    if selection.get("status") != "selected" or len(models) != 1:
        raise ValueError("resource plan requires one frozen selected host")
    model = models[0]
    method = config.get("method_experiment")
    if not isinstance(method, dict):
        raise ValueError("method_experiment is missing")

    pilot = method.get("pilot", {})
    layer = method.get("layer_ablation", {})
    fine = method.get("fine_grained_ablation", {})
    beam = method.get("beam_selection_study", {})
    intervention = method.get("conditioned_intervention", {})
    if pilot.get("configurations") != ["U", "BF", "WF"]:
        raise ValueError("pilot resource contract is not U/BF/WF")
    if layer.get("generated_configurations") != ["B", "W"]:
        raise ValueError("layer resource contract is not B/W")
    if fine.get("generated_configurations") != ["S", "R", "G"]:
        raise ValueError("fine resource contract is not S/R/G")
    if beam.get("generated_conditions") != ["U4-candidates", "W", "WF"]:
        raise ValueError("beam resource contract has changed")
    if intervention.get("configurations") != ["U", "B", "W", "BF", "WF"]:
        raise ValueError("intervention resource contract has changed")
    if intervention.get("splits") != ["dev", "test", "tax"]:
        raise ValueError("intervention split contract has changed")

    matrices = {split: _unique_matrix(method, split) for split in ("dev", "test", "tax")}
    for split, matrix in matrices.items():
        if matrix.get("configurations") != ["U", "BF", "WF"]:
            raise ValueError(f"publication resource contract changed: {split}")

    max_new_tokens = int(pilot.get("max_new_tokens", 0))
    horizons = {
        max_new_tokens,
        int(layer.get("max_new_tokens", 0)),
        int(fine.get("max_new_tokens", 0)),
        int(beam.get("max_new_tokens", 0)),
        int(intervention.get("max_new_tokens", 0)),
    }
    if horizons != {256}:
        raise ValueError(f"method horizons are not uniformly 256: {sorted(horizons)}")

    pilot_records = int(pilot["records"])
    tax_records = int(matrices["tax"]["records"])
    beam_records = int(beam["records"])
    beam_returns = int(beam["u_num_return_sequences"])
    stages = [
        {
            "stage": "pilot-batch-equivalence",
            "generated_surfaces_upper": 20,
            "note": "one U surface per record; batched output is checked against sequential U",
        },
        {
            "stage": "pilot-dev",
            "generated_surfaces_upper": pilot_records * 3,
            "note": "U/BF/WF on the frozen first-dev population",
        },
        {
            "stage": "publication-dev",
            "generated_surfaces_upper": int(matrices["dev"]["records"]) * 3,
            "note": "full dev U/BF/WF; includes the pilot prefix under a distinct immutable artifact",
        },
        {
            "stage": "layer-supplement",
            "generated_surfaces_upper": int(layer["records"]) * 2,
            "note": "B/W only; U/BF/WF rows are reused",
        },
        {
            "stage": "fine-supplement",
            "generated_surfaces_upper": int(fine["records"]) * 3,
            "note": "S/R/G only; the five-layer artifact is reused",
        },
        {
            "stage": "conditioned-dev-interventions-worst-case",
            "generated_surfaces_upper": int(matrices["dev"]["records"]) * 5,
            "note": "development-only upper bound assuming every U row fails",
        },
        {
            "stage": "beam-candidates-and-online",
            "generated_surfaces_upper": beam_records * (beam_returns + 2),
            "note": "four returned U candidates plus one W and one WF output per record",
        },
        {
            "stage": "heldout-test-publication",
            "generated_surfaces_upper": int(matrices["test"]["records"]) * 3,
            "note": "held-out test U/BF/WF runs only after all development artifacts are frozen",
        },
        {
            "stage": "heldout-tax-publication",
            "generated_surfaces_upper": tax_records * 3,
            "note": "U/BF/WF tax matrix runs only after all development artifacts are frozen",
        },
        {
            "stage": "conditioned-test-interventions-worst-case",
            "generated_surfaces_upper": int(matrices["test"]["records"]) * 5,
            "note": "held-out test upper bound assuming every U row fails",
        },
        {
            "stage": "conditioned-tax-interventions-worst-case",
            "generated_surfaces_upper": tax_records * 5,
            "note": "held-out tax upper bound assuming every U row fails",
        },
    ]
    observed_schedule = [str(stage["stage"]) for stage in stages]
    if method.get("execution_schedule") != observed_schedule:
        raise ValueError(
            "method execution schedule differs from frozen configuration"
        )
    generated_surfaces = sum(
        int(stage["generated_surfaces_upper"]) for stage in stages
    )
    theoretical_tokens = generated_surfaces * max_new_tokens
    checkpoint_bytes = int(model["safetensors_bytes"])
    passed = (
        generated_surfaces <= MAX_GENERATED_SURFACES
        and theoretical_tokens <= MAX_THEORETICAL_OUTPUT_TOKENS
        and checkpoint_bytes > 0
    )
    plan = {
        "schema_version": 1,
        "kind": "selected-lora-method-resource-plan",
        "selected_key": selected_key,
        "selected_model": model["repo_id"],
        "evidence": {
            "config_sha256": config_sha256,
            "selection_sha256": selection_sha256,
        },
        "runtime_limits": {
            "concurrent_model_processes": 1,
            "cpu_math_threads_per_model_process": 2,
            "max_new_tokens_per_surface": max_new_tokens,
            "completion_cache_entries_per_example": (
                MAX_COMPLETION_CACHE_ENTRIES_PER_EXAMPLE
            ),
            "private_scratch_required": True,
            "checkpoint_bytes": checkpoint_bytes,
            "minimum_free_scratch_bytes": checkpoint_bytes + SCRATCH_MARGIN_BYTES,
            "scratch_margin_bytes": SCRATCH_MARGIN_BYTES,
        },
        "stages": stages,
        "upper_bounds": {
            "generated_surfaces": generated_surfaces,
            "theoretical_output_tokens": theoretical_tokens,
            "allowed_generated_surfaces": MAX_GENERATED_SURFACES,
            "allowed_theoretical_output_tokens": MAX_THEORETICAL_OUTPUT_TOKENS,
        },
        "gate": {"passed": passed},
    }
    if not passed:
        raise ValueError(
            "selected method suite exceeds frozen resource ceiling: "
            f"surfaces={generated_surfaces}, tokens={theoretical_tokens}"
        )
    return plan


__all__ = [
    "MAX_GENERATED_SURFACES",
    "MAX_THEORETICAL_OUTPUT_TOKENS",
    "build_method_resource_plan",
]
