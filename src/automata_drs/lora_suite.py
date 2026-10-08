"""Collect recent-model LoRA artifacts into reviewer-facing comparison tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .baseline_comparison import analyze_lora_baseline_comparisons
from .host_selection import verify_frozen_lora_host_selection


def _read_json(
    path: Path, evidence: dict[str, str] | None = None
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = path.read_bytes()
    if evidence is not None:
        evidence[str(path)] = hashlib.sha256(payload).hexdigest()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _hash_file(path: Path, evidence: dict[str, str]) -> str:
    if not path.is_file():
        raise ValueError(f"missing release-audit input: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence[str(path)] = digest
    return digest


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _group(
    summary: dict[str, Any] | None,
    configuration: str,
    strategy: str = "greedy",
) -> dict[str, Any] | None:
    if summary is None:
        return None
    matches = [
        row
        for row in summary.get("groups", [])
        if row.get("configuration") == configuration
        and row.get("strategy") == strategy
    ]
    if len(matches) > 1:
        raise ValueError(f"duplicate {configuration}/greedy summary group")
    return matches[0] if matches else None


def _pair(
    summary: dict[str, Any] | None,
    configuration: str,
    strategy: str = "greedy",
) -> dict[str, Any] | None:
    if summary is None:
        return None
    matches = [
        row
        for row in summary.get("paired_vs_u", [])
        if row.get("configuration") == configuration
        and row.get("strategy") == strategy
    ]
    if len(matches) > 1:
        raise ValueError(f"duplicate {configuration}/greedy paired summary")
    return matches[0] if matches else None


def _statistics(
    report: dict[str, Any] | None,
    configuration: str,
    strategy: str = "greedy",
) -> dict[str, Any] | None:
    if report is None:
        return None
    matches = [
        row
        for row in report.get("comparisons", [])
        if row.get("configuration") == configuration
        and row.get("strategy") == strategy
    ]
    if len(matches) > 1:
        raise ValueError(
            f"duplicate {configuration}/greedy paired-statistics comparison"
        )
    return matches[0] if matches else None


def collect_lora_suite(
    config: dict[str, Any],
    runs_root: Path,
    *,
    config_sha256: str | None = None,
    config_bytes: bytes | None = None,
    repository_root: Path | None = None,
    scope: str = "full",
    require_complete_methods: bool | None = None,
) -> dict[str, Any]:
    if scope not in {"baseline", "full"}:
        raise ValueError("LoRA suite scope must be baseline or full")
    include_methods = scope == "full"
    strict_methods = (
        include_methods and repository_root is not None
        if require_complete_methods is None
        else require_complete_methods
    )
    if strict_methods and not include_methods:
        raise ValueError("complete method evidence requires full suite scope")
    input_sha256: dict[str, str] = {}
    selection_path = runs_root / "selected-host.json"
    selection = _read_json(selection_path, input_sha256)
    selected_key = (
        None if selection is None else str(selection.get("selected_key", ""))
    )
    if strict_methods and (
        selection is None
        or selection.get("status") != "selected"
        or not selected_key
    ):
        raise ValueError("complete method evidence requires a frozen selected host")
    models = []
    for model_row in sorted(config["models"], key=lambda row: row["priority"]):
        key = str(model_row["key"])
        run_root = runs_root / f"{key}-r16-seed42"
        train_manifest = _read_json(
            run_root / "train_manifest.json", input_sha256
        )
        train_metrics = _read_json(run_root / "train_metrics.json", input_sha256)
        smoke_gate = _read_json(
            runs_root / f"{key}-smoke" / "smoke_gate.json", input_sha256
        )
        baselines = []
        for split in ("dev", "test", "tax"):
            path = run_root / f"{split}-u-greedy.summary.json"
            summary = _read_json(path, input_sha256)
            baselines.append(
                {
                    "split": split,
                    "path": str(path),
                    "group": _group(summary, "U"),
                }
            )

        comparisons = []
        method_baselines = []
        evaluations = (
            ("dev400", "dev400-u-bf-wf-greedy", ("BF", "WF"), "greedy"),
            (
                "dev400-fine",
                "dev400-u-s-r-g-b-w-bf-wf-greedy",
                ("S", "R", "G", "B", "W"),
                "greedy",
            ),
            (
                "dev400-beam4",
                "dev400-u-postw-w-wf-beam4",
                ("PostW", "W", "WF"),
                "beam4",
            ),
            ("dev1132", "dev1132-u-bf-wf-greedy", ("BF", "WF"), "greedy"),
            ("test1132", "test1132-u-bf-wf-greedy", ("BF", "WF"), "greedy"),
            ("tax117", "tax-u-bf-wf-greedy", ("BF", "WF"), "greedy"),
        )
        if include_methods and key == selected_key:
            for evaluation, stem, configurations, strategy in evaluations:
                path = run_root / f"{stem}.summary.json"
                summary = _read_json(path, input_sha256)
                if summary is None:
                    if strict_methods:
                        raise ValueError(f"missing selected-host method summary: {path}")
                    continue
                method_baseline = _group(summary, "U", strategy)
                if strict_methods and method_baseline is None:
                    raise ValueError(
                        "incomplete selected-host evaluation baseline: "
                        f"{evaluation}/U/{strategy}"
                    )
                method_baselines.append(
                    {
                        "evaluation": evaluation,
                        "configuration": "U",
                        "strategy": strategy,
                        "path": str(path),
                        "group": method_baseline,
                    }
                )
                statistics_path = run_root / f"{stem}.paired-statistics.json"
                statistics = _read_json(statistics_path, input_sha256)
                if statistics is None and strict_methods:
                    raise ValueError(
                        f"missing selected-host paired statistics: {statistics_path}"
                    )
                for configuration in configurations:
                    group = _group(summary, configuration, strategy)
                    paired = _pair(summary, configuration, strategy)
                    comparison_statistics = _statistics(
                        statistics, configuration, strategy
                    )
                    if strict_methods and (
                        group is None
                        or paired is None
                        or comparison_statistics is None
                    ):
                        raise ValueError(
                            "incomplete selected-host comparison cell: "
                            f"{evaluation}/{configuration}/{strategy}"
                        )
                    comparisons.append(
                        {
                            "evaluation": evaluation,
                            "configuration": configuration,
                            "strategy": strategy,
                            "path": str(path),
                            "group": group,
                            "paired": paired,
                            "statistics_path": str(statistics_path),
                            "statistics": comparison_statistics,
                        }
                    )

        interventions = []
        if include_methods and key == selected_key:
            for split in ("dev", "test", "tax"):
                stem = f"{split}-u-invalid-u-b-w-bf-wf-greedy-v2"
                summary_path = run_root / f"{stem}.summary.json"
                statistics_path = run_root / f"{stem}.paired-statistics.json"
                summary = _read_json(summary_path, input_sha256)
                statistics = _read_json(statistics_path, input_sha256)
                if summary is None and statistics is None:
                    continue
                if summary is None or statistics is None:
                    raise ValueError(
                        f"partial failure-conditioned evidence for split: {split}"
                    )
                for configuration in ("B", "W", "BF", "WF"):
                    group = _group(summary, configuration)
                    paired = _pair(summary, configuration)
                    comparison_statistics = _statistics(
                        statistics, configuration
                    )
                    if strict_methods and (
                        group is None
                        or paired is None
                        or comparison_statistics is None
                    ):
                        raise ValueError(
                            "incomplete conditioned comparison cell: "
                            f"{split}/{configuration}"
                        )
                    interventions.append(
                        {
                            "split": split,
                            "configuration": configuration,
                            "population_scope": (
                                "failure-conditioned-on-u-invalidity"
                            ),
                            "population_validity_estimate_allowed": False,
                            "summary_path": str(summary_path),
                            "statistics_path": str(statistics_path),
                            "group": group,
                            "paired": paired,
                            "statistics": comparison_statistics,
                        }
                    )

        model_result = {
                "key": key,
                "model": model_row["repo_id"],
                "revision": model_row["revision"],
                "initialization": model_row["initialization"],
                "parameters_billions": model_row["parameters_billions"],
                "checkpoint_parameters_billions": model_row.get(
                    "checkpoint_parameters_billions",
                    model_row["parameters_billions"],
                ),
                "baseline_generation_contract": model_row.get(
                    "baseline_generation_contract"
                ),
                "baseline_generation_split_contracts": model_row.get(
                    "baseline_generation_split_contracts",
                    {
                        split: model_row.get("baseline_generation_contract")
                        for split in ("dev", "test", "tax")
                    },
                ),
                "run_root": str(run_root),
                "smoke_gate": smoke_gate,
                "training": {
                    "status": (
                        None if train_manifest is None else train_manifest.get("status")
                    ),
                    "train_loss": (
                        None
                        if train_metrics is None
                        else train_metrics.get("train", {}).get("train_loss")
                    ),
                    "eval_loss": (
                        None
                        if train_metrics is None
                        else train_metrics.get("eval", {}).get("eval_loss")
                    ),
                    "wall_seconds": (
                        None if train_metrics is None else train_metrics.get("wall_seconds")
                    ),
                    "best_model_checkpoint": (
                        None
                        if train_metrics is None
                        else train_metrics.get("best_model_checkpoint")
                    ),
                },
                "baselines": baselines,
                "comparisons": comparisons,
            }
        if include_methods:
            model_result["method_baselines"] = method_baselines
            model_result["interventions"] = interventions
        models.append(model_result)
    baseline_verification = None
    baseline_comparisons = None
    if repository_root is not None:
        if config_bytes is None or selection is None:
            raise ValueError(
                "strict suite collection requires config bytes and frozen selection"
            )
        verify_frozen_lora_host_selection(
            config,
            config_bytes,
            selection,
            runs_root,
            repository_root=repository_root,
        )
        paired = config["method_experiment"]["paired_statistics"]
        baseline_comparisons = analyze_lora_baseline_comparisons(
            config,
            runs_root,
            repository_root,
            bootstrap_resamples=int(paired["bootstrap_resamples"]),
            confidence=float(paired["confidence"]),
            seed=int(paired["seed"]),
        )
        baseline_verification = baseline_comparisons.pop(
            "baseline_verification"
        )
    resource_plan_path = runs_root / "selected-method-resource-plan.json"
    resource_plan = (
        _read_json(resource_plan_path, input_sha256)
        if include_methods
        else None
    )
    if strict_methods and resource_plan is None:
        raise ValueError("complete method evidence requires the resource plan")
    if resource_plan is not None:
        if selection is None:
            raise ValueError("method resource plan exists without selected host")
        evidence = resource_plan.get("evidence", {})
        if (
            resource_plan.get("kind") != "selected-lora-method-resource-plan"
            or resource_plan.get("gate", {}).get("passed") is not True
            or resource_plan.get("selected_key") != selection.get("selected_key")
            or evidence.get("config_sha256") != config_sha256
            or evidence.get("selection_sha256")
            != input_sha256.get(str(selection_path))
        ):
            raise ValueError(
                "selected-host method resource plan is not frozen/complete"
            )
    method_gates = None
    if include_methods and selection is not None and selected_key:
        method = config.get("method_experiment", {})
        pilot_records = int(method.get("pilot", {}).get("records", 0))
        selected_run_root = runs_root / f"{selected_key}-r16-seed42"
        gate_specs = [
            (
                "pilot-dev",
                selected_run_root
                / f"dev{pilot_records}-method-pilot-gate.json",
            ),
            (
                "layer-supplement",
                selected_run_root
                / f"dev{pilot_records}-layer-ablation-gate.json",
            ),
            (
                "fine-supplement",
                selected_run_root
                / f"dev{pilot_records}-fine-grained-ablation-gate.json",
            ),
            (
                "beam-candidates-and-online",
                selected_run_root
                / f"dev{pilot_records}-beam-selection-gate.json",
            ),
        ]
        matrices = {
            str(row.get("split")): row
            for row in method.get("publication_matrices", [])
            if isinstance(row, dict)
        }
        for split in ("dev", "test", "tax"):
            matrix = matrices.get(split, {})
            records = int(matrix.get("records", 0))
            gate_name = (
                "tax-publication-matrix-gate.json"
                if split == "tax"
                else f"{split}{records}-publication-matrix-gate.json"
            )
            gate_specs.append(
                (f"publication-{split}", selected_run_root / gate_name)
            )
        for split in ("dev", "test", "tax"):
            gate_specs.append(
                (
                    f"conditioned-{split}",
                    selected_run_root
                    / f"{split}-u-invalid-intervention-gate-v2.json",
                )
            )
        method_gates = []
        gate_by_stage: dict[str, dict[str, Any]] = {}
        for stage, path in gate_specs:
            gate = _read_json(path, input_sha256)
            if gate is None:
                if strict_methods:
                    raise ValueError(f"missing selected-host method gate: {path}")
                continue
            if gate.get("passed") is not True:
                raise ValueError(f"selected-host method gate did not pass: {path}")
            gate_by_stage[stage] = gate
            method_gates.append(
                {"stage": stage, "path": str(path), "passed": True}
            )
        if strict_methods:
            for split in ("dev", "test", "tax"):
                gate = gate_by_stage[f"conditioned-{split}"]
                selected_records = int(
                    gate.get("diagnostics", {}).get("selected_records", -1)
                )
                if selected_records < 0:
                    raise ValueError(
                        f"conditioned gate has no selected denominator: {split}"
                    )
                observed_cells = [
                    row
                    for model in models
                    if model.get("key") == selected_key
                    for row in model.get("interventions", [])
                    if row.get("split") == split
                ]
                expected_cells = 0 if selected_records == 0 else 4
                if len(observed_cells) != expected_cells:
                    raise ValueError(
                        "conditioned report cells differ from frozen denominator: "
                        f"{split} observed={len(observed_cells)} "
                        f"expected={expected_cells}"
                    )
    certified_releases = None
    if include_methods and selection is not None and selected_key:
        method = config.get("method_experiment", {})
        matrices = {
            str(row.get("split")): row
            for row in method.get("publication_matrices", [])
            if isinstance(row, dict)
        }
        selected_run_root = runs_root / f"{selected_key}-r16-seed42"
        certified_releases = []
        for split in ("dev", "test", "tax"):
            records = int(matrices.get(split, {}).get("records", 0))
            if split == "tax":
                stem = "tax-u-bf-wf-greedy"
                audit_name = "tax-certified-release-audit.json"
            else:
                stem = f"{split}{records}-u-bf-wf-greedy"
                audit_name = f"{split}{records}-certified-release-audit.json"
            audit_path = selected_run_root / audit_name
            audit = _read_json(audit_path, input_sha256)
            if audit is None:
                if strict_methods:
                    raise ValueError(
                        f"missing selected-host certified release audit: {audit_path}"
                    )
                continue
            predictions_path = selected_run_root / f"{stem}.jsonl"
            manifest_path = Path(f"{predictions_path}.manifest.json")
            predictions_sha256 = _hash_file(predictions_path, input_sha256)
            manifest_sha256 = _hash_file(manifest_path, input_sha256)
            audit_input = audit.get("input", {})
            population = audit.get("population", {})
            release = audit.get("release", {})
            implementation = audit.get("implementation", {})
            guarantee_counts = release.get("guarantee_counts", {})
            expected_rows = records * 2
            valid = (
                records > 0
                and audit.get("schema_version") == 2
                and audit.get("audit") == "sbn-certified-release"
                and audit.get("claim_boundary")
                == (
                    "policy-relative structural/well-formedness release only; "
                    "not task-semantic correctness"
                )
                and audit.get("gate", {}).get("passed") is True
                and audit.get("gate", {}).get("performance_threshold_used") is False
                and audit_input.get("configurations") == ["BF", "WF"]
                and float(audit_input.get("completion_budget_factor", -1.0)) == 1.0
                and audit_input.get("predictions_sha256") == predictions_sha256
                and audit_input.get("manifest_sha256") == manifest_sha256
                and population.get("examples") == records
                and population.get("rows") == expected_rows
                and population.get("unique_identities") == expected_rows
                and release.get("releasable_rows") == expected_rows
                and release.get("refused_rows") == 0
                and release.get("failures") == []
                and release.get("reason_counts") == {}
                and guarantee_counts
                == {
                    "finite-horizon-structural-and-referee-policy-membership": records,
                    "finite-horizon-structural-policy-membership": records,
                }
                and _is_sha256(implementation.get("audit_sha256"))
                and _is_sha256(implementation.get("release_sha256"))
            )
            if not valid:
                raise ValueError(
                    f"selected-host certified release audit is not valid: {audit_path}"
                )
            certified_releases.append(
                {"split": split, "path": str(audit_path), "audit": audit}
            )
        if strict_methods and len(certified_releases) != 3:
            raise ValueError("complete method evidence requires three release audits")

    cross_representation = None
    if include_methods and selection is not None:
        selected_key = str(selection.get("selected_key", ""))
        selected_rows = [
            row for row in config["models"] if row["key"] == selected_key
        ]
        if len(selected_rows) != 1:
            raise ValueError("selected host is missing or duplicated in config")
        audit_path = (
            runs_root.parent
            / "penman"
            / f"{selected_key}-grapes-token-lifting.json"
        )
        audit = _read_json(audit_path, input_sha256)
        if audit is None and strict_methods:
            raise ValueError(
                "complete method evidence requires the PENMAN token audit"
            )
        if audit is not None:
            selected_row = selected_rows[0]
            coverage = audit.get("coverage", {})
            source = audit.get("source", {})
            tokenizer = audit.get("tokenizer", {})
            implementation = audit.get("implementation", {})
            if (
                audit.get("audit") != "penman-bytelevel-token-lifting"
                or audit.get("gate", {}).get("passed") is not True
                or source.get("records") != 1471
                or source.get("sha256")
                != "d3f2be3224570ad433c951a2950d2271a94db4c1f4ce3f9749c9a4ae18d8b103"
                or coverage.get("denominator") != 1471
                or coverage.get("exact_byte_roundtrip") != 1471
                or coverage.get("accepted_with_eos") != 1471
                or tokenizer.get("model") != selected_row["repo_id"]
                or tokenizer.get("requested_revision")
                != selected_row["revision"]
                or implementation.get("config_sha256") != config_sha256
            ):
                raise ValueError(
                    "selected-host PENMAN token-lifting audit is not frozen/complete"
                )
            cross_representation = {
                "path": str(audit_path),
                "audit": audit,
            }
    suite = {
        "schema_version": 4 if scope == "baseline" else 7,
        "scope": scope,
        "evidence": {
            "config_sha256": config_sha256,
            "input_sha256": input_sha256,
        },
        "recipe": dict(config["shared_recipe"]),
        "host_selection": selection,
        "baseline_verification": baseline_verification,
        "baseline_comparisons": baseline_comparisons,
        "method_resource_plan": resource_plan,
        "certified_releases": certified_releases,
        "cross_representation": cross_representation,
        "models": models,
    }
    if include_methods:
        suite["method_gates"] = method_gates
    return suite


def verify_lora_suite_outputs(
    suite: dict[str, Any], json_path: Path, markdown_path: Path
) -> None:
    """Fail closed if stored reviewer-facing outputs differ from current inputs."""

    try:
        stored_json = json.loads(json_path.read_text(encoding="utf-8"))
        stored_markdown = markdown_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read stored LoRA suite output: {error}") from error
    if stored_json != suite:
        raise ValueError("stored LoRA suite JSON differs from current evidence")
    expected_markdown = lora_suite_to_markdown(suite)
    if stored_markdown != expected_markdown:
        raise ValueError("stored LoRA suite Markdown differs from current evidence")


def _number(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _count(group: dict[str, Any] | None, key: str) -> str:
    if group is None:
        return "—"
    return f"{group.get(key, 0)}/{group.get('total', 0)}"


def _raw_referee_count(group: dict[str, Any] | None) -> str:
    if group is None:
        return "—"
    value = group.get("raw_referee_valid", group.get("referee_valid", 0))
    return f"{value}/{group.get('total', 0)}"


def _interval(metric: dict[str, Any] | None, digits: int = 4) -> str:
    if metric is None:
        return "—"
    delta = metric.get("delta_mean")
    lower = metric.get("delta_ci_lower")
    upper = metric.get("delta_ci_upper")
    if delta is None or lower is None or upper is None:
        return "—"
    return (
        f"{float(delta):.{digits}f} "
        f"[{float(lower):.{digits}f}, {float(upper):.{digits}f}]"
    )


def _p_values(metric: dict[str, Any] | None) -> str:
    if metric is None:
        return "—"
    raw = metric.get("mcnemar_exact_p")
    adjusted = metric.get("mcnemar_exact_p_holm")
    if raw is None or adjusted is None:
        return "—"
    return f"{float(raw):.4f}/{float(adjusted):.4f}"


def _generation_contract_label(model: dict[str, Any]) -> str:
    contracts = model.get("baseline_generation_split_contracts", {})
    if not isinstance(contracts, dict) or not contracts:
        return str(model.get("baseline_generation_contract") or "—")
    values = [contracts.get(split) for split in ("dev", "test", "tax")]
    if len(set(values)) == 1:
        return str(values[0] or "—")
    return "; ".join(
        f"{split}={contracts.get(split) or '—'}"
        for split in ("dev", "test", "tax")
    )


def _parameter_label(model: dict[str, Any]) -> str:
    language = float(model["parameters_billions"])
    checkpoint = float(
        model.get("checkpoint_parameters_billions", language)
    )
    if checkpoint == language:
        return f"{language:.2f}"
    return f"{language:.2f}/{checkpoint:.2f}"


def _baseline_interval(metric: dict[str, Any], digits: int = 4) -> str:
    return (
        f"{float(metric['delta_right_minus_left']):.{digits}f} "
        f"[{float(metric['delta_ci_lower']):.{digits}f}, "
        f"{float(metric['delta_ci_upper']):.{digits}f}]"
    )


def lora_suite_to_markdown(suite: dict[str, Any]) -> str:
    selection = suite.get("host_selection")
    baseline_verification = suite.get("baseline_verification")
    baseline_comparisons = suite.get("baseline_comparisons")
    lines = [
        "# Recent sub-10B LoRA suite",
        "",
        (
            "Artifact scope: baseline-only; later method artifacts are excluded."
            if suite.get("scope") == "baseline"
            else "Artifact scope: full baseline-and-method evidence."
        ),
        "",
        (
            "Primary constrained-decoding host: —"
            if selection is None
            else f"Primary constrained-decoding host: `{selection['selected_key']}` "
            "(selected from development results only)."
        ),
    ]
    if baseline_verification is not None:
        lines.extend(
            [
                "",
                (
                    "All dev/test/tax baseline rows, score identities, source "
                    "bindings, and recomputed core summaries passed the frozen "
                    "verification gate."
                ),
            ]
        )
    if selection is not None and selection.get("ranking"):
        candidates = {
            row.get("key"): row
            for row in selection.get("candidates", [])
            if isinstance(row, dict)
        }
        ranking = list(selection["ranking"])
        if set(ranking) != set(candidates) or len(ranking) != len(candidates):
            raise ValueError("host-selection ranking/candidates are inconsistent")
        lines.extend(
            [
                "",
                "## Development-only host selection",
                "",
                (
                    "Ranked by development SBN Smatch, then development Referee "
                    "validity, raw-SBN validity, smaller parameter count, and "
                    "pre-registered priority. Test and tax metrics are excluded."
                ),
                "",
                "| Rank | Model | Dev Smatch | Dev raw W-valid | Dev raw SBN valid | LM params (B) | Selected |",
                "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for rank, key in enumerate(ranking, start=1):
            candidate = candidates[key]
            metrics = candidate.get("metrics", {})
            lines.append(
                f"| {rank} | `{key}` | "
                f"{_number(metrics.get('sbn_smatch_mean'))} | "
                f"{_number(metrics.get('referee_valid_rate_total'))} | "
                f"{_number(metrics.get('raw_sbn_valid_rate'))} | "
                f"{_number(candidate.get('parameters_billions'), 2)} | "
                f"{'yes' if key == selection.get('selected_key') else ''} |"
            )
    if baseline_comparisons is not None:
        lines.extend(
            [
                "",
                "## Paired baseline quality comparisons",
                "",
                (
                    "All deltas are right model minus left model on identical "
                    "examples. These intervals and tests are descriptive and "
                    "do not enter host selection."
                ),
                "",
                "| Split | Pair | N | Δ Smatch [95% CI] | Δ raw well-formed [95% CI] | Right fixed/regressed | McNemar/Holm |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for comparison in baseline_comparisons["comparisons"]:
            smatch = comparison["metrics"]["sbn_smatch"]
            wellformed = comparison["metrics"]["wellformed_valid"]
            lines.append(
                f"| {comparison['split']} | `{comparison['left']}` → "
                f"`{comparison['right']}` | {comparison['paired']} | "
                f"{_baseline_interval(smatch)} | "
                f"{_baseline_interval(wellformed)} | "
                f"{wellformed['right_fixed']}/{wellformed['right_regressed']} | "
                f"{float(wellformed['mcnemar_exact_p']):.4f}/"
                f"{float(wellformed['mcnemar_exact_p_holm']):.4f} |"
            )
        headroom = baseline_comparisons.get("constraint_headroom", [])
        if headroom:
            lines.extend(
                [
                    "",
                    "## Unconstrained baseline constraint headroom",
                    "",
                    (
                        "Raw model surfaces are replayed offline through the "
                        "monotonic S/R/G/B/W policies. This diagnostic does not "
                        "enter host selection."
                    ),
                    "",
                    "| Model | Split | N | S valid | R valid | G valid | B valid | W valid | First block S/R/G/B/W | Raw ≠ postprocessed | Truncated ∩ W-invalid |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for report in headroom:
                acceptance = report["acceptance"]
                blocking = report["first_blocking_layer"]
                truncation = report["truncation"]
                lines.append(
                    f"| `{report['model']}` | {report['split']} | "
                    f"{report['records']} | "
                    + " | ".join(
                        f"{acceptance[configuration]['accepted']}/{report['records']}"
                        for configuration in ("S", "R", "G", "B", "W")
                    )
                    + " | "
                    + "/".join(
                        str(blocking[configuration])
                        for configuration in ("S", "R", "G", "B", "W")
                    )
                    + " | "
                    + str(
                        report.get("surface", {}).get(
                            "raw_postprocessed_different", 0
                        )
                    )
                    + f" | {truncation['and_w_invalid']} |"
                )
        paired_headroom = [
            comparison
            for comparison in baseline_comparisons["comparisons"]
            if isinstance(comparison.get("constraint_headroom"), dict)
        ]
        if paired_headroom:
            lines.extend(
                [
                    "",
                    "### Paired W-validity overlap",
                    "",
                    "| Split | Pair | Both valid | Left only valid | Right only valid | Both invalid |",
                    "| --- | --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for comparison in paired_headroom:
                overlap = comparison["constraint_headroom"][
                    "w_validity_overlap"
                ]
                lines.append(
                    f"| {comparison['split']} | `{comparison['left']}` → "
                    f"`{comparison['right']}` | {overlap['both_valid']} | "
                    f"{overlap['left_only_valid']} | "
                    f"{overlap['right_only_valid']} | "
                    f"{overlap['both_invalid']} |"
                )
    lines.extend(
        [
            "",
            "## Training",
            "",
            "| Model | Init | Baseline generation | LM/checkpoint params (B) | Status | Train loss | Dev loss | Wall min |",
            "| --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for model in suite["models"]:
        training = model["training"]
        wall = training.get("wall_seconds")
        lines.append(
            f"| `{model['key']}` | {model['initialization']} | "
            f"{_generation_contract_label(model)} | "
            f"{_parameter_label(model)} | "
            f"{training.get('status') or '—'} | "
            f"{_number(training.get('train_loss'))} | "
            f"{_number(training.get('eval_loss'))} | "
            f"{_number(None if wall is None else float(wall) / 60, 2)} |"
        )

    lines.extend(
        [
            "",
            "## Unconstrained baselines",
            "",
            "| Model | Split | N | Raw SBN | Raw W-valid | Smatch | Exact | Canonical tok | Truncated |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in suite["models"]:
        for baseline in model["baselines"]:
            group = baseline["group"]
            lines.append(
                f"| `{model['key']}` | {baseline['split']} | "
                f"{('—' if group is None else group.get('total'))} | "
                f"{_count(group, 'raw_sbn_valid')} | "
                f"{_raw_referee_count(group)} | "
                f"{_number(None if group is None else group.get('sbn_smatch_mean'))} | "
                f"{_count(group, 'exact_serialization')} | "
                f"{_count(group, 'canonical_tokenization')} | "
                f"{('—' if group is None else group.get('truncated'))} |"
            )

    lines.extend(
        [
            "",
            "## Paired selected-host method comparisons",
            "",
            "Intervals are 95% paired-percentile bootstrap intervals; the p-value column is exact McNemar/Holm-adjusted.",
            "",
            "| Model | Set | Config | Decode | N | Raw SBN | Raw W-valid | Smatch | PostW fallback / mean valid candidates | W-fixed/regressed | Δ raw well-formed [CI] | McNemar/Holm | Δ Smatch [CI] | Δ mean log-p [CI] | Δ gen ms [CI] |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in suite["models"]:
        for baseline in model.get("method_baselines", []):
            group = baseline["group"]
            lines.append(
                f"| `{model['key']}` | {baseline['evaluation']} | U | "
                f"{baseline.get('strategy', 'greedy')} | "
                f"{('—' if group is None else group.get('total'))} | "
                f"{_count(group, 'raw_sbn_valid')} | "
                f"{_raw_referee_count(group)} | "
                f"{_number(None if group is None else group.get('sbn_smatch_mean'))} | "
                "— | — | baseline | baseline | baseline | baseline | baseline |"
            )
        for comparison in model["comparisons"]:
            group = comparison["group"]
            pair = comparison["paired"]
            statistics = comparison.get("statistics")
            metrics = (
                {} if statistics is None else statistics.get("metrics", {})
            )
            fixed = (
                "—"
                if pair is None
                else f"{pair.get('wellformed_fixed', pair.get('raw_sbn_fixed', 0))}/"
                f"{pair.get('wellformed_regressed', pair.get('raw_sbn_regressed', 0))}"
            )
            postw = (
                "—"
                if group is None
                or comparison["configuration"] != "PostW"
                else f"{group.get('postw_fallbacks', 0)}/{group.get('total', 0)} / "
                f"{_number(group.get('postw_w_valid_candidates_mean'), 2)}"
            )
            lines.append(
                f"| `{model['key']}` | {comparison['evaluation']} | "
                f"{comparison['configuration']} | "
                f"{comparison.get('strategy', 'greedy')} | "
                f"{('—' if group is None else group.get('total'))} | "
                f"{_count(group, 'raw_sbn_valid')} | "
                f"{_raw_referee_count(group)} | "
                f"{_number(None if group is None else group.get('sbn_smatch_mean'))} | "
                f"{postw} | "
                f"{fixed} | "
                f"{_interval(metrics.get('wellformed_valid'))} | "
                f"{_p_values(metrics.get('wellformed_valid'))} | "
                f"{_interval(metrics.get('sbn_smatch'))} | "
                f"{_interval(metrics.get('raw_model_mean_logprob'))} | "
                f"{_interval(metrics.get('generation_ms'), 2)} |"
            )
    conditioned_rows = [
        (model, row)
        for model in suite["models"]
        for row in model.get("interventions", [])
    ]
    if conditioned_rows:
        lines.extend(
            [
                "",
                "## Failure-conditioned mechanism diagnostics",
                "",
                (
                    "Membership is selected by an invalid unconstrained U "
                    "surface. These paired rows measure repair/regression "
                    "mechanisms only and must not be reported as population "
                    "validity or quality estimates."
                ),
                "",
                "| Model | Split | Selected N | Config | Raw SBN | Raw W-valid | Smatch | W-fixed/regressed | Δ Smatch [CI] | Δ mean log-p [CI] | Δ gen ms [CI] |",
                "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for model, row in conditioned_rows:
            group = row.get("group")
            paired = row.get("paired")
            statistics = row.get("statistics")
            metrics = (
                {} if statistics is None else statistics.get("metrics", {})
            )
            fixed = (
                "—"
                if paired is None
                else f"{paired.get('wellformed_fixed', paired.get('raw_sbn_fixed', 0))}/"
                f"{paired.get('wellformed_regressed', paired.get('raw_sbn_regressed', 0))}"
            )
            lines.append(
                f"| `{model['key']}` | {row['split']} | "
                f"{('—' if group is None else group.get('total'))} | "
                f"{row['configuration']} | "
                f"{_count(group, 'raw_sbn_valid')} | "
                f"{_raw_referee_count(group)} | "
                f"{_number(None if group is None else group.get('sbn_smatch_mean'))} | "
                f"{fixed} | "
                f"{_interval(metrics.get('sbn_smatch'))} | "
                f"{_interval(metrics.get('raw_model_mean_logprob'))} | "
                f"{_interval(metrics.get('generation_ms'), 2)} |"
            )
    method_gates = suite.get("method_gates")
    if method_gates:
        lines.extend(
            [
                "",
                "## Immutable method gates",
                "",
                (
                    f"All {len(method_gates)} registered pilot, publication, "
                    "ablation, beam, and conditioned-intervention gates passed "
                    "before this full report was collected."
                ),
                "",
                "| Stage | Passed | Gate |",
                "| --- | --- | --- |",
            ]
        )
        for gate in method_gates:
            lines.append(
                f"| {gate['stage']} | {'yes' if gate.get('passed') else 'no'} | "
                f"`{gate['path']}` |"
            )
    certified_releases = suite.get("certified_releases")
    if certified_releases:
        lines.extend(
            [
                "",
                "## Certified structural release",
                "",
                (
                    "Every BF/WF publication row below passed a hash-bound, "
                    "fail-closed recomputation of policy acceptance, EOS, horizon, "
                    "all-masked, and finite-horizon certificate preconditions. "
                    "Certified policy validity is conditional on release; release "
                    "coverage is reported over all attempted BF/WF rows. This strict "
                    "suite requires both rates to be 100%. This is not "
                    "task-semantic correctness."
                ),
                "",
                "| Split | Examples | Attempts | Released | Release coverage | Certified policy validity among released | Structural | Structural + Referee | Refused |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in certified_releases:
            audit = row["audit"]
            population = audit["population"]
            release = audit["release"]
            guarantees = release["guarantee_counts"]
            attempts = int(population["rows"])
            released = int(release["releasable_rows"])
            release_coverage = 100.0 * released / attempts
            certified_validity = 100.0 if released else float("nan")
            lines.append(
                f"| {row['split']} | {population['examples']} | "
                f"{attempts} | {released} | {release_coverage:.2f}% | "
                f"{certified_validity:.2f}% | "
                f"{guarantees['finite-horizon-structural-policy-membership']} | "
                f"{guarantees['finite-horizon-structural-and-referee-policy-membership']} | "
                f"{release['refused_rows']} |"
            )
    cross = suite.get("cross_representation")
    resource_plan = suite.get("method_resource_plan")
    if resource_plan is not None:
        bounds = resource_plan["upper_bounds"]
        runtime = resource_plan["runtime_limits"]
        lines.extend(
            [
                "",
                "## Frozen resource ceiling",
                "",
                (
                    f"The selected-host suite is capped at "
                    f"{bounds['generated_surfaces']:,} returned surfaces and "
                    f"{bounds['theoretical_output_tokens']:,} theoretical output-token "
                    f"slots, with {runtime['concurrent_model_processes']} model process, "
                    f"{runtime['cpu_math_threads_per_model_process']} CPU math threads, "
                    "and private scratch."
                ),
            ]
        )
    if cross is not None:
        audit = cross["audit"]
        coverage = audit["coverage"]
        lengths = audit["token_lengths"]
        lines.extend(
            [
                "",
                "## Cross-representation token lifting",
                "",
                "| Host | Corpus | Exact token bytes | Accepted with EOS | Token length p50/p95/max |",
                "| --- | --- | ---: | ---: | ---: |",
                (
                    f"| `{audit['tokenizer']['model']}` | GrAPES public 1,471 | "
                    f"{coverage['exact_byte_roundtrip']}/{coverage['denominator']} | "
                    f"{coverage['accepted_with_eos']}/{coverage['denominator']} | "
                    f"{lengths['median']}/{lengths['p95']}/{lengths['maximum']} |"
                ),
            ]
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "collect_lora_suite",
    "lora_suite_to_markdown",
    "verify_lora_suite_outputs",
]
