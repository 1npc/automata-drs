#!/usr/bin/env python3
"""Build the only paper-facing numeric package from canonical DRS/AMR suites.

The package is deliberately descriptive.  It reports every frozen outcome and
keeps certified validity conditional on release; it does not use a performance
threshold or automatically promote the paper to an algorithm framing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DRS_EVALUATION_CONFIGURATIONS = {
    "dev400": ("BF", "WF"),
    "dev400-fine": ("S", "R", "G", "B", "W"),
    "dev400-beam4": ("PostW", "W", "WF"),
    "dev1132": ("BF", "WF"),
    "test1132": ("BF", "WF"),
    "tax117": ("BF", "WF"),
}
PUBLICATION_EVALUATIONS = {
    "dev": "dev1132",
    "test": "test1132",
    "tax": "tax117",
}
MECHANISM_SOURCES = {
    "U": ("dev400-fine", None),
    "S": ("dev400-fine", "S"),
    "R": ("dev400-fine", "R"),
    "G": ("dev400-fine", "G"),
    "B": ("dev400-fine", "B"),
    "W": ("dev400-fine", "W"),
    "BF": ("dev400", "BF"),
    "WF": ("dev400", "WF"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_result_lock() -> Any:
    path = REPOSITORY_ROOT / "scripts" / "check_manuscript_result_lock.py"
    spec = importlib.util.spec_from_file_location("paper_result_lock", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load manuscript result lock: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_lock_builder(drs: Path, amr: Path) -> dict[str, Any]:
    return _load_result_lock().build_report(drs, amr)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"missing or non-numeric paper result: {label}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite paper result: {label}")
    return result


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"missing or non-integer paper result: {label}")
    if value < (1 if positive else 0):
        raise ValueError(f"out-of-range paper result: {label}")
    return value


def _unique(rows: Iterable[dict[str, Any]], keys: tuple[str, ...], label: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} contains a non-object row")
        key = tuple(row.get(name) for name in keys)
        if key in result:
            raise ValueError(f"duplicate {label} row: {key}")
        result[key] = row
    return result


def _group(group: Any, label: str) -> dict[str, Any]:
    if not isinstance(group, dict) or not group:
        raise ValueError(f"missing paper-result group: {label}")
    total = _integer(group.get("total"), f"{label}.total", positive=True)
    raw_sbn = _integer(group.get("raw_sbn_valid"), f"{label}.raw_sbn_valid")
    raw_w = _integer(
        group.get("raw_referee_valid", group.get("referee_valid")),
        f"{label}.raw_w_valid",
    )
    if raw_sbn > total or raw_w > total:
        raise ValueError(f"validity count exceeds population: {label}")
    return {
        "attempts": total,
        "raw_sbn_valid": raw_sbn,
        "raw_sbn_valid_rate": raw_sbn / total,
        "raw_w_valid": raw_w,
        "raw_w_valid_rate": raw_w / total,
        "smatch": _finite(group.get("sbn_smatch_mean"), f"{label}.smatch"),
        "mean_raw_model_logprob": _finite(
            group.get("raw_model_mean_logprob_mean"), f"{label}.mean_logprob"
        ),
        "generation_ms": _finite(
            group.get("generation_ms_mean"), f"{label}.generation_ms"
        ),
        "all_masked_rows": _integer(
            group.get("all_masked_rows", 0), f"{label}.all_masked_rows"
        ),
        "postw_fallbacks": _integer(
            group.get("postw_fallbacks", 0), f"{label}.postw_fallbacks"
        ),
    }


def _metric(statistics: Any, name: str, label: str) -> dict[str, float]:
    if not isinstance(statistics, dict):
        raise ValueError(f"missing paper-result statistics: {label}")
    metric = statistics.get("metrics", {}).get(name)
    if not isinstance(metric, dict):
        raise ValueError(f"missing paper-result metric: {label}.{name}")
    return {
        "delta": _finite(metric.get("delta_mean"), f"{label}.{name}.delta"),
        "ci_lower": _finite(
            metric.get("delta_ci_lower"), f"{label}.{name}.ci_lower"
        ),
        "ci_upper": _finite(
            metric.get("delta_ci_upper"), f"{label}.{name}.ci_upper"
        ),
    }


def _pair(pair: Any, label: str) -> dict[str, int]:
    if not isinstance(pair, dict):
        raise ValueError(f"missing paper-result pair: {label}")
    return {
        "changed": _integer(pair.get("prediction_changed"), f"{label}.changed"),
        "repairs": _integer(
            pair.get("wellformed_fixed", pair.get("raw_sbn_fixed")),
            f"{label}.repairs",
        ),
        "regressions": _integer(
            pair.get("wellformed_regressed", pair.get("raw_sbn_regressed")),
            f"{label}.regressions",
        ),
    }


def _release(audit_row: dict[str, Any], configuration: str) -> dict[str, Any]:
    audit = audit_row.get("audit", {})
    population = audit.get("population", {})
    release = audit.get("release", {})
    attempts = _integer(population.get("examples"), "DRS release attempts", positive=True)
    guarantee = (
        "finite-horizon-structural-policy-membership"
        if configuration == "BF"
        else "finite-horizon-structural-and-referee-policy-membership"
    )
    released = _integer(
        release.get("guarantee_counts", {}).get(guarantee),
        f"DRS {configuration} release count",
    )
    if released > attempts:
        raise ValueError("DRS released count exceeds attempts")
    return {
        "released": released,
        "refused": attempts - released,
        "attempts": attempts,
        "coverage": released / attempts,
        "certified_policy_validity_among_released": (
            1.0 if released else None
        ),
    }


def _selected_drs(drs: dict[str, Any]) -> dict[str, Any]:
    key = drs.get("host_selection", {}).get("selected_key")
    selected = [
        row
        for row in drs.get("models", [])
        if isinstance(row, dict) and row.get("key") == key
    ]
    if len(selected) != 1:
        raise ValueError("DRS suite does not contain exactly one selected model")
    return selected[0]


def _drs_results(drs: dict[str, Any]) -> dict[str, Any]:
    model = _selected_drs(drs)
    baselines = _unique(
        model.get("method_baselines", []), ("evaluation",), "DRS method baseline"
    )
    comparisons = _unique(
        model.get("comparisons", []),
        ("evaluation", "configuration"),
        "DRS comparison",
    )
    releases = _unique(
        drs.get("certified_releases", []), ("split",), "DRS release"
    )
    if set(key[0] for key in baselines) != set(DRS_EVALUATION_CONFIGURATIONS):
        raise ValueError("DRS suite has incomplete evaluation-specific U baselines")

    publication: list[dict[str, Any]] = []
    for split, evaluation in PUBLICATION_EVALUATIONS.items():
        baseline = _group(
            baselines[(evaluation,)]["group"], f"DRS {evaluation}/U"
        )
        publication.append(
            {
                "split": split,
                "condition": "U",
                **baseline,
                "certified_release": None,
                "delta_smatch": None,
                "delta_mean_logprob": None,
                "delta_generation_ms": None,
            }
        )
        for configuration in ("BF", "WF"):
            row = comparisons[(evaluation, configuration)]
            publication.append(
                {
                    "split": split,
                    "condition": configuration,
                    **_group(row["group"], f"DRS {evaluation}/{configuration}"),
                    "certified_release": _release(
                        releases[(split,)], configuration
                    ),
                    "delta_smatch": _metric(
                        row["statistics"], "sbn_smatch", f"{evaluation}/{configuration}"
                    ),
                    "delta_mean_logprob": _metric(
                        row["statistics"],
                        "raw_model_mean_logprob",
                        f"{evaluation}/{configuration}",
                    ),
                    "delta_generation_ms": _metric(
                        row["statistics"],
                        "generation_ms",
                        f"{evaluation}/{configuration}",
                    ),
                }
            )

    mechanism: list[dict[str, Any]] = []
    for condition, (evaluation, comparison_condition) in MECHANISM_SOURCES.items():
        if comparison_condition is None:
            facts = _group(baselines[(evaluation,)]["group"], f"DRS {evaluation}/U")
            mechanism.append(
                {
                    "condition": condition,
                    **facts,
                    "paired": None,
                    "delta_smatch": None,
                }
            )
            continue
        row = comparisons[(evaluation, comparison_condition)]
        mechanism.append(
            {
                "condition": condition,
                **_group(row["group"], f"DRS {evaluation}/{condition}"),
                "paired": _pair(row["paired"], f"{evaluation}/{condition}"),
                "delta_smatch": _metric(
                    row["statistics"], "sbn_smatch", f"{evaluation}/{condition}"
                ),
            }
        )

    beam: list[dict[str, Any]] = []
    evaluation = "dev400-beam4"
    beam.append(
        {
            "condition": "U",
            **_group(baselines[(evaluation,)]["group"], "DRS beam/U"),
            "paired": None,
            "delta_smatch": None,
        }
    )
    for configuration in ("PostW", "W", "WF"):
        row = comparisons[(evaluation, configuration)]
        beam.append(
            {
                "condition": configuration,
                **_group(row["group"], f"DRS beam/{configuration}"),
                "paired": _pair(row["paired"], f"DRS beam/{configuration}"),
                "delta_smatch": _metric(
                    row["statistics"], "sbn_smatch", f"DRS beam/{configuration}"
                ),
            }
        )
    return {
        "selected_model": model["key"],
        "publication": publication,
        "mechanism_ladder": mechanism,
        "beam_diagnostic": beam,
    }


def _amr_results(amr: dict[str, Any]) -> dict[str, Any]:
    matrices: list[dict[str, Any]] = []
    for matrix in amr.get("matrices", []):
        split = str(matrix.get("split"))
        attempts = _integer(matrix.get("examples"), f"AMR {split}.attempts", positive=True)
        paired = matrix.get("paired", {})
        smatch = paired.get("smatch", {})
        validity = paired.get("validity", {})
        probability = paired.get("probability", {})
        search = paired.get("search", {})
        release = matrix.get("release", {})
        released = _integer(
            release.get("releasable_rows"), f"AMR {split}.released"
        )
        refused = _integer(release.get("refused_rows"), f"AMR {split}.refused")
        if released + refused != attempts:
            raise ValueError(f"AMR {split} release accounting is incomplete")
        u_valid = _integer(
            matrix.get("u", {}).get("raw_official_penman"), f"AMR {split}.u_valid"
        )
        p_valid = _integer(
            matrix.get("p", {}).get("raw_official_penman"), f"AMR {split}.p_valid"
        )
        matrices.append(
            {
                "split": split,
                "attempts": attempts,
                "u_penman_valid": u_valid,
                "p_penman_valid": p_valid,
                "u_penman_valid_rate": u_valid / attempts,
                "p_penman_valid_rate": p_valid / attempts,
                "p_release": {
                    "released": released,
                    "refused": refused,
                    "attempts": attempts,
                    "coverage": released / attempts,
                    "certified_policy_validity_among_released": (
                        1.0 if released else None
                    ),
                    "reason_counts": release.get("reason_counts", {}),
                },
                "u_smatch": _finite(smatch.get("u_corpus_f1"), f"AMR {split}.u_smatch"),
                "p_smatch": _finite(smatch.get("p_corpus_f1"), f"AMR {split}.p_smatch"),
                "delta_smatch": {
                    "delta": _finite(smatch.get("delta_p_minus_u"), f"AMR {split}.delta"),
                    "ci_lower": _finite(smatch.get("delta_ci_lower"), f"AMR {split}.lower"),
                    "ci_upper": _finite(smatch.get("delta_ci_upper"), f"AMR {split}.upper"),
                },
                "validity_repairs": _integer(validity.get("repairs"), f"AMR {split}.repairs"),
                "validity_regressions": _integer(validity.get("regressions"), f"AMR {split}.regressions"),
                "delta_mean_logprob": _finite(
                    probability.get("mean_raw_model_logprob_delta_p_minus_u"),
                    f"AMR {split}.logprob_delta",
                ),
                "delta_generation_ms": _finite(
                    search.get("p_generation_ms_mean"), f"AMR {split}.p_ms"
                )
                - _finite(search.get("u_generation_ms_mean"), f"AMR {split}.u_ms"),
                "p_all_masked_rows": _integer(
                    search.get("p_all_masked_rows"), f"AMR {split}.all_masked"
                ),
            }
        )
    if {row["split"] for row in matrices} != {"pilot", "dev", "test"}:
        raise ValueError("AMR paper package requires pilot/dev/test matrices")

    beam = amr.get("beam", {})
    beam_rows: list[dict[str, Any]] = []
    for name in ("postp_minus_u", "p_minus_u", "p_minus_postp"):
        comparison = beam.get("comparisons", {}).get(name, {})
        beam_rows.append(
            {
                "contrast": name,
                "delta_smatch": {
                    "delta": _finite(comparison.get("corpus_smatch_delta"), f"AMR beam {name}.delta"),
                    "ci_lower": _finite(comparison.get("delta_ci_lower"), f"AMR beam {name}.lower"),
                    "ci_upper": _finite(comparison.get("delta_ci_upper"), f"AMR beam {name}.upper"),
                },
                "validity_repairs": _integer(comparison.get("repairs"), f"AMR beam {name}.repairs"),
                "validity_regressions": _integer(comparison.get("regressions"), f"AMR beam {name}.regressions"),
                "mcnemar_p": _finite(comparison.get("exact_mcnemar_p"), f"AMR beam {name}.mcnemar"),
                "holm_p": _finite(comparison.get("holm_adjusted_mcnemar_p"), f"AMR beam {name}.holm"),
            }
        )
    return {
        "matrices": matrices,
        "beam": {
            "comparisons": beam_rows,
            "postp_fallbacks": _integer(
                beam.get("selection", {}).get("fallbacks"), "AMR beam fallbacks"
            ),
            "online_p_all_masked_rows": _integer(
                beam.get("search", {}).get("online_p_all_masked_rows"),
                "AMR beam all-masked",
            ),
            "p_release": beam.get("release"),
        },
    }


def build_package(
    drs_path: Path,
    amr_path: Path,
    *,
    lock_builder: Callable[[Path, Path], dict[str, Any]] = _default_lock_builder,
) -> dict[str, Any]:
    drs = _read_object(drs_path)
    amr = _read_object(amr_path)
    lock = lock_builder(drs_path, amr_path)
    if (
        lock.get("drs", {}).get("ready") is not True
        or lock.get("amr", {}).get("ready") is not True
    ):
        raise ValueError("both canonical suites must pass the result lock")
    drs_results = _drs_results(drs)
    amr_results = _amr_results(amr)
    return {
        "schema_version": 1,
        "audit": "manuscript-result-package",
        "sources": {
            "drs": {"path": drs_path.name, "sha256": _sha256(drs_path)},
            "amr": {"path": amr_path.name, "sha256": _sha256(amr_path)},
        },
        "claim_boundary": {
            "certified_policy_validity_is_conditional_on_release": True,
            "release_coverage_is_reported_over_attempts": True,
            "structural_certification_implies_task_semantic_correctness": False,
            "performance_threshold_used": False,
            "automatic_paper_framing_used": False,
        },
        "drs": drs_results,
        "amr": amr_results,
        "framing": {
            "decision": "manual-under-frozen-submission-rule",
            "source": "docs/paper-review-audit.md#submission-decision-gate",
            "allowed_stories": [
                "algorithm-paper candidate",
                "reliability/diagnostic study",
                "negative mechanism study",
                "scope-limited diagnostic",
            ],
            "note": (
                "Every frozen outcome remains reportable; this package does not "
                "turn statistical significance or validity alone into a framing decision."
            ),
        },
        "claim_evidence": {
            "released_drs_rows_satisfy_registered_policy": "supported-by-release-audit",
            "drs_population_improvement": "outcome-dependent-do-not-assume",
            "released_amr_rows_satisfy_registered_policy": "supported-among-released",
            "amr_release_availability": "empirical-coverage-not-guaranteed",
            "amr_task_improvement": "outcome-dependent-do-not-assume",
        },
        "gate": {"passed": True, "performance_threshold_used": False},
    }


def _number(value: Any, digits: int = 4, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _count_rate(count: int, total: int) -> str:
    return f"{count}/{total} ({100.0 * count / total:.2f}%)"


def _interval(value: dict[str, Any] | None, digits: int = 4) -> str:
    if value is None:
        return "baseline"
    return (
        f"{_number(value['delta'], digits, signed=True)} "
        f"[{_number(value['ci_lower'], digits, signed=True)}, "
        f"{_number(value['ci_upper'], digits, signed=True)}]"
    )


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Canonical manuscript result package",
        "",
        (
            "All numbers below come from the two result-locked canonical suites. "
            "Certified validity is conditional on release; coverage is over attempts."
        ),
        "",
        "## DRS publication matrices",
        "",
        "| Split | Condition | N | Raw W-valid | Certified release | Certified validity among released | Smatch | Δ Smatch [95% CI] | Δ mean log-p [95% CI] | Δ generation ms [95% CI] |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["drs"]["publication"]:
        release = row["certified_release"]
        lines.append(
            f"| {row['split']} | {row['condition']} | {row['attempts']} | "
            f"{_count_rate(row['raw_w_valid'], row['attempts'])} | "
            + (
                "n/a | n/a | "
                if release is None
                else f"{_count_rate(release['released'], release['attempts'])} | "
                f"{_number(100.0 * release['certified_policy_validity_among_released'], 2)}% | "
            )
            + f"{_number(row['smatch'])} | {_interval(row['delta_smatch'])} | "
            f"{_interval(row['delta_mean_logprob'])} | "
            f"{_interval(row['delta_generation_ms'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## DRS first-400 mechanism ladder",
            "",
            "| Condition | Raw SBN valid | Raw W-valid | Smatch | Changed | Repairs/regressions | Mean raw log-p | Generation ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["drs"]["mechanism_ladder"]:
        pair = row["paired"]
        lines.append(
            f"| {row['condition']} | {_count_rate(row['raw_sbn_valid'], row['attempts'])} | "
            f"{_count_rate(row['raw_w_valid'], row['attempts'])} | "
            f"{_number(row['smatch'])} | "
            f"{('baseline' if pair is None else pair['changed'])} | "
            f"{('baseline' if pair is None else str(pair['repairs']) + '/' + str(pair['regressions']))} | "
            f"{_number(row['mean_raw_model_logprob'])} | "
            f"{_number(row['generation_ms'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## DRS development beam diagnostic",
            "",
            "| Condition | Raw W-valid | Smatch | Δ Smatch [95% CI] | PostW fallbacks | All-masked rows |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["drs"]["beam_diagnostic"]:
        lines.append(
            f"| {row['condition']} | {_count_rate(row['raw_w_valid'], row['attempts'])} | "
            f"{_number(row['smatch'])} | {_interval(row['delta_smatch'])} | "
            f"{row['postw_fallbacks']} | {row['all_masked_rows']} |"
        )
    lines.extend(
        [
            "",
            "## AMR U/P transfer",
            "",
            "| Split | N | U PENMAN | P PENMAN | P release coverage | Certified validity among released | U Smatch | P Smatch | Δ P−U [95% CI] | Δ mean log-p | Δ generation ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["amr"]["matrices"]:
        release = row["p_release"]
        certified = release["certified_policy_validity_among_released"]
        lines.append(
            f"| {row['split']} | {row['attempts']} | "
            f"{_count_rate(row['u_penman_valid'], row['attempts'])} | "
            f"{_count_rate(row['p_penman_valid'], row['attempts'])} | "
            f"{_count_rate(release['released'], release['attempts'])} | "
            f"{('n/a' if certified is None else _number(100.0 * certified, 2) + '%')} | "
            f"{_number(row['u_smatch'])} | {_number(row['p_smatch'])} | "
            f"{_interval(row['delta_smatch'])} | "
            f"{_number(row['delta_mean_logprob'], signed=True)} | "
            f"{_number(row['delta_generation_ms'], 2, signed=True)} |"
        )
    lines.extend(
        [
            "",
            "## AMR development beam diagnostic",
            "",
            "| Contrast | Δ Smatch [95% CI] | Repairs/regressions | McNemar/Holm |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in report["amr"]["beam"]["comparisons"]:
        lines.append(
            f"| {row['contrast']} | {_interval(row['delta_smatch'])} | "
            f"{row['validity_repairs']}/{row['validity_regressions']} | "
            f"{_number(row['mcnemar_p'])}/{_number(row['holm_p'])} |"
        )
    lines.extend(
        [
            "",
            (
                f"PostP fallbacks: {report['amr']['beam']['postp_fallbacks']}; "
                f"online-P all-masked rows: "
                f"{report['amr']['beam']['online_p_all_masked_rows']}."
            ),
            "",
            "## Claim and framing boundary",
            "",
            "- Structural certification is reported only among released rows.",
            "- Release coverage is reported separately over all attempts.",
            "- Smatch, likelihood, latency, and validity are separate outcomes.",
            "- Paper framing remains a manual decision under the frozen submission rule; no result is suppressed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drs-suite",
        type=Path,
        default=Path("runs/lora/recent-model-and-method-results-v3.json"),
    )
    parser.add_argument(
        "--amr-suite",
        type=Path,
        default=Path("runs/amr/qwen35-4b-base-transfer-suite-v1.json"),
    )
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output-json", type=Path)
    destination.add_argument("--verify-existing-json", type=Path)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_package(args.drs_suite, args.amr_suite)
        markdown = to_markdown(report)
    except (OSError, KeyError, TypeError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.verify_existing_json is not None:
        try:
            stored = _read_object(args.verify_existing_json)
            stored_markdown = args.output_markdown.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        if stored != report:
            parser.error("stored manuscript result JSON differs from canonical suites")
        if stored_markdown != markdown:
            parser.error("stored manuscript result Markdown differs from canonical suites")
        print("verified canonical manuscript result package")
        return 0
    assert args.output_json is not None
    if args.output_json.exists() or args.output_markdown.exists():
        parser.error("refusing to overwrite manuscript result package")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output_json.open("x", encoding="utf-8") as destination_json:
            json.dump(report, destination_json, ensure_ascii=False, indent=2)
            destination_json.write("\n")
        with args.output_markdown.open("x", encoding="utf-8") as destination_markdown:
            destination_markdown.write(markdown)
    except FileExistsError:
        parser.error("refusing to overwrite manuscript result package")
    print(f"manuscript result package: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
