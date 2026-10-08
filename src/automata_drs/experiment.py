"""Aggregate and pair constrained-decoding experiment JSONL records."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Iterable


CONFIGURATION_ORDER = {
    name: index
    for index, name in enumerate(
        ("U", "PostW", "S", "R", "G", "B", "BF", "W", "WF")
    )
}


def _strategy_key(row: dict[str, Any]) -> str:
    decode = row["decode"]
    if decode["strategy"] == "greedy":
        return "greedy"
    return f"beam{decode['num_beams']}"


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _wellformed(row: dict[str, Any]) -> bool:
    validity = row["validity"]
    return (
        validity.get("raw_sbn", validity.get("sbn")) is True
        and validity.get(
            "raw_clf_referee", validity.get("clf_referee")
        )
        is True
    )


def _raw_sbn_valid(row: dict[str, Any]) -> bool:
    validity = row["validity"]
    return validity.get("raw_sbn", validity.get("sbn")) is True


def summarize_experiment(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    seen: set[tuple[str, str, str]] = set()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in rows:
        strategy = _strategy_key(row)
        identity = (str(row["id"]), str(row["configuration"]), strategy)
        if identity in seen:
            raise ValueError(f"duplicate experiment record: {identity}")
        seen.add(identity)
        groups[(identity[1], strategy)].append(row)
        indexed[identity] = row

    group_summaries: list[dict[str, Any]] = []
    for (configuration, strategy), group in sorted(
        groups.items(),
        key=lambda item: (
            CONFIGURATION_ORDER.get(item[0][0], 999),
            item[0][1],
        ),
    ):
        total = len(group)
        raw_sbn_valid = sum(
            row["validity"].get("raw_sbn", row["validity"].get("sbn")) is True
            for row in group
        )
        sbn_valid = sum(row["validity"].get("sbn") is True for row in group)
        referee_scored = sum(
            row["validity"].get("clf_referee") is not None for row in group
        )
        referee_valid = sum(
            row["validity"].get("clf_referee") is True for row in group
        )
        raw_referee_scored = sum(
            row["validity"].get(
                "raw_clf_referee", row["validity"].get("clf_referee")
            )
            is not None
            and row["validity"].get(
                "raw_sbn", row["validity"].get("sbn")
            )
            is True
            for row in group
        )
        raw_referee_valid = sum(
            row["validity"].get(
                "raw_sbn", row["validity"].get("sbn")
            )
            is True
            and row["validity"].get(
                "raw_clf_referee", row["validity"].get("clf_referee")
            )
            is True
            for row in group
        )
        generation_ms = [float(row["timing"]["generation_ms"]) for row in group]
        constraint_ms = [float(row["timing"].get("constraint_ms", 0.0)) for row in group]
        decoder_steps = [
            int(row.get("generation", {}).get("decoder_steps", 0)) for row in group
        ]
        output_tokens = [
            int(row.get("generation", {}).get("output_tokens_including_eos", 0))
            for row in group
        ]
        canonical_scored = sum(
            row.get("generation", {}).get("canonical_tokenization") is not None
            for row in group
        )
        canonical_tokenization = sum(
            row.get("generation", {}).get("canonical_tokenization") is True
            for row in group
        )
        raw_mean_logprobs = [
            float(value)
            for row in group
            if (value := row.get("score", {}).get("raw_model_mean_logprob"))
            is not None
        ]
        beam_selection_rows = [
            row["beam_selection"]
            for row in group
            if isinstance(row.get("beam_selection"), dict)
            and row["beam_selection"].get("method")
            == "highest-sequence-score-w-valid-u-beam"
        ]
        postw_valid_candidate_counts = [
            float(selection["w_valid_candidate_count"])
            for selection in beam_selection_rows
            if isinstance(selection.get("w_valid_candidate_count"), int)
            and selection["w_valid_candidate_count"] >= 0
        ]
        allowed_means = [
            float(value)
            for row in group
            if row.get("constraint") is not None
            and (value := row["constraint"].get("allowed_mean")) is not None
        ]
        all_masked_rows = sum(
            int(row.get("constraint", {}).get("all_masked_rows", 0))
            for row in group
            if row.get("constraint") is not None
        )
        certificate_tests = sum(
            int(row.get("constraint", {}).get("completion_certificates_tested", 0))
            for row in group
            if row.get("constraint") is not None
        )
        bounded_rows = sum(
            int(row.get("constraint", {}).get("bounded_rows", 0))
            for row in group
            if row.get("constraint") is not None
        )
        certified_limit_hits = sum(
            int(row.get("constraint", {}).get("certified_limit_hits", 0))
            for row in group
            if row.get("constraint") is not None
        )
        certificates_accepted = sum(
            int(row.get("constraint", {}).get("completion_certificates_accepted", 0))
            for row in group
            if row.get("constraint") is not None
        )
        witness_tokens_total = sum(
            int(row.get("constraint", {}).get("completion_witness_tokens_total", 0))
            for row in group
            if row.get("constraint") is not None
        )
        completion_slack_total = sum(
            int(row.get("constraint", {}).get("completion_slack_total", 0))
            for row in group
            if row.get("constraint") is not None
        )
        completion_slack_mins = [
            int(value)
            for row in group
            if row.get("constraint") is not None
            and (value := row["constraint"].get("completion_slack_min"))
            is not None
        ]
        smatch_scores = [
            float(value)
            for row in group
            if (value := row.get("semantic_scores", {}).get("sbn_smatch"))
            is not None
        ]
        group_summaries.append(
            {
                "configuration": configuration,
                "strategy": strategy,
                "total": total,
                "raw_sbn_valid": raw_sbn_valid,
                "raw_sbn_valid_rate": raw_sbn_valid / total if total else None,
                "sbn_valid": sbn_valid,
                "sbn_valid_rate": sbn_valid / total if total else None,
                "referee_scored": referee_scored,
                "referee_valid": referee_valid,
                "referee_valid_rate_total": (
                    referee_valid / total if total else None
                ),
                "referee_valid_rate_scored": (
                    referee_valid / referee_scored if referee_scored else None
                ),
                "raw_referee_scored": raw_referee_scored,
                "raw_referee_valid": raw_referee_valid,
                "raw_referee_valid_rate_total": (
                    raw_referee_valid / total if total else None
                ),
                "raw_referee_valid_rate_scored": (
                    raw_referee_valid / raw_referee_scored
                    if raw_referee_scored
                    else None
                ),
                "exact_serialization": sum(
                    row.get("prediction") == row.get("reference") for row in group
                ),
                "truncated": sum(
                    row.get("generation", {}).get("hit_max_new_tokens") is True
                    for row in group
                ),
                "canonical_tokenization_scored": canonical_scored,
                "canonical_tokenization": canonical_tokenization,
                "canonical_tokenization_rate": (
                    canonical_tokenization / canonical_scored
                    if canonical_scored
                    else None
                ),
                "generation_ms_mean": _mean(generation_ms),
                "generation_ms_median": _median(generation_ms),
                "generation_ms_p95": _percentile(generation_ms, 0.95),
                "constraint_ms_mean": _mean(constraint_ms),
                "decoder_steps_mean": _mean([float(value) for value in decoder_steps]),
                "output_tokens_mean": _mean([float(value) for value in output_tokens]),
                "generation_ms_per_step": (
                    sum(generation_ms) / sum(decoder_steps)
                    if sum(decoder_steps)
                    else None
                ),
                "constraint_ms_per_step": (
                    sum(constraint_ms) / sum(decoder_steps)
                    if sum(decoder_steps)
                    else None
                ),
                "raw_model_mean_logprob_mean": _mean(raw_mean_logprobs),
                "postw_selection_records": len(beam_selection_rows),
                "postw_fallbacks": sum(
                    selection.get("fallback_to_u_top1") is True
                    for selection in beam_selection_rows
                ),
                "postw_w_valid_candidates_mean": _mean(
                    postw_valid_candidate_counts
                ),
                "allowed_tokens_mean": _mean(allowed_means),
                "all_masked_rows": all_masked_rows,
                "completion_certificates_tested_per_step": (
                    certificate_tests / sum(decoder_steps)
                    if sum(decoder_steps)
                    else None
                ),
                "certified_limit_hit_rate": (
                    certified_limit_hits / bounded_rows if bounded_rows else None
                ),
                "completion_witness_tokens_mean": (
                    witness_tokens_total / certificates_accepted
                    if certificates_accepted
                    else None
                ),
                "completion_slack_mean": (
                    completion_slack_total / certificates_accepted
                    if certificates_accepted
                    else None
                ),
                "completion_slack_min": (
                    min(completion_slack_mins)
                    if completion_slack_mins
                    else None
                ),
                "sbn_smatch_mean": _mean(smatch_scores),
            }
        )

    pair_summaries: list[dict[str, Any]] = []
    for configuration, strategy in sorted(
        (key for key in groups if key[0] != "U"),
        key=lambda key: (CONFIGURATION_ORDER.get(key[0], 999), key[1]),
    ):
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        missing_u = 0
        for constrained in groups[(configuration, strategy)]:
            unconstrained = indexed.get((str(constrained["id"]), "U", strategy))
            if unconstrained is None:
                missing_u += 1
                continue
            pairs.append((unconstrained, constrained))

        logprob_deltas = [
            float(constrained["score"]["raw_model_logprob"])
            - float(unconstrained["score"]["raw_model_logprob"])
            for unconstrained, constrained in pairs
            if unconstrained.get("score", {}).get("raw_model_logprob") is not None
            and constrained.get("score", {}).get("raw_model_logprob") is not None
        ]
        mean_logprob_deltas = [
            float(constrained["score"]["raw_model_mean_logprob"])
            - float(unconstrained["score"]["raw_model_mean_logprob"])
            for unconstrained, constrained in pairs
            if unconstrained.get("score", {}).get("raw_model_mean_logprob")
            is not None
            and constrained.get("score", {}).get("raw_model_mean_logprob")
            is not None
        ]
        generation_deltas = [
            float(constrained["timing"]["generation_ms"])
            - float(unconstrained["timing"]["generation_ms"])
            for unconstrained, constrained in pairs
        ]
        decoder_step_deltas = [
            int(constrained.get("generation", {}).get("decoder_steps", 0))
            - int(unconstrained.get("generation", {}).get("decoder_steps", 0))
            for unconstrained, constrained in pairs
        ]
        smatch_deltas = [
            float(constrained["semantic_scores"]["sbn_smatch"])
            - float(unconstrained["semantic_scores"]["sbn_smatch"])
            for unconstrained, constrained in pairs
            if unconstrained.get("semantic_scores", {}).get("sbn_smatch")
            is not None
            and constrained.get("semantic_scores", {}).get("sbn_smatch")
            is not None
        ]
        pair_summaries.append(
            {
                "configuration": configuration,
                "strategy": strategy,
                "paired": len(pairs),
                "missing_unconstrained": missing_u,
                "prediction_changed": sum(
                    unconstrained.get("prediction") != constrained.get("prediction")
                    for unconstrained, constrained in pairs
                ),
                "raw_sbn_fixed": sum(
                    not _raw_sbn_valid(unconstrained)
                    and _raw_sbn_valid(constrained)
                    for unconstrained, constrained in pairs
                ),
                "raw_sbn_regressed": sum(
                    _raw_sbn_valid(unconstrained)
                    and not _raw_sbn_valid(constrained)
                    for unconstrained, constrained in pairs
                ),
                "sbn_fixed": sum(
                    unconstrained["validity"].get("sbn") is not True
                    and constrained["validity"].get("sbn") is True
                    for unconstrained, constrained in pairs
                ),
                "sbn_regressed": sum(
                    unconstrained["validity"].get("sbn") is True
                    and constrained["validity"].get("sbn") is not True
                    for unconstrained, constrained in pairs
                ),
                "wellformed_fixed": sum(
                    not _wellformed(unconstrained) and _wellformed(constrained)
                    for unconstrained, constrained in pairs
                ),
                "wellformed_regressed": sum(
                    _wellformed(unconstrained) and not _wellformed(constrained)
                    for unconstrained, constrained in pairs
                ),
                "generation_ms_delta_mean": _mean(generation_deltas),
                "decoder_steps_delta_mean": _mean(
                    [float(value) for value in decoder_step_deltas]
                ),
                "raw_model_logprob_delta_mean": _mean(logprob_deltas),
                "raw_model_logprob_delta_median": _median(logprob_deltas),
                "raw_model_mean_logprob_delta_mean": _mean(mean_logprob_deltas),
                "raw_model_mean_logprob_delta_median": _median(
                    mean_logprob_deltas
                ),
                "sbn_smatch_delta_mean": _mean(smatch_deltas),
            }
        )

    return {
        "records": len(rows),
        "unique_examples": len({str(row["id"]) for row in rows}),
        "groups": group_summaries,
        "paired_vs_u": pair_summaries,
    }


def _fmt_rate(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def _fmt_number(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def summary_to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Constrained decoding experiment summary",
        "",
        f"Records: {summary['records']}; unique examples: {summary['unique_examples']}.",
        "",
        "## Aggregate results",
        "",
        "| Config | Decode | N | Raw SBN valid | Postprocessed SBN valid | Referee valid | Smatch | Exact | Canonical tok | Truncated | All-masked | Steps | Output tok | Gen ms/step | Constraint ms/step | Raw mean log-p | Allowed | Cert tests/step | Witness tok | Mean/min slack | Limit hit |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["groups"]:
        lines.append(
            "| {configuration} | {strategy} | {total} | {raw_sbn_rate} | {sbn_rate} | "
            "{referee_rate} | {smatch} | {exact_serialization} | {canonical_rate} | {truncated} | {all_masked_rows} | {steps} | "
            "{output_tokens} | {gen_per_step} | {constraint_per_step} | {logprob} | {allowed} | {certificate_tests} | {witness_tokens} | {slack_mean}/{slack_min} | {limit_hit} |".format(
                **row,
                raw_sbn_rate=_fmt_rate(row["raw_sbn_valid_rate"]),
                sbn_rate=_fmt_rate(row["sbn_valid_rate"]),
                referee_rate=_fmt_rate(row["referee_valid_rate_total"]),
                canonical_rate=_fmt_rate(row["canonical_tokenization_rate"]),
                smatch=_fmt_number(row["sbn_smatch_mean"], 4),
                steps=_fmt_number(row["decoder_steps_mean"]),
                output_tokens=_fmt_number(row["output_tokens_mean"]),
                gen_per_step=_fmt_number(row["generation_ms_per_step"], 3),
                constraint_per_step=_fmt_number(
                    row["constraint_ms_per_step"], 3
                ),
                logprob=_fmt_number(row["raw_model_mean_logprob_mean"], 4),
                allowed=_fmt_number(row["allowed_tokens_mean"]),
                certificate_tests=_fmt_number(
                    row["completion_certificates_tested_per_step"]
                ),
                witness_tokens=_fmt_number(
                    row["completion_witness_tokens_mean"]
                ),
                slack_mean=_fmt_number(row["completion_slack_mean"]),
                slack_min=_fmt_number(row["completion_slack_min"], 0),
                limit_hit=_fmt_rate(row["certified_limit_hit_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "## Paired changes versus U",
            "",
            "| Config | Decode | Pairs | Changed | Raw SBN fixed/regressed | Postprocessed SBN fixed/regressed | WF fixed/regressed | Δ Smatch | Δ steps | Δ gen ms | Δ raw log-p | Δ raw mean log-p | Missing U |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["paired_vs_u"]:
        lines.append(
            "| {configuration} | {strategy} | {paired} | {prediction_changed} | "
            "{raw_sbn_fixed}/{raw_sbn_regressed} | {sbn_fixed}/{sbn_regressed} | {wellformed_fixed}/{wellformed_regressed} | {smatch_delta} | {step_delta} | "
            "{gen_delta} | {logprob_delta} | {mean_logprob_delta} | {missing_unconstrained} |".format(
                **row,
                gen_delta=_fmt_number(row["generation_ms_delta_mean"]),
                logprob_delta=_fmt_number(row["raw_model_logprob_delta_mean"], 3),
                mean_logprob_delta=_fmt_number(
                    row["raw_model_mean_logprob_delta_mean"], 4
                ),
                smatch_delta=_fmt_number(row["sbn_smatch_delta_mean"], 4),
                step_delta=_fmt_number(row["decoder_steps_delta_mean"]),
            )
        )
    return "\n".join(lines) + "\n"


__all__ = ["summarize_experiment", "summary_to_markdown"]
