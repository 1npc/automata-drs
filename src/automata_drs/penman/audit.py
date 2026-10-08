"""Reproducible corpus conformance audits for incremental PENMAN states.

The audit deliberately distinguishes three questions:

1. Does an independent PENMAN implementation decode the corpus surface?
2. Does :func:`validate_penman` accept that same, unnormalized surface?
3. Does it accept the independent implementation's normalized serialization?

Only (2) measures whether the online constraint would falsely reject the
actual strings seen by a decoder.  Normalized coverage is a diagnostic that
helps separate unsupported surface notation from graph-language gaps.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import math
import re
from typing import Any

from .state import PENMANPolicy, validate_penman


_METADATA_RE = re.compile(r"^\s*#\s*::([^\s]+)(?:\s+(.*?))?\s*$")
_ALIGNMENT_RE = re.compile(r"~(?:[A-Za-z]\.?)?\d")
_INVERSE_ROLE_RE = re.compile(r":[^\s():\"/]+-of(?=\s|\(|\")")


@dataclass(frozen=True, slots=True)
class PENMANCorpusRecord:
    """One blank-line-delimited PENMAN corpus record."""

    ordinal: int
    start_line: int
    metadata: tuple[tuple[str, str], ...]
    comments: tuple[str, ...]
    surface: str

    @property
    def identifier(self) -> str | None:
        for key, value in self.metadata:
            if key == "id":
                return value
        return None


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_penman_records(lines: Iterable[str]) -> Iterator[PENMANCorpusRecord]:
    """Yield graph records while treating whole-line comments as transport data.

    AMR corpora conventionally put metadata and comments before a graph and
    separate records with blank lines.  These lines are not part of the string
    that a constrained graph decoder produces, so the raw-surface audit strips
    only whole-line comments and surrounding record whitespace.  Internal
    graph whitespace and line breaks remain byte-for-character unchanged.
    """

    block: list[tuple[int, str]] = []
    ordinal = 0

    def emit() -> PENMANCorpusRecord | None:
        nonlocal ordinal
        if not block:
            return None
        metadata: list[tuple[str, str]] = []
        comments: list[str] = []
        graph_lines: list[str] = []
        for _, line in block:
            if line.lstrip().startswith("#"):
                comments.append(line)
                match = _METADATA_RE.match(line)
                if match:
                    metadata.append((match.group(1), match.group(2) or ""))
            else:
                graph_lines.append(line)
        surface = "\n".join(graph_lines).strip()
        if not surface:
            return None
        ordinal += 1
        return PENMANCorpusRecord(
            ordinal=ordinal,
            start_line=block[0][0],
            metadata=tuple(metadata),
            comments=tuple(comments),
            surface=surface,
        )

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r\n")
        if line.strip():
            block.append((line_number, line))
            continue
        record = emit()
        if record is not None:
            yield record
        block = []

    record = emit()
    if record is not None:
        yield record


def load_penman_records(path: Path) -> list[PENMANCorpusRecord]:
    with path.open(encoding="utf-8") as source:
        return list(iter_penman_records(source))


def _maximum_nesting_depth(surface: str) -> int:
    depth = 0
    maximum = 0
    in_string = False
    escaped = False
    for char in surface:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "(":
            depth += 1
            maximum = max(maximum, depth)
        elif char == ")":
            depth = max(0, depth - 1)
    return maximum


def _surface_features(record: PENMANCorpusRecord) -> dict[str, bool]:
    surface = record.surface
    return {
        "metadata": bool(record.metadata),
        "comments": bool(record.comments),
        "multiline": "\n" in surface,
        "quoted_string": '"' in surface,
        "surface_alignment": bool(_ALIGNMENT_RE.search(surface)),
        "inverse_role": bool(_INVERSE_ROLE_RE.search(surface)),
        "non_ascii": not surface.isascii(),
    }


def _triple_parts(triple: Any) -> tuple[Any, Any, Any]:
    if all(hasattr(triple, name) for name in ("source", "role", "target")):
        return triple.source, triple.role, triple.target
    source, role, target = triple
    return source, role, target


def _graph_features(graph: Any) -> dict[str, bool]:
    """Extract conservative features through the public ``penman.Graph`` API."""

    try:
        variables = set(graph.variables())
        edges = [_triple_parts(triple) for triple in graph.edges()]
    except (AttributeError, TypeError, ValueError):
        return {"reentrancy": False}
    incoming = Counter(
        target for _, _, target in edges if target in variables
    )
    return {"reentrancy": any(count > 1 for count in incoming.values())}


def _safe_reason(error: BaseException) -> str:
    first_line = str(error).splitlines()[0].strip()
    return (first_line or type(error).__name__)[:240]


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _coverage_summary(
    *,
    total: int,
    accepted: int,
    reasons: Counter[str],
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "denominator": total,
        "accepted": accepted,
        "rejected": total - accepted,
        "acceptance_rate": accepted / total if total else None,
        "reasons": dict(sorted(reasons.items())),
        "examples": examples,
    }


def audit_penman_corpus(
    path: Path,
    *,
    decode: Callable[[str], Any],
    encode: Callable[[Any], str],
    oracle_name: str,
    oracle_version: str,
    policy: PENMANPolicy | None = None,
    source: Mapping[str, Any] | None = None,
    example_limit: int = 5,
) -> dict[str, Any]:
    """Compare the incremental validator with an independent PENMAN oracle."""

    active_policy = policy or PENMANPolicy()
    records = load_penman_records(path)
    official_reasons: Counter[str] = Counter()
    raw_reasons: Counter[str] = Counter()
    normalized_reasons: Counter[str] = Counter()
    official_examples: list[dict[str, Any]] = []
    raw_examples: list[dict[str, Any]] = []
    normalized_examples: list[dict[str, Any]] = []
    feature_totals: Counter[str] = Counter()
    feature_raw_accepts: Counter[str] = Counter()
    feature_normalized_accepts: Counter[str] = Counter()
    depths: list[int] = []
    official_accepted = 0
    raw_accepted = 0
    normalized_accepted = 0

    for record in records:
        depths.append(_maximum_nesting_depth(record.surface))
        surface_features = _surface_features(record)
        try:
            graph = decode(record.surface)
        except Exception as error:  # independent parser exceptions are data
            reason = _safe_reason(error)
            official_reasons[reason] += 1
            if len(official_examples) < example_limit:
                official_examples.append(
                    {
                        "ordinal": record.ordinal,
                        "id": record.identifier,
                        "start_line": record.start_line,
                        "reason": reason,
                    }
                )
            continue

        official_accepted += 1
        features = surface_features | _graph_features(graph)
        for name, present in features.items():
            if present:
                feature_totals[name] += 1

        raw_result = validate_penman(record.surface, active_policy)
        if raw_result.valid:
            raw_accepted += 1
            for name, present in features.items():
                if present:
                    feature_raw_accepts[name] += 1
        else:
            reason = raw_result.reason or "unknown local rejection"
            raw_reasons[reason] += 1
            if len(raw_examples) < example_limit:
                raw_examples.append(
                    {
                        "ordinal": record.ordinal,
                        "id": record.identifier,
                        "start_line": record.start_line,
                        "reason": reason,
                        "features": sorted(
                            name for name, present in features.items() if present
                        ),
                    }
                )

        try:
            normalized_surface = encode(graph).strip()
        except Exception as error:
            reason = f"oracle encode failed: {_safe_reason(error)}"
            normalized_reasons[reason] += 1
            if len(normalized_examples) < example_limit:
                normalized_examples.append(
                    {
                        "ordinal": record.ordinal,
                        "id": record.identifier,
                        "start_line": record.start_line,
                        "reason": reason,
                    }
                )
            continue

        normalized_result = validate_penman(normalized_surface, active_policy)
        if normalized_result.valid:
            normalized_accepted += 1
            for name, present in features.items():
                if present:
                    feature_normalized_accepts[name] += 1
        else:
            reason = normalized_result.reason or "unknown local rejection"
            normalized_reasons[reason] += 1
            if len(normalized_examples) < example_limit:
                normalized_examples.append(
                    {
                        "ordinal": record.ordinal,
                        "id": record.identifier,
                        "start_line": record.start_line,
                        "reason": reason,
                        "features": sorted(
                            name for name, present in features.items() if present
                        ),
                    }
                )

    feature_coverage = {}
    for name in sorted(feature_totals):
        total = feature_totals[name]
        raw = feature_raw_accepts[name]
        normalized = feature_normalized_accepts[name]
        feature_coverage[name] = {
            "officially_decoded_records": total,
            "raw_accepted": raw,
            "raw_acceptance_rate": raw / total,
            "normalized_accepted": normalized,
            "normalized_acceptance_rate": normalized / total,
        }

    source_details = dict(source or {})
    source_details.update(
        {
            "file_name": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    )
    return {
        "schema_version": 1,
        "audit": "penman-corpus-conformance",
        "source": source_details,
        "oracle": {"name": oracle_name, "version": oracle_version},
        "local_policy": asdict(active_policy),
        "corpus": {
            "records": len(records),
            "records_with_metadata": sum(bool(record.metadata) for record in records),
            "records_with_comments": sum(bool(record.comments) for record in records),
            "maximum_nesting_depth": max(depths, default=None),
            "nesting_depth_p50": _percentile(depths, 0.50),
            "nesting_depth_p95": _percentile(depths, 0.95),
        },
        "official_decode": _coverage_summary(
            total=len(records),
            accepted=official_accepted,
            reasons=official_reasons,
            examples=official_examples,
        ),
        "local_raw_on_official": _coverage_summary(
            total=official_accepted,
            accepted=raw_accepted,
            reasons=raw_reasons,
            examples=raw_examples,
        ),
        "local_normalized_on_official": _coverage_summary(
            total=official_accepted,
            accepted=normalized_accepted,
            reasons=normalized_reasons,
            examples=normalized_examples,
        ),
        "feature_coverage": feature_coverage,
        "interpretation": {
            "primary_false_rejection_measure": "local_raw_on_official",
            "normalized_is_diagnostic_only": True,
            "metadata_is_transport_not_decoder_surface": True,
            "semantic_amr_validity_measured": False,
        },
    }


__all__ = [
    "PENMANCorpusRecord",
    "audit_penman_corpus",
    "iter_penman_records",
    "load_penman_records",
    "sha256_file",
]
