#!/usr/bin/env python3
"""Build the paper's empirical finite-horizon budget-cliff figure.

The figure is derived from a frozen paired U/BF/WF JSONL.  It does not invent a
trajectory: the selected BF surface is replayed byte by byte through the same
finite-horizon certificate oracle, and the reconstructed witness/slack totals
must match the counters stored by generation before any output is written.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import html
import json
from pathlib import Path
from typing import Any, Iterable

from automata_drs import (
    ByT5ConstraintState,
    ByT5SBNLogitsProcessor,
    ByT5Vocabulary,
    SBNPolicy,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPOSITORY_ROOT / "runs/server-tax-20-u-bf-wf-greedy-v2.jsonl"
DEFAULT_RECORD_ID = "data/en/test/tax_challenge.sbn:9"
DEFAULT_SVG = REPOSITORY_ROOT / "figures/budget_cliff_tax9.svg"
DEFAULT_DATA = REPOSITORY_ROOT / "figures/data/budget_cliff_tax9_trace.json"


@dataclass(frozen=True, slots=True)
class TracePoint:
    step: int
    token_id: int
    byte: int | None
    remaining_after_candidate: int
    witness_tokens: int
    slack_tokens: int


@dataclass(frozen=True, slots=True)
class Divergence:
    step: int
    remaining_after_candidate: int
    unconstrained_byte: int
    bounded_byte: int
    unconstrained_locally_live: bool
    unconstrained_witness_tokens: int | None
    bounded_locally_live: bool
    bounded_witness_tokens: int | None
    bounded_slack_tokens: int | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rows(path: Path, record_id: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") != record_id:
                continue
            configuration = row.get("configuration")
            if configuration in rows:
                raise ValueError(
                    f"duplicate {configuration!r} row for {record_id!r} "
                    f"at {path}:{line_number}"
                )
            rows[configuration] = row
    missing = {"U", "BF", "WF"} - rows.keys()
    if missing:
        raise ValueError(f"missing configurations for {record_id!r}: {sorted(missing)}")
    return rows


def _advance_surface(surface: str, policy: SBNPolicy) -> ByT5ConstraintState:
    state = ByT5ConstraintState.initial(policy)
    for offset, byte in enumerate(surface.encode("utf-8")):
        successor = state.advance_byte(byte)
        if successor is None:
            raise ValueError(f"surface becomes dead at byte offset {offset}")
        state = successor
    return state


def reconstruct_trace(
    row: dict[str, Any],
    *,
    max_new_tokens: int,
) -> tuple[list[TracePoint], ByT5SBNLogitsProcessor]:
    """Replay one BF row and return the exact selected-path certificate trace."""
    if row.get("configuration") != "BF":
        raise ValueError("the trace source must be the BF row")
    generation = row.get("generation") or {}
    if generation.get("output_tokens_including_eos") != max_new_tokens:
        raise ValueError("the selected row does not end exactly at the hard cap")
    if not generation.get("ended_with_eos") or generation.get("hit_max_new_tokens"):
        raise ValueError("the BF row must end with EOS without truncation")

    policy = SBNPolicy()
    vocabulary = ByT5Vocabulary()
    processor = ByT5SBNLogitsProcessor(
        policy=policy,
        max_new_tokens=max_new_tokens,
        bounded_completion_window=max_new_tokens,
        max_certified_candidates=1,
        completion_budget_factor=1.0,
    )
    surface_tokens = [
        vocabulary.token_for_byte(byte)
        for byte in row["prediction"].encode("utf-8")
    ]
    tokens = surface_tokens + [vocabulary.eos_token_id]
    if len(tokens) != max_new_tokens:
        raise ValueError(
            f"expected {max_new_tokens} selected tokens including EOS, got {len(tokens)}"
        )

    prefix = [vocabulary.pad_token_id]
    trace: list[TracePoint] = []
    for step, token_id in enumerate(tokens, 1):
        state = processor.prefix_states.state_for(prefix)
        if state is None:
            raise ValueError(f"BF prefix is dead before step {step}")
        successor = processor.controller._compute_transition(state, token_id)
        if successor is None:
            raise ValueError(f"BF selected token is dead at step {step}")
        remaining_after_candidate = max_new_tokens - step
        distance = processor._successor_completion_distance(
            successor, remaining_after_candidate
        )
        if distance is None:
            raise ValueError(f"BF selected token has no certificate at step {step}")
        trace.append(
            TracePoint(
                step=step,
                token_id=token_id,
                byte=(
                    None
                    if token_id == vocabulary.eos_token_id
                    else vocabulary.byte_for_token(token_id)
                ),
                remaining_after_candidate=remaining_after_candidate,
                witness_tokens=distance,
                slack_tokens=remaining_after_candidate - distance,
            )
        )
        prefix.append(token_id)

    final_state = processor.prefix_states.state_for(prefix)
    if final_state is None or not final_state.is_accepting:
        raise ValueError("replayed BF path does not finish in an accepting state")
    return trace, processor


def find_divergence(
    unconstrained: dict[str, Any],
    bounded: dict[str, Any],
    processor: ByT5SBNLogitsProcessor,
    *,
    max_new_tokens: int,
) -> Divergence:
    """Audit the first U/BF byte divergence under the BF prefix state."""
    vocabulary = ByT5Vocabulary()
    u_bytes = unconstrained["prediction"].encode("utf-8")
    b_bytes = bounded["prediction"].encode("utf-8")
    common = next(
        (index for index, pair in enumerate(zip(u_bytes, b_bytes)) if pair[0] != pair[1]),
        None,
    )
    if common is None:
        raise ValueError("U and BF do not diverge within their shared byte span")

    prefix = [vocabulary.pad_token_id]
    prefix.extend(vocabulary.token_for_byte(byte) for byte in b_bytes[:common])
    state = processor.prefix_states.state_for(prefix)
    if state is None:
        raise ValueError("shared U/BF prefix is dead at divergence")
    remaining = max_new_tokens - common - 1

    def audit(byte: int) -> tuple[bool, int | None]:
        successor = processor.controller._compute_transition(
            state, vocabulary.token_for_byte(byte)
        )
        if successor is None:
            return False, None
        return True, processor._successor_completion_distance(successor, remaining)

    u_live, u_distance = audit(u_bytes[common])
    b_live, b_distance = audit(b_bytes[common])
    return Divergence(
        step=common + 1,
        remaining_after_candidate=remaining,
        unconstrained_byte=u_bytes[common],
        bounded_byte=b_bytes[common],
        unconstrained_locally_live=u_live,
        unconstrained_witness_tokens=u_distance,
        bounded_locally_live=b_live,
        bounded_witness_tokens=b_distance,
        bounded_slack_tokens=(None if b_distance is None else remaining - b_distance),
    )


def _validate_stored_counters(trace: Iterable[TracePoint], row: dict[str, Any]) -> None:
    points = list(trace)
    stored = row.get("constraint") or {}
    reconstructed = {
        "completion_witness_tokens_total": sum(p.witness_tokens for p in points),
        "completion_witness_tokens_min": min(p.witness_tokens for p in points),
        "completion_witness_tokens_max": max(p.witness_tokens for p in points),
        "completion_slack_total": sum(p.slack_tokens for p in points),
        "completion_slack_min": min(p.slack_tokens for p in points),
        "completion_slack_max": max(p.slack_tokens for p in points),
    }
    mismatches = {
        key: {"stored": stored.get(key), "reconstructed": value}
        for key, value in reconstructed.items()
        if stored.get(key) != value
    }
    if mismatches:
        raise ValueError(f"reconstructed trace does not match stored counters: {mismatches}")


def build_evidence(
    input_path: Path,
    record_id: str,
    *,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    rows = _load_rows(input_path, record_id)
    trace, processor = reconstruct_trace(rows["BF"], max_new_tokens=max_new_tokens)
    _validate_stored_counters(trace, rows["BF"])
    divergence = find_divergence(
        rows["U"], rows["BF"], processor, max_new_tokens=max_new_tokens
    )

    u_state = _advance_surface(rows["U"]["prediction"], SBNPolicy())
    bf_state = _advance_surface(rows["BF"]["prediction"], SBNPolicy())
    if not u_state.is_extendable or u_state.can_end():
        raise ValueError("U terminal surface is not the expected locally-live open prefix")
    if not bf_state.can_end():
        raise ValueError("BF surface cannot accept EOS")
    if not divergence.unconstrained_locally_live:
        raise ValueError("U divergence token is not locally live")
    if divergence.unconstrained_witness_tokens is not None:
        raise ValueError("U divergence token unexpectedly has a bounded certificate")
    if divergence.bounded_witness_tokens is None:
        raise ValueError("BF divergence token lacks its expected bounded certificate")

    return {
        "schema_version": 1,
        "source": {
            "path": str(input_path.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256(input_path),
            "record_id": record_id,
            "max_new_tokens": max_new_tokens,
            "scope": "frozen legacy-ByT5 tax20 mechanism case; not a population result",
        },
        "input": rows["U"]["input"],
        "divergence": asdict(divergence),
        "unconstrained": {
            "prediction": rows["U"]["prediction"],
            "output_tokens_including_eos": rows["U"]["generation"][
                "output_tokens_including_eos"
            ],
            "ended_with_eos": rows["U"]["generation"]["ended_with_eos"],
            "hit_max_new_tokens": rows["U"]["generation"]["hit_max_new_tokens"],
            "locally_extendable_at_cap": u_state.is_extendable,
            "can_end_at_cap": u_state.can_end(),
            "validity": rows["U"]["validity"],
        },
        "bounded": {
            "prediction": rows["BF"]["prediction"],
            "output_tokens_including_eos": rows["BF"]["generation"][
                "output_tokens_including_eos"
            ],
            "ended_with_eos": rows["BF"]["generation"]["ended_with_eos"],
            "hit_max_new_tokens": rows["BF"]["generation"]["hit_max_new_tokens"],
            "can_end_before_eos": bf_state.can_end(),
            "validity": rows["BF"]["validity"],
            "stored_constraint": rows["BF"]["constraint"],
        },
        "trace": [asdict(point) for point in trace],
    }


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _short_surface(surface: str, *, start: int, end: int) -> str:
    encoded = surface.encode("utf-8")
    prefix = "…" if start else ""
    suffix = "…" if end < len(encoded) else ""
    return prefix + encoded[start:end].decode("utf-8", errors="replace") + suffix


def _polyline(points: Iterable[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _render_svg_legacy(evidence: dict[str, Any], *, tail_start: int = 192) -> str:
    divergence = evidence["divergence"]
    trace = [point for point in evidence["trace"] if point["step"] >= tail_start]
    max_new_tokens = evidence["source"]["max_new_tokens"]
    plot_x0, plot_y0, plot_w, plot_h = 760.0, 278.0, 760.0, 410.0
    x_min, x_max = tail_start, max_new_tokens
    y_max = max(point["remaining_after_candidate"] for point in trace)

    def x(step: int) -> float:
        return plot_x0 + (step - x_min) / (x_max - x_min) * plot_w

    def y(value: int) -> float:
        return plot_y0 + plot_h - value / y_max * plot_h

    remaining_points = [(x(p["step"]), y(p["remaining_after_candidate"])) for p in trace]
    witness_points = [(x(p["step"]), y(p["witness_tokens"])) for p in trace]
    slack_points = [(x(p["step"]), y(p["slack_tokens"])) for p in trace]
    fill_points = remaining_points + list(reversed(witness_points))
    zero_slack = [p for p in trace if p["slack_tokens"] == 0]
    divergence_x = x(divergence["step"])

    u_prediction = evidence["unconstrained"]["prediction"]
    bf_prediction = evidence["bounded"]["prediction"]
    split = divergence["step"] - 1
    context_start = max(0, split - 49)
    u_tail = _short_surface(u_prediction, start=context_start, end=len(u_prediction.encode()))
    bf_tail = _short_surface(bf_prediction, start=context_start, end=len(bf_prediction.encode()))
    u_char = chr(divergence["unconstrained_byte"])
    bf_char = chr(divergence["bounded_byte"])
    record_label = evidence["source"]["record_id"].split(":")[-1]

    y_ticks = list(range(0, y_max + 1, 10))
    x_ticks = [192, 208, 224, 240, 256]
    grid = "".join(
        f'<line x1="{plot_x0}" y1="{y(t)}" x2="{plot_x0 + plot_w}" y2="{y(t)}" class="grid"/>'
        f'<text x="{plot_x0 - 14}" y="{y(t) + 5}" text-anchor="end" class="axis">{t}</text>'
        for t in y_ticks
    )
    grid += "".join(
        f'<line x1="{x(t)}" y1="{plot_y0}" x2="{x(t)}" y2="{plot_y0 + plot_h}" class="grid vertical"/>'
        f'<text x="{x(t)}" y="{plot_y0 + plot_h + 28}" text-anchor="middle" class="axis">{t}</text>'
        for t in x_ticks
    )
    zero_marks = "".join(
        f'<circle cx="{x(p["step"]):.2f}" cy="{y(0):.2f}" r="4" class="zero"/>'
        for p in zero_slack
    )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" role="img" aria-labelledby="title desc">
  <title id="title">A real finite-horizon budget cliff in DRS decoding</title>
  <desc id="desc">A frozen paired ByT5 example where the locally live unconstrained token has no accepting completion within the remaining budget, while BF selects a lower-ranked token with a replayed certificate and reaches EOS at the 256-token cap.</desc>
  <defs>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#14213d" flood-opacity="0.10"/></filter>
    <style>
      .title {{ font-family: Inter, Arial, sans-serif; font-size: 34px; font-weight: 700; fill: #14213d; }}
      .subtitle {{ font-family: Inter, Arial, sans-serif; font-size: 17px; fill: #526779; }}
      .panel-title {{ font-family: Inter, Arial, sans-serif; font-size: 21px; font-weight: 700; fill: #14213d; }}
      .panel-letter {{ font-family: Inter, Arial, sans-serif; font-size: 20px; font-weight: 700; fill: #fff; }}
      .section {{ font-family: Inter, Arial, sans-serif; font-size: 15px; font-weight: 700; fill: #263c50; }}
      .body {{ font-family: Inter, Arial, sans-serif; font-size: 14px; fill: #31495d; }}
      .small {{ font-family: Inter, Arial, sans-serif; font-size: 12px; fill: #526779; }}
      .axis {{ font-family: Inter, Arial, sans-serif; font-size: 12px; fill: #526779; }}
      .mono {{ font-family: "SFMono-Regular", Consolas, monospace; font-size: 13px; fill: #20394d; }}
      .mono-small {{ font-family: "SFMono-Regular", Consolas, monospace; font-size: 11.5px; fill: #20394d; }}
      .good {{ fill: #17766c; }} .bad {{ fill: #bd4b3f; }} .amber {{ fill: #9a6710; }}
      .grid {{ stroke: #dfe7ec; stroke-width: 1; }} .grid.vertical {{ stroke-dasharray: 4 5; }}
      .zero {{ fill: #d66a55; stroke: #fff; stroke-width: 1.5; }}
    </style>
  </defs>
  <rect width="1600" height="900" fill="#f4f7f9"/>
  <text x="60" y="58" class="title">A real budget cliff under a 256-token decoding cap</text>
  <text x="60" y="88" class="subtitle">A locally legal token can still be doomed by the remaining native-token horizon; BF keeps only a replayable accepting continuation.</text>

  <rect x="50" y="118" width="640" height="695" rx="18" fill="#fff" filter="url(#shadow)"/>
  <circle cx="82" cy="153" r="19" fill="#d66a55"/><text x="82" y="160" text-anchor="middle" class="panel-letter">A</text>
  <text x="112" y="160" class="panel-title">The first decisive token</text>
  <text x="78" y="197" class="small">Frozen tax20 record {record_label} · shared prefix through byte/token {split}</text>
  <rect x="75" y="216" width="590" height="72" rx="10" fill="#edf4f8" stroke="#bfd2de"/>
  <text x="94" y="242" class="section">Shared SBN tail</text>
  <text x="94" y="269" class="mono-small">{_escape(_short_surface(bf_prediction, start=context_start, end=split))}</text>

  <rect x="75" y="312" width="590" height="116" rx="11" fill="#fff1ed" stroke="#e5a294"/>
  <text x="94" y="340" class="section">U raw top-1 at step {divergence["step"]}: “{_escape(u_char)}”</text>
  <text x="94" y="368" class="body">local transition: OPEN ✓</text>
  <text x="300" y="368" class="bad" font-family="Inter, Arial" font-size="14" font-weight="700">accepting suffix within {divergence["remaining_after_candidate"]} tokens: none found</text>
  <text x="94" y="398" class="mono-small">{_escape(u_tail)}</text>
  <text x="94" y="418" class="bad" font-family="Inter, Arial" font-size="12" font-weight="700">cap reached on “Attrib” · no EOS · invalid completed output</text>

  <rect x="75" y="448" width="590" height="128" rx="11" fill="#eaf7f3" stroke="#78b9ad" stroke-width="2"/>
  <text x="94" y="477" class="section">BF selected at step {divergence["step"]}: “{_escape(bf_char)}”</text>
  <text x="94" y="505" class="body">local transition: OPEN ✓</text>
  <text x="300" y="505" class="good" font-family="Inter, Arial" font-size="14" font-weight="700">witness {divergence["bounded_witness_tokens"]} + slack {divergence["bounded_slack_tokens"]} = remaining {divergence["remaining_after_candidate"]}</text>
  <text x="94" y="535" class="mono-small">{_escape(bf_tail)}</text>
  <text x="94" y="560" class="good" font-family="Inter, Arial" font-size="12" font-weight="700">EOS at step 256 · B-valid and W-valid · zero all-masked rows</text>

  <text x="78" y="615" class="section">What this example proves—and does not prove</text>
  <rect x="75" y="632" width="590" height="125" rx="10" fill="#f7f9fb" stroke="#ccd9e0"/>
  <text x="94" y="660" class="body">✓ Local prefix viability and finite-budget realizability are different.</text>
  <text x="94" y="687" class="body">✓ This positive decision has an exact selected-path certificate.</text>
  <text x="94" y="714" class="body">✗ Policy validity is not semantic correctness or planner completeness.</text>
  <text x="94" y="741" class="small">Mechanism case from the frozen legacy-ByT5 pilot; recent-host population effects remain separate.</text>

  <rect x="720" y="118" width="830" height="695" rx="18" fill="#fff" filter="url(#shadow)"/>
  <circle cx="752" cy="153" r="19" fill="#775da6"/><text x="752" y="160" text-anchor="middle" class="panel-letter">B</text>
  <text x="782" y="160" class="panel-title">Replayed BF certificate trace near the cap</text>
  <text x="750" y="199" class="small">Every point is recomputed from the stored selected surface; aggregate totals must match the immutable generation row.</text>

  {grid}
  <rect x="{plot_x0}" y="{plot_y0}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#bfcdd6" stroke-width="1.2"/>
  <polygon points="{_polyline(fill_points)}" fill="#dff3ed" fill-opacity="0.72"/>
  <polyline points="{_polyline(remaining_points)}" fill="none" stroke="#738897" stroke-width="2" stroke-dasharray="7 5"/>
  <polyline points="{_polyline(witness_points)}" fill="none" stroke="#775da6" stroke-width="3"/>
  <polyline points="{_polyline(slack_points)}" fill="none" stroke="#23867b" stroke-width="2.4"/>
  {zero_marks}
  <line x1="{divergence_x:.2f}" y1="{plot_y0}" x2="{divergence_x:.2f}" y2="{plot_y0 + plot_h}" stroke="#d66a55" stroke-width="2" stroke-dasharray="5 4"/>
  <text x="{divergence_x + 8:.2f}" y="{plot_y0 + 20}" class="bad" font-family="Inter, Arial" font-size="12" font-weight="700">first U/BF divergence</text>
  <text x="{plot_x0 + plot_w / 2}" y="{plot_y0 + plot_h + 48}" text-anchor="middle" class="axis">decoder step (native ByT5 token)</text>
  <text x="{plot_x0 - 54}" y="{plot_y0 + plot_h / 2}" text-anchor="middle" class="axis" transform="rotate(-90 {plot_x0 - 54} {plot_y0 + plot_h / 2})">tokens</text>

  <g transform="translate(765 230)">
    <line x1="0" y1="0" x2="28" y2="0" stroke="#738897" stroke-width="2" stroke-dasharray="7 5"/><text x="36" y="5" class="small">remaining budget r</text>
    <line x1="190" y1="0" x2="218" y2="0" stroke="#775da6" stroke-width="3"/><text x="226" y="5" class="small">witness length d</text>
    <line x1="390" y1="0" x2="418" y2="0" stroke="#23867b" stroke-width="2.4"/><text x="426" y="5" class="small">slack r − d</text>
    <circle cx="576" cy="0" r="4" class="zero"/><text x="588" y="5" class="small">exact-fit certificate</text>
  </g>

  <rect x="750" y="749" width="770" height="48" rx="10" fill="#f2eff9" stroke="#c6b8df"/>
  <text x="770" y="771" class="body">Audit match: Σ witness = {evidence["bounded"]["stored_constraint"]["completion_witness_tokens_total"]}; Σ slack = {evidence["bounded"]["stored_constraint"]["completion_slack_total"]};</text>
  <text x="770" y="790" class="body">min slack = {evidence["bounded"]["stored_constraint"]["completion_slack_min"]}; accepting EOS = true; all-masked = {evidence["bounded"]["stored_constraint"]["all_masked_rows"]}.</text>

  <line x1="60" y1="842" x2="1540" y2="842" stroke="#d9e2e8" stroke-width="1.4"/>
  <text x="60" y="870" class="small">Source-bound empirical trace · {_escape(evidence["source"]["path"])} · SHA-256 {_escape(evidence["source"]["sha256"][:12])}…</text>
  <text x="1540" y="870" text-anchor="end" class="small">Guarantee: positive certificate under the implemented B policy; negative search remains incomplete.</text>
</svg>
'''


def render_svg(evidence: dict[str, Any], *, tail_start: int = 192) -> str:
    """Render a compact, single-column certificate-cost plot."""
    divergence = evidence["divergence"]
    trace = [point for point in evidence["trace"] if point["step"] >= tail_start]
    max_new_tokens = evidence["source"]["max_new_tokens"]
    plot_x0, plot_y0, plot_w, plot_h = 82.0, 78.0, 620.0, 340.0
    x_min, x_max = tail_start, max_new_tokens
    y_max = 64

    def x(step: int) -> float:
        return plot_x0 + (step - x_min) / (x_max - x_min) * plot_w

    def y(value: int) -> float:
        return plot_y0 + plot_h - value / y_max * plot_h

    remaining_points = [(x(p["step"]), y(p["remaining_after_candidate"])) for p in trace]
    witness_points = [(x(p["step"]), y(p["witness_tokens"])) for p in trace]
    divergence_x = x(divergence["step"])
    divergence_remaining_y = y(divergence["remaining_after_candidate"])
    divergence_witness_y = y(divergence["bounded_witness_tokens"])

    y_ticks = list(range(0, 61, 10))
    x_ticks = [192, 208, 224, 240, 256]
    grid = "".join(
        f'<line x1="{plot_x0}" y1="{y(t)}" x2="{plot_x0 + plot_w}" '
        f'y2="{y(t)}" class="grid"/>'
        f'<text x="{plot_x0 - 12}" y="{y(t) + 6}" text-anchor="end" '
        f'class="axis">{t}</text>'
        for t in y_ticks
    )
    x_axis = "".join(
        f'<line x1="{x(t)}" y1="{plot_y0 + plot_h}" x2="{x(t)}" '
        f'y2="{plot_y0 + plot_h + 7}" class="axis-line"/>'
        f'<text x="{x(t)}" y="{plot_y0 + plot_h + 28}" text-anchor="middle" '
        f'class="axis">{t}</text>'
        for t in x_ticks
    )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="520" viewBox="0 0 760 520" role="img" aria-labelledby="title desc">
  <title id="title">Certificate cost near the decoding limit</title>
  <desc id="desc">Remaining native-token budget and replayed accepting-witness length from decoder step 192 to 256.  The first U versus BF divergence has eight tokens of certified slack.</desc>
  <defs>
    <style>
      .label {{ font-family: "Times New Roman", Times, serif; font-size: 21px; fill: #111111; }}
      .axis {{ font-family: "Times New Roman", Times, serif; font-size: 18px; fill: #333333; }}
      .note {{ font-family: "Times New Roman", Times, serif; font-size: 19px; font-weight: 700; fill: #087a24; }}
      .grid {{ stroke: #d4d4d4; stroke-width: 1; }}
      .axis-line {{ stroke: #111111; stroke-width: 1.5; }}
    </style>
  </defs>

  <g transform="translate(100 35)">
    <line x1="0" y1="0" x2="34" y2="0" stroke="#777777" stroke-width="2.2" stroke-dasharray="8 6"/>
    <text x="44" y="7" class="label">remaining budget r</text>
    <line x1="285" y1="0" x2="319" y2="0" stroke="#111111" stroke-width="3"/>
    <text x="329" y="7" class="label">shortest replayed witness d</text>
  </g>

  <line x1="{plot_x0}" y1="{plot_y0}" x2="{plot_x0}" y2="{plot_y0 + plot_h}" class="axis-line"/>
  <line x1="{plot_x0}" y1="{plot_y0 + plot_h}" x2="{plot_x0 + plot_w}" y2="{plot_y0 + plot_h}" class="axis-line"/>
  {grid}
  {x_axis}
  <polyline points="{_polyline(remaining_points)}" fill="none" stroke="#777777" stroke-width="2.2" stroke-dasharray="8 6"/>
  <polyline points="{_polyline(witness_points)}" fill="none" stroke="#111111" stroke-width="3"/>

  <line x1="{divergence_x:.2f}" y1="{plot_y0}" x2="{divergence_x:.2f}" y2="{plot_y0 + plot_h}" stroke="#087a24" stroke-width="1.7" stroke-dasharray="4 4"/>
  <line x1="{divergence_x:.2f}" y1="{divergence_remaining_y:.2f}" x2="{divergence_x:.2f}" y2="{divergence_witness_y:.2f}" stroke="#087a24" stroke-width="5"/>
  <circle cx="{divergence_x:.2f}" cy="{divergence_witness_y:.2f}" r="5" fill="white" stroke="#087a24" stroke-width="2.5"/>
  <text x="{divergence_x + 12:.2f}" y="{divergence_witness_y - 16:.2f}" class="note">step {divergence["step"]}: d = {divergence["bounded_witness_tokens"]}, slack = {divergence["bounded_slack_tokens"]}</text>

  <text x="{plot_x0 + plot_w / 2}" y="{plot_y0 + plot_h + 66}" text-anchor="middle" class="label">decoder step (native token)</text>
  <text x="25" y="{plot_y0 + plot_h / 2}" text-anchor="middle" class="label" transform="rotate(-90 25 {plot_y0 + plot_h / 2})">tokens</text>
</svg>
'''


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--record-id", default=DEFAULT_RECORD_ID)
    parser.add_argument("--output-svg", type=Path, default=DEFAULT_SVG)
    parser.add_argument("--output-data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--tail-start", type=int, default=192)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input.resolve()
    evidence = build_evidence(
        input_path, args.record_id, max_new_tokens=args.max_new_tokens
    )
    svg = render_svg(evidence, tail_start=args.tail_start)
    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_data.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_svg.write_text(svg, encoding="utf-8")
    print(f"wrote trace: {args.output_data}")
    print(f"wrote figure: {args.output_svg}")


if __name__ == "__main__":
    main()
