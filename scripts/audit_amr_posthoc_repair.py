#!/usr/bin/env python3
"""Evaluate deterministic PENMAN repair under the original token horizon.

This is the reviewer-facing ``P + repair`` baseline.  It receives the frozen
raw output of prefix-constrained decoding, keeps the longest native-token
prefix that can still be completed, appends the repository's deterministic
PENMAN completion, and counts EOS inside the same generation horizon.  It
never queries the language model and never borrows extra output tokens.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import (  # noqa: E402
    ByteLevelFiniteHorizonPlanner,
    ByteLevelTokenConstraintController,
    ByteLevelVocabulary,
    PENMANConstraintState,
    construct_penman_completion_suffix,
    validate_penman,
)
from automata_drs.amr_scoring import (  # noqa: E402
    aggregate_amr_scores,
    create_amr_smatchpp,
    score_amr_pair,
    smatchpp_scoreable,
)
from automata_drs.generation_manifest import (  # noqa: E402
    complete_generation_manifest,
    sha256_file,
)


@dataclass(frozen=True, slots=True)
class BudgetMatchedRepair:
    """One maximal-prefix deterministic repair."""

    surface: str
    kept_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    appended_surface: str
    original_token_count: int
    horizon: int

    @property
    def dropped_tokens(self) -> int:
        return self.original_token_count - len(self.kept_token_ids)

    @property
    def output_tokens_including_eos(self) -> int:
        return len(self.kept_token_ids) + len(self.completion_token_ids)


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("tokenizer returned a non-singleton batch")
        encoded = encoded[0]
    return tuple(int(value) for value in encoded)


def _surface_bytes(
    vocabulary: ByteLevelVocabulary, token_ids: tuple[int, ...]
) -> bytes | None:
    fragments = [vocabulary.fragments.get(token_id) for token_id in token_ids]
    if any(fragment is None for fragment in fragments):
        return None
    return b"".join(fragment for fragment in fragments if fragment is not None)


def repair_within_token_budget(
    raw_prediction: str,
    *,
    tokenizer: Any,
    vocabulary: ByteLevelVocabulary,
    horizon: int,
) -> BudgetMatchedRepair | None:
    """Keep the longest canonical token prefix repairable within ``horizon``.

    The returned completion includes EOS in ``completion_token_ids`` but not
    in ``surface``.  Searching from the longest prefix downwards makes the
    baseline deterministic and maximally conservative: it removes no model
    token unless doing so is necessary to fit an accepting completion.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    original_ids = _token_ids(tokenizer, raw_prediction)
    recovered = _surface_bytes(vocabulary, original_ids)
    if recovered != raw_prediction.encode("utf-8"):
        raise ValueError("raw prediction is not canonically tokenized")

    controller = ByteLevelTokenConstraintController(
        vocabulary,
        incomplete_utf8_is_extendable=(
            PENMANConstraintState.incomplete_utf8_is_extendable
        ),
        max_cache_entries=20_000,
    )
    planner = ByteLevelFiniteHorizonPlanner(
        controller,
        construct_surface_suffix=construct_penman_completion_suffix,
        encode_surface=lambda text: _token_ids(tokenizer, text),
    )

    states = [controller.initial(PENMANConstraintState.initial())]
    for token_id in original_ids:
        successor = controller.transition(states[-1], token_id)
        if successor is None:
            break
        states.append(successor)

    longest_live_prefix = min(len(original_ids), len(states) - 1, horizon - 1)
    for kept in range(longest_live_prefix, -1, -1):
        witness = planner.completion_within(states[kept], horizon - kept)
        if witness is None:
            continue
        prefix_bytes = _surface_bytes(vocabulary, original_ids[:kept])
        if prefix_bytes is None:
            continue
        completed_bytes = prefix_bytes + witness.appended_surface_bytes
        try:
            surface = completed_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not validate_penman(surface).valid:
            continue
        return BudgetMatchedRepair(
            surface=surface,
            kept_token_ids=original_ids[:kept],
            completion_token_ids=witness.token_ids,
            appended_surface=witness.appended_surface_bytes.decode("utf-8"),
            original_token_count=len(original_ids),
            horizon=horizon,
        )
    return None


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank row at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"row {line_number} is not an object")
            rows.append(row)
    return rows


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")


def _complete_manifest(
    predictions: Path,
    manifest_path: Path,
    *,
    corpus_override: Path | None,
) -> dict[str, Any]:
    """Validate a frozen manifest, optionally remapping its physical corpus path.

    Completed manifests may retain a container-internal path that is not mounted
    on a later scoring host.  A remap is admissible only when the replacement
    corpus has the exact source SHA-256 already sealed in the manifest.  The
    original manifest is never modified.
    """

    if corpus_override is None:
        return complete_generation_manifest(predictions, manifest_path)
    if not corpus_override.is_file():
        raise ValueError(f"corpus override is missing: {corpus_override}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"expected a JSON object: {manifest_path}")
    expected_digest = manifest.get("source_sha256")
    observed_digest = sha256_file(corpus_override)
    if expected_digest != observed_digest:
        raise ValueError(
            "corpus override SHA-256 does not match the frozen manifest: "
            f"observed={observed_digest} expected={expected_digest}"
        )
    if manifest.get("status") != "complete":
        raise ValueError("corpus override requires a completed frozen manifest")
    expected_output_digest = manifest.get("output_sha256")
    observed_output_digest = sha256_file(predictions)
    if expected_output_digest != observed_output_digest:
        raise ValueError(
            "prediction SHA-256 does not match the frozen manifest: "
            f"observed={observed_output_digest} expected={expected_output_digest}"
        )
    rows = _read_rows(predictions)
    for name in ("rows", "expected_rows"):
        expected_rows = manifest.get(name)
        if isinstance(expected_rows, bool) or not isinstance(expected_rows, int):
            raise ValueError(f"completed manifest has no integer {name}")
        if len(rows) != expected_rows:
            raise ValueError(
                f"prediction row count does not match manifest {name}: "
                f"observed={len(rows)} expected={expected_rows}"
            )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--configuration", default="P")
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--corpus-override", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-solver-seconds", type=int, default=240)
    args = parser.parse_args()
    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    if args.max_solver_seconds <= 0:
        parser.error("--max-solver-seconds must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite report: {args.output}")

    try:
        import penman
        from transformers import AutoTokenizer
    except ImportError as error:
        parser.error(f"AMR repair dependencies are unavailable: {error}")

    load_kwargs: dict[str, Any] = {"local_files_only": True}
    if args.revision:
        load_kwargs["revision"] = args.revision
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = str(args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    vocabulary = ByteLevelVocabulary.from_tokenizer(tokenizer)
    measure = create_amr_smatchpp(max_solver_seconds=args.max_solver_seconds)

    manifest_path = Path(str(args.predictions) + ".manifest.json")
    try:
        manifest = _complete_manifest(
            args.predictions,
            manifest_path,
            corpus_override=args.corpus_override,
        )
        rows = [
            row
            for row in _read_rows(args.predictions)
            if row.get("configuration") == args.configuration
        ]
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if not rows:
        parser.error(f"no {args.configuration} rows in generation matrix")
    if len(rows) != manifest.get("expected_examples"):
        parser.error("selected configuration does not cover every example")

    details: list[dict[str, Any]] = []
    pair_scores: list[dict[str, Any]] = []
    dropped: list[int] = []
    appended: list[int] = []
    official_valid = 0
    for row in rows:
        raw = str(row.get("raw_prediction", ""))
        reference = str(row.get("reference", ""))
        repair = repair_within_token_budget(
            raw,
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            horizon=args.horizon,
        )
        official = False
        scoreable = False
        if repair is not None:
            try:
                penman.decode(repair.surface)
                official = True
            except Exception:
                official = False
            scoreable = official and smatchpp_scoreable(measure, repair.surface)
        if official:
            official_valid += 1
        scored_surface = repair.surface if repair is not None and scoreable else None
        pair_score = score_amr_pair(measure, scored_surface, reference)
        pair_scores.append(pair_score)
        if repair is not None:
            dropped.append(repair.dropped_tokens)
            appended.append(len(repair.completion_token_ids))
        details.append(
            {
                "id": str(row.get("id", "")),
                "repair_constructed": repair is not None,
                "official_penman": official,
                "smatchpp_scoreable": scoreable,
                "original_tokens": (
                    None if repair is None else repair.original_token_count
                ),
                "kept_tokens": (
                    None if repair is None else len(repair.kept_token_ids)
                ),
                "dropped_tokens": (
                    None if repair is None else repair.dropped_tokens
                ),
                "completion_tokens_including_eos": (
                    None if repair is None else len(repair.completion_token_ids)
                ),
                "output_tokens_including_eos": (
                    None if repair is None else repair.output_tokens_including_eos
                ),
                "appended_surface": (
                    None if repair is None else repair.appended_surface
                ),
                "smatchpp": pair_score,
            }
        )

    aggregate = aggregate_amr_scores(pair_scores)
    constructed = sum(item["repair_constructed"] for item in details)
    scoreable = sum(item["smatchpp_scoreable"] for item in details)
    count = len(details)
    within_budget = all(
        item["output_tokens_including_eos"] is not None
        and item["output_tokens_including_eos"] <= args.horizon
        for item in details
    )
    gate_passed = (
        constructed == official_valid == scoreable == count
        and within_budget
        and aggregate["all_alignments_optimal"]
    )
    report = {
        "schema_version": 1,
        "kind": "amr-budget-matched-posthoc-repair",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "predictions": str(args.predictions),
            "predictions_sha256": sha256_file(args.predictions),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "corpus_override": (
                None if args.corpus_override is None else str(args.corpus_override)
            ),
            "corpus_sha256": (
                manifest.get("source_sha256")
                if args.corpus_override is None
                else sha256_file(args.corpus_override)
            ),
            "configuration": args.configuration,
            "records": count,
            "model": args.model,
            "revision": args.revision,
            "horizon": args.horizon,
        },
        "method": {
            "name": "P+Repair",
            "model_calls": 0,
            "selection": "longest repairable native-token prefix",
            "budget_includes_eos": True,
        },
        "summary": {
            "repair_constructed": constructed,
            "official_penman": official_valid,
            "smatchpp_scoreable": scoreable,
            "dropped_tokens_mean": statistics.fmean(dropped) if dropped else None,
            "dropped_tokens_max": max(dropped, default=None),
            "completion_tokens_mean": (
                statistics.fmean(appended) if appended else None
            ),
            "completion_tokens_max": max(appended, default=None),
            "corpus_smatchpp": aggregate,
        },
        "rows": details,
        "gate": {
            "passed": gate_passed,
            "all_rows_repaired": constructed == count,
            "all_rows_official_penman": official_valid == count,
            "all_rows_smatchpp_scoreable": scoreable == count,
            "all_rows_within_original_horizon": within_budget,
            "all_alignments_optimal": aggregate["all_alignments_optimal"],
        },
    }
    try:
        _write_exclusive(args.output, report)
    except FileExistsError:
        parser.error(f"refusing to overwrite report: {args.output}")
    print(f"report: {args.output}")
    print(
        f"repaired={constructed}/{count} valid={official_valid}/{count} "
        f"Smatch++={aggregate['f1']:.6f} gate={gate_passed}"
    )
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
