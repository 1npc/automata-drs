"""Fail-closed release decisions for generated structured outputs.

Experiment metrics describe a population. A deployment boundary needs a
per-output decision that recomputes policy acceptance and refuses to release a
row when the finite-horizon execution evidence is missing or inconsistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .sbn import (
    SBNPolicy,
    check_sbn_state_wellformedness,
    validate_sbn,
)
from .penman import validate_penman
from .triples import (
    validate_canonical_triples,
    validate_canonical_triples_batch,
)


CERTIFIED_SBN_CONFIGURATIONS = frozenset({"B", "W", "BF", "WF"})
PENMAN_RELEASE_GUARANTEE = "final-local-and-independent-penman-membership"
CANONICAL_TRIPLE_RELEASE_GUARANTEE = (
    "final-incremental-and-independent-canonical-triple-membership"
)


@dataclass(frozen=True, slots=True)
class CertifiedReleaseDecision:
    """One policy-relative release decision with explicit failure reasons."""

    releasable: bool
    configuration: str
    guarantee: str | None
    reasons: tuple[str, ...]

    def require(self) -> None:
        """Raise rather than expose a row that failed the release boundary."""

        if not self.releasable:
            detail = "; ".join(self.reasons) or "uncertified output"
            raise CertifiedReleaseError(detail)


class CertifiedReleaseError(RuntimeError):
    """A generated output failed the fail-closed release boundary."""


@dataclass(frozen=True, slots=True)
class CertifiedSurface:
    """A raw generated surface that crossed a fail-closed release boundary.

    Applications that require policy-valid structured input should accept this
    type, rather than reading ``raw_prediction`` directly from generation rows.
    The certificate is deliberately policy-relative: it carries no claim that
    the surface expresses the correct meaning of the source sentence.
    """

    text: str
    configuration: str
    guarantee: str


@dataclass(frozen=True, slots=True)
class FinalSurfaceReleasePolicy:
    """Representation-owned inputs to the shared observed-EOS release gate.

    This contract deliberately covers final-state certification only. A
    representation with a finite-horizon completion proof, such as BF/WF,
    keeps its additional certificate checks in its own assessor.
    """

    configuration: str
    guarantee: str
    validate_surface: Callable[[str], bool]
    surface_rejection_reason: str
    validity_requirements: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.configuration or not self.guarantee:
            raise ValueError("release configuration and guarantee must be non-empty")
        if not self.surface_rejection_reason:
            raise ValueError("surface rejection reason must be non-empty")
        if not callable(self.validate_surface):
            raise TypeError("validate_surface must be callable")
        if any(
            not field or not reason
            for field, reason in self.validity_requirements
        ):
            raise ValueError(
                "validity requirements must contain non-empty strings"
            )


def _nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def has_complete_selective_release_accounting(
    release: dict[str, Any],
    *,
    attempts: int,
    guarantee: str = PENMAN_RELEASE_GUARANTEE,
) -> bool:
    """Check selective-release accounting without requiring 100% coverage.

    A complete report accounts for every attempt as either released or
    refused, certifies every released row under exactly ``guarantee``, and
    preserves per-row refusal details.  This is an evidence-integrity check;
    release coverage remains an empirical outcome.
    """

    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
        return False
    released = _nonnegative_integer(release.get("releasable_rows"))
    refused = _nonnegative_integer(release.get("refused_rows"))
    if released is None or refused is None or released + refused != attempts:
        return False
    expected_guarantees = {guarantee: released} if released else {}
    if release.get("guarantee_counts") != expected_guarantees:
        return False
    refusals = release.get("refusals")
    reason_counts = release.get("reason_counts")
    if not isinstance(refusals, list) or len(refusals) != refused:
        return False
    if not isinstance(reason_counts, dict):
        return False
    if refused == 0 and reason_counts:
        return False
    if refused > 0 and not reason_counts:
        return False
    if any(
        not isinstance(reason, str)
        or not reason
        or _nonnegative_integer(count) is None
        or count <= 0
        for reason, count in reason_counts.items()
    ):
        return False
    seen: set[str] = set()
    for row in refusals:
        if not isinstance(row, dict):
            return False
        identity = row.get("id")
        reasons = row.get("reasons")
        if (
            not isinstance(identity, str)
            or not identity
            or identity in seen
            or not isinstance(reasons, list)
            or not reasons
            or any(not isinstance(reason, str) or not reason for reason in reasons)
        ):
            return False
        seen.add(identity)
    return True


def assess_final_state_certified_release(
    row: dict[str, Any],
    *,
    policy: FinalSurfaceReleasePolicy,
    additional_surface_reasons: tuple[str, ...] = (),
) -> CertifiedReleaseDecision:
    """Apply shared fail-closed execution checks to one final-state policy.

    The representation owns raw-surface membership and stored validity fields;
    this function owns the common observed-EOS, no-truncation, and nonempty
    online-mask evidence. It therefore transfers without pretending that KG,
    PENMAN, and SBN have the same parser or graph semantics.
    """

    configuration = str(row.get("configuration", ""))
    if configuration != policy.configuration:
        return CertifiedReleaseDecision(
            False,
            configuration,
            None,
            (f"configuration-is-not-{policy.configuration.lower()}-policy",),
        )

    reasons: list[str] = []
    raw_prediction = row.get("raw_prediction")
    if not isinstance(raw_prediction, str) or not raw_prediction:
        reasons.append("missing-raw-surface")
    else:
        try:
            valid_surface = policy.validate_surface(raw_prediction)
        except Exception:
            valid_surface = False
            reasons.append("surface-validator-raised-an-exception")
        if valid_surface is not True and not reasons:
            reasons.append(policy.surface_rejection_reason)
    reasons.extend(additional_surface_reasons)

    validity = row.get("validity")
    for field, reason in policy.validity_requirements:
        if not isinstance(validity, dict) or validity.get(field) is not True:
            reasons.append(reason)

    generation = row.get("generation")
    if not isinstance(generation, dict):
        reasons.append("missing-generation-evidence")
    else:
        if generation.get("ended_with_eos") is not True:
            reasons.append("generation-did-not-end-with-eos")
        if generation.get("hit_max_new_tokens") is not False:
            reasons.append("generation-hit-or-did-not-report-hard-horizon")

    constraint = row.get("constraint")
    if not isinstance(constraint, dict):
        reasons.append("missing-constraint-evidence")
    else:
        rows = _nonnegative_integer(constraint.get("rows"))
        calls = _nonnegative_integer(constraint.get("calls"))
        all_masked = _nonnegative_integer(constraint.get("all_masked_rows"))
        allowed_min = _nonnegative_integer(constraint.get("allowed_min"))
        if (
            rows is None
            or rows <= 0
            or calls is None
            or calls <= 0
            or rows < calls
        ):
            reasons.append("inconsistent-processor-row-counters")
        if all_masked != 0 or allowed_min is None or allowed_min <= 0:
            reasons.append("processor-exposed-an-all-masked-row")

    return CertifiedReleaseDecision(
        releasable=not reasons,
        configuration=configuration,
        guarantee=policy.guarantee if not reasons else None,
        reasons=tuple(reasons),
    )


def assess_sbn_certified_release(
    row: dict[str, Any],
    *,
    completion_budget_factor: float = 1.0,
) -> CertifiedReleaseDecision:
    """Recompute whether one B/W/BF/WF row may be released.

    This decision guarantees membership only in the implemented SBN policy.
    It does not certify that the graph is the correct meaning of the input.
    BF/WF additionally require evidence that the full-horizon certificate was
    active on every processed logits row.
    """

    configuration = str(row.get("configuration", ""))
    reasons: list[str] = []
    if configuration not in CERTIFIED_SBN_CONFIGURATIONS:
        reasons.append("configuration-is-not-a-certified-sbn-policy")
        return CertifiedReleaseDecision(False, configuration, None, tuple(reasons))

    raw_prediction = row.get("raw_prediction")
    if not isinstance(raw_prediction, str) or not raw_prediction:
        reasons.append("missing-raw-surface")
    else:
        base_configuration = "B" if configuration in {"B", "BF"} else "W"
        validation = validate_sbn(
            raw_prediction, SBNPolicy.for_configuration(base_configuration)
        )
        if not validation.valid:
            reasons.append("raw-surface-rejected-by-policy")
        elif configuration in {"W", "WF"}:
            semantic = check_sbn_state_wellformedness(validation.state)
            if not semantic.valid:
                reasons.append("raw-surface-rejected-by-wellformedness")

    validity = row.get("validity")
    if not isinstance(validity, dict) or validity.get("raw_sbn") is not True:
        reasons.append("stored-raw-validity-is-not-true")
    if configuration in {"W", "WF"} and (
        not isinstance(validity, dict)
        or validity.get("raw_clf_referee") is not True
    ):
        reasons.append("stored-raw-referee-validity-is-not-true")

    generation = row.get("generation")
    if not isinstance(generation, dict):
        reasons.append("missing-generation-evidence")
    else:
        if generation.get("ended_with_eos") is not True:
            reasons.append("generation-did-not-end-with-eos")
        if generation.get("hit_max_new_tokens") is not False:
            reasons.append("generation-hit-or-did-not-report-hard-horizon")

    constraint = row.get("constraint")
    if not isinstance(constraint, dict):
        reasons.append("missing-constraint-evidence")
    else:
        rows = _nonnegative_integer(constraint.get("rows"))
        calls = _nonnegative_integer(constraint.get("calls"))
        all_masked = _nonnegative_integer(constraint.get("all_masked_rows"))
        allowed_min = _nonnegative_integer(constraint.get("allowed_min"))
        if rows is None or rows <= 0 or calls != rows:
            reasons.append("inconsistent-processor-row-counters")
        if all_masked != 0 or allowed_min is None or allowed_min <= 0:
            reasons.append("processor-exposed-an-all-masked-row")

        if configuration in {"BF", "WF"}:
            bounded_rows = _nonnegative_integer(constraint.get("bounded_rows"))
            allowed_total = _nonnegative_integer(constraint.get("allowed_total"))
            certificates_tested = _nonnegative_integer(
                constraint.get("completion_certificates_tested")
            )
            certificates_accepted = _nonnegative_integer(
                constraint.get("completion_certificates_accepted")
            )
            if completion_budget_factor != 1.0:
                reasons.append("completion-budget-factor-is-not-full-horizon")
            if rows is None or bounded_rows != rows:
                reasons.append("certificate-was-not-active-on-every-row")
            if (
                allowed_total is None
                or certificates_tested is None
                or certificates_accepted is None
                or allowed_total < (rows or 0)
                or certificates_tested < certificates_accepted
                or certificates_accepted != allowed_total
            ):
                reasons.append("inconsistent-completion-certificate-counters")

    guarantee = {
        "B": "final-structural-policy-membership",
        "W": "final-structural-and-referee-policy-membership",
        "BF": "finite-horizon-structural-policy-membership",
        "WF": "finite-horizon-structural-and-referee-policy-membership",
    }[configuration]
    return CertifiedReleaseDecision(
        releasable=not reasons,
        configuration=configuration,
        guarantee=guarantee if not reasons else None,
        reasons=tuple(reasons),
    )


def assess_penman_certified_release(
    row: dict[str, Any],
    *,
    independent_penman_valid: bool | None,
) -> CertifiedReleaseDecision:
    """Recompute whether one online-PENMAN row may be released.

    The incremental P controller is prefix-sound but does not yet reserve a
    full-horizon accepting completion. Release therefore requires observed EOS
    without truncation, a fresh local parse, and a separately recomputed
    independent ``penman``-package decision. If any evidence is unavailable,
    the interface abstains rather than exposing an unchecked graph.
    """

    configuration = str(row.get("configuration", ""))
    if configuration != "P":
        return CertifiedReleaseDecision(
            False,
            configuration,
            None,
            ("configuration-is-not-online-penman",),
        )
    policy = FinalSurfaceReleasePolicy(
        configuration="P",
        guarantee=PENMAN_RELEASE_GUARANTEE,
        validate_surface=lambda text: validate_penman(text).valid,
        surface_rejection_reason="raw-surface-rejected-by-local-penman-policy",
        validity_requirements=(
            ("raw_local_penman", "stored-local-penman-validity-is-not-true"),
            (
                "raw_official_penman",
                "stored-independent-penman-validity-is-not-true",
            ),
        ),
    )
    extra = (
        ()
        if independent_penman_valid is True
        else ("raw-surface-rejected-or-not-checked-by-independent-penman",)
    )
    return assess_final_state_certified_release(
        row,
        policy=policy,
        additional_surface_reasons=extra,
    )


def assess_canonical_triple_certified_release(
    row: dict[str, Any],
) -> CertifiedReleaseDecision:
    """Recompute final canonical-KG membership and shared execution evidence."""

    policy = FinalSurfaceReleasePolicy(
        configuration="KG",
        guarantee=CANONICAL_TRIPLE_RELEASE_GUARANTEE,
        validate_surface=lambda text: (
            validate_canonical_triples(text).valid
            and validate_canonical_triples_batch(text).valid
        ),
        surface_rejection_reason="raw-surface-rejected-by-canonical-triple-policy",
        validity_requirements=(
            (
                "raw_canonical_triples",
                "stored-canonical-triple-validity-is-not-true",
            ),
        ),
    )
    return assess_final_state_certified_release(row, policy=policy)


def release_sbn_surface(
    row: dict[str, Any],
    *,
    completion_budget_factor: float = 1.0,
) -> CertifiedSurface:
    """Return the raw SBN surface only after fresh fail-closed certification.

    A refusal raises :class:`CertifiedReleaseError`; this function never
    returns an unchecked or known-invalid surface.
    """

    decision = assess_sbn_certified_release(
        row, completion_budget_factor=completion_budget_factor
    )
    decision.require()
    surface = row.get("raw_prediction")
    assert isinstance(surface, str) and surface
    assert decision.guarantee is not None
    return CertifiedSurface(surface, decision.configuration, decision.guarantee)


def release_penman_surface(
    row: dict[str, Any],
    *,
    independent_penman_valid: bool | None,
) -> CertifiedSurface:
    """Return the raw PENMAN surface only after both release checks pass."""

    decision = assess_penman_certified_release(
        row, independent_penman_valid=independent_penman_valid
    )
    decision.require()
    surface = row.get("raw_prediction")
    assert isinstance(surface, str) and surface
    assert decision.guarantee is not None
    return CertifiedSurface(surface, decision.configuration, decision.guarantee)


def release_canonical_triple_surface(row: dict[str, Any]) -> CertifiedSurface:
    """Return canonical KG text only after fresh fail-closed certification."""

    decision = assess_canonical_triple_certified_release(row)
    decision.require()
    surface = row.get("raw_prediction")
    assert isinstance(surface, str) and surface
    assert decision.guarantee is not None
    return CertifiedSurface(surface, decision.configuration, decision.guarantee)


__all__ = [
    "CANONICAL_TRIPLE_RELEASE_GUARANTEE",
    "CERTIFIED_SBN_CONFIGURATIONS",
    "PENMAN_RELEASE_GUARANTEE",
    "CertifiedReleaseDecision",
    "CertifiedReleaseError",
    "CertifiedSurface",
    "FinalSurfaceReleasePolicy",
    "assess_canonical_triple_certified_release",
    "assess_final_state_certified_release",
    "assess_penman_certified_release",
    "assess_sbn_certified_release",
    "has_complete_selective_release_accounting",
    "release_penman_surface",
    "release_canonical_triple_surface",
    "release_sbn_surface",
]
