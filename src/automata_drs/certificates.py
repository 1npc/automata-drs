"""Representation-neutral accepting certificates for constrained decoding.

The production completion planners in this project are constructive: finding
one replayable suffix proves bounded reachability, while failure to construct
one proves nothing.  This module makes that asymmetry explicit and provides
the small trusted operation needed by witness-carrying decoding: verify a
native-token trace once, then consume its head without losing ownership of the
verified residual path.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Hashable, TypeVar

from .reachability import BoundedReachabilityResult, ReachabilityStatus


StateT = TypeVar("StateT", bound=Hashable)
TokenTransition = Callable[[StateT, int], StateT | None]
AcceptingPredicate = Callable[[StateT], bool]
CertificateProposer = Callable[[StateT, int], Sequence[int] | None]


@dataclass(frozen=True, slots=True)
class VerifiedAcceptingCertificate(Generic[StateT]):
    """A verifier-produced native-token path to an accepting state.

    Instances returned by :func:`verify_accepting_certificate` have been
    replayed from ``start_state`` under a deterministic transition function.
    The final state is accepting and ``token_ids`` fits the checked budget.
    """

    start_state: StateT
    token_ids: tuple[int, ...]
    final_state: StateT

    @property
    def distance(self) -> int:
        """Number of remaining native model-token steps, including EOS."""

        return len(self.token_ids)

    @property
    def head(self) -> int | None:
        return None if not self.token_ids else self.token_ids[0]

    def consume_head(
        self,
        token_id: int,
        *,
        transition: TokenTransition[StateT],
        observed_successor: StateT | None = None,
    ) -> "VerifiedAcceptingCertificate[StateT]":
        """Return the verified residual after consuming the expected head.

        Replaying the one consumed edge prevents a certificate from being
        transferred to a different prefix.  ``observed_successor`` lets a
        decoder additionally bind the residual to its independently
        reconstructed prefix state, which is essential after beam reordering.
        """

        if not self.token_ids:
            raise ValueError("an empty certificate has no head to consume")
        token_id = int(token_id)
        if token_id != self.token_ids[0]:
            raise ValueError("token does not match the certificate head")
        successor = transition(self.start_state, token_id)
        if successor is None:
            raise ValueError("certificate head no longer has a valid transition")
        if observed_successor is not None and successor != observed_successor:
            raise ValueError("certificate residual does not own this prefix state")
        if len(self.token_ids) == 1 and successor != self.final_state:
            raise ValueError("certificate final state changed during consumption")
        return VerifiedAcceptingCertificate(
            start_state=successor,
            token_ids=self.token_ids[1:],
            final_state=self.final_state,
        )


@dataclass(frozen=True, slots=True)
class CertificateVerificationResult(Generic[StateT]):
    accepted: bool
    certificate: VerifiedAcceptingCertificate[StateT] | None
    failure_reason: str | None = None
    failure_index: int | None = None

    def __post_init__(self) -> None:
        if self.accepted != (self.certificate is not None):
            raise ValueError("accepted iff a verified certificate is present")
        if self.accepted and (
            self.failure_reason is not None or self.failure_index is not None
        ):
            raise ValueError("an accepted certificate cannot carry a failure")
        if not self.accepted and self.failure_reason is None:
            raise ValueError("a rejected certificate needs a failure reason")


@dataclass(frozen=True, slots=True)
class CertificateSearchResult(Generic[StateT]):
    """Honest three-valued search result with optional positive evidence."""

    status: ReachabilityStatus
    certificate: VerifiedAcceptingCertificate[StateT] | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is ReachabilityStatus.REACHABLE:
            if self.certificate is None:
                raise ValueError("REACHABLE requires a verified certificate")
            if self.reason is not None:
                raise ValueError("REACHABLE cannot carry a failure reason")
        elif self.certificate is not None:
            raise ValueError("only REACHABLE may carry a certificate")
        elif self.reason is None:
            raise ValueError("UNKNOWN/UNREACHABLE require an explanation")

    @property
    def distance(self) -> int | None:
        return None if self.certificate is None else self.certificate.distance


class PrefixCertificateStore(Generic[StateT]):
    """Bounded certificate ownership keyed by generated token prefixes.

    Beam row indices are deliberately absent from this API: they are mutable
    scheduling positions, not hypothesis identities. Equal prefixes share the
    same deterministic residual certificate, while different prefixes cannot
    accidentally inherit one another's evidence after reorder or pruning.
    """

    def __init__(self, max_entries: int = 20_000) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._entries: OrderedDict[
            tuple[int, ...], VerifiedAcceptingCertificate[StateT]
        ] = OrderedDict()

    @staticmethod
    def _prefix(token_ids: Sequence[int]) -> tuple[int, ...]:
        return tuple(int(token_id) for token_id in token_ids)

    def bind(
        self,
        token_ids: Sequence[int],
        state: StateT,
        certificate: VerifiedAcceptingCertificate[StateT],
    ) -> None:
        """Bind evidence only to the prefix state from which it was verified."""

        if certificate.start_state != state:
            raise ValueError("certificate does not start at this prefix state")
        prefix = self._prefix(token_ids)
        self._entries.pop(prefix, None)
        if len(self._entries) >= self.max_entries:
            self._entries.popitem(last=False)
        self._entries[prefix] = certificate

    def get(
        self, token_ids: Sequence[int]
    ) -> VerifiedAcceptingCertificate[StateT] | None:
        prefix = self._prefix(token_ids)
        certificate = self._entries.get(prefix)
        if certificate is not None:
            self._entries.move_to_end(prefix)
        return certificate

    def carry_if_head(
        self,
        parent_token_ids: Sequence[int],
        token_id: int,
        *,
        transition: TokenTransition[StateT],
        observed_successor: StateT,
    ) -> VerifiedAcceptingCertificate[StateT] | None:
        """Carry the residual to ``parent + token`` when token is its head.

        A non-head token is a safe miss rather than an error: the decoder may
        attach a newly verified certificate for that successor. Invalid state
        ownership still fails closed with ``ValueError``.
        """

        parent = self._prefix(parent_token_ids)
        certificate = self.get(parent)
        token_id = int(token_id)
        if certificate is None or certificate.head != token_id:
            return None
        residual = certificate.consume_head(
            token_id,
            transition=transition,
            observed_successor=observed_successor,
        )
        self.bind(parent + (token_id,), observed_successor, residual)
        return residual

    def retain(self, prefixes: Iterable[Sequence[int]]) -> None:
        """Drop evidence for hypotheses no longer retained by a beam."""

        retained = {self._prefix(prefix) for prefix in prefixes}
        for prefix in tuple(self._entries):
            if prefix not in retained:
                del self._entries[prefix]

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


class WitnessDecisionKind(str, Enum):
    """How WCD justified or rejected one proposed native token."""

    CARRIED = "carried"
    CACHED = "cached"
    REPLANNED = "replanned"
    INVALID_TRANSITION = "invalid-transition"
    NO_CERTIFICATE = "no-certificate"


@dataclass(frozen=True, slots=True)
class WitnessTokenDecision(Generic[StateT]):
    kind: WitnessDecisionKind
    successor: StateT | None
    certificate: VerifiedAcceptingCertificate[StateT] | None
    reason: str | None = None

    def __post_init__(self) -> None:
        allowed = self.kind in {
            WitnessDecisionKind.CARRIED,
            WitnessDecisionKind.CACHED,
            WitnessDecisionKind.REPLANNED,
        }
        if allowed != (self.successor is not None and self.certificate is not None):
            raise ValueError(
                "allowed decisions require both successor and certificate"
            )
        if allowed and self.reason is not None:
            raise ValueError("allowed decisions cannot carry a failure reason")
        if not allowed and self.reason is None:
            raise ValueError("rejected decisions require a reason")

    @property
    def allowed(self) -> bool:
        return self.certificate is not None


@dataclass(frozen=True, slots=True)
class WitnessCarryingProcessorStats:
    """Representation-neutral counters shared by WCD logits adapters."""

    calls: int
    rows: int
    seconds: float
    candidates_tested: int
    candidates_structurally_valid: int
    allowed_total: int
    allowed_min: int | None
    allowed_max: int | None
    all_masked_rows: int
    prefix_rows_certified: int
    prefix_rows_unavailable: int
    carried_tokens: int
    cached_tokens: int
    replanned_tokens: int
    rejected_without_certificate: int


CertificatePlanner = Callable[[StateT, int], CertificateSearchResult[StateT]]


class WitnessCarryingController(Generic[StateT]):
    """Representation-neutral certificate preservation across decoding.

    The proposer may request any token. WCD admits it only if the token either
    consumes the head of the prefix-owned certificate or its successor gets a
    newly verified certificate. Therefore a certified non-accepting prefix
    always retains at least its certificate head as an admissible way out.
    """

    def __init__(
        self,
        *,
        transition: TokenTransition[StateT],
        plan: CertificatePlanner[StateT],
        max_prefix_certificates: int = 20_000,
    ) -> None:
        self.transition = transition
        self.plan = plan
        self.certificates = PrefixCertificateStore[StateT](
            max_entries=max_prefix_certificates
        )

    def ensure_prefix(
        self,
        token_ids: Sequence[int],
        state: StateT,
        remaining: int,
    ) -> CertificateSearchResult[StateT]:
        """Return and bind a valid certificate for this exact prefix."""

        if remaining < 0:
            raise ValueError("remaining token budget must be non-negative")
        cached = self.certificates.get(token_ids)
        if cached is not None:
            if cached.start_state != state:
                raise ValueError("cached certificate has the wrong prefix state")
            if cached.distance > remaining:
                return CertificateSearchResult(
                    status=ReachabilityStatus.UNKNOWN,
                    certificate=None,
                    reason="cached-certificate-exceeds-remaining-budget",
                )
            return CertificateSearchResult(
                status=ReachabilityStatus.REACHABLE,
                certificate=cached,
            )

        result = self.plan(state, remaining)
        if result.certificate is not None:
            if result.certificate.start_state != state:
                raise ValueError("planner certificate has the wrong start state")
            if result.certificate.distance > remaining:
                raise ValueError("planner certificate exceeds its requested budget")
            self.certificates.bind(token_ids, state, result.certificate)
        return result

    def certify_token(
        self,
        prefix_token_ids: Sequence[int],
        state: StateT,
        token_id: int,
        *,
        remaining_after_token: int,
    ) -> WitnessTokenDecision[StateT]:
        """Certify a proposed token by carrying or replacing its witness."""

        if remaining_after_token < 0:
            return WitnessTokenDecision(
                kind=WitnessDecisionKind.NO_CERTIFICATE,
                successor=None,
                certificate=None,
                reason="generation-horizon-exhausted",
            )
        prefix = tuple(int(value) for value in prefix_token_ids)
        current = self.ensure_prefix(
            prefix, state, remaining_after_token + 1
        )
        if current.certificate is None:
            return WitnessTokenDecision(
                kind=WitnessDecisionKind.NO_CERTIFICATE,
                successor=None,
                certificate=None,
                reason=current.reason or current.status.value,
            )

        token_id = int(token_id)
        successor = self.transition(state, token_id)
        if successor is None:
            return WitnessTokenDecision(
                kind=WitnessDecisionKind.INVALID_TRANSITION,
                successor=None,
                certificate=None,
                reason="invalid-structure-transition",
            )
        child = prefix + (token_id,)
        if current.certificate.head == token_id:
            residual = self.certificates.carry_if_head(
                prefix,
                token_id,
                transition=self.transition,
                observed_successor=successor,
            )
            if residual is None:
                raise AssertionError("certificate head was not carried")
            if residual.distance > remaining_after_token:
                raise AssertionError("carried residual exceeds remaining budget")
            return WitnessTokenDecision(
                kind=WitnessDecisionKind.CARRIED,
                successor=successor,
                certificate=residual,
            )

        was_cached = self.certificates.get(child) is not None
        replacement = self.ensure_prefix(
            child, successor, remaining_after_token
        )
        if replacement.certificate is None:
            return WitnessTokenDecision(
                kind=WitnessDecisionKind.NO_CERTIFICATE,
                successor=None,
                certificate=None,
                reason=replacement.reason or replacement.status.value,
            )
        return WitnessTokenDecision(
            kind=(
                WitnessDecisionKind.CACHED
                if was_cached
                else WitnessDecisionKind.REPLANNED
            ),
            successor=successor,
            certificate=replacement.certificate,
        )


def verify_accepting_certificate(
    initial_state: StateT,
    token_ids: Sequence[int],
    *,
    transition: TokenTransition[StateT],
    is_accepting: AcceptingPredicate[StateT],
    max_steps: int,
    expected_final_state: StateT | None = None,
) -> CertificateVerificationResult[StateT]:
    """Replay a candidate certificate and check budget and accepting EOS."""

    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    native_tokens = tuple(int(token_id) for token_id in token_ids)
    if len(native_tokens) > max_steps:
        return CertificateVerificationResult(
            accepted=False,
            certificate=None,
            failure_reason="budget-exceeded",
            failure_index=max_steps,
        )

    cursor = initial_state
    for index, token_id in enumerate(native_tokens):
        successor = transition(cursor, token_id)
        if successor is None:
            return CertificateVerificationResult(
                accepted=False,
                certificate=None,
                failure_reason="invalid-transition",
                failure_index=index,
            )
        cursor = successor

    if expected_final_state is not None and cursor != expected_final_state:
        return CertificateVerificationResult(
            accepted=False,
            certificate=None,
            failure_reason="final-state-mismatch",
            failure_index=len(native_tokens),
        )
    if not is_accepting(cursor):
        return CertificateVerificationResult(
            accepted=False,
            certificate=None,
            failure_reason="non-accepting-final-state",
            failure_index=len(native_tokens),
        )
    return CertificateVerificationResult(
        accepted=True,
        certificate=VerifiedAcceptingCertificate(
            start_state=initial_state,
            token_ids=native_tokens,
            final_state=cursor,
        ),
    )


def certify_constructive_plan(
    initial_state: StateT,
    *,
    max_steps: int,
    propose: CertificateProposer[StateT],
    transition: TokenTransition[StateT],
    is_accepting: AcceptingPredicate[StateT],
) -> CertificateSearchResult[StateT]:
    """Verify a constructive plan; planner failure remains ``UNKNOWN``."""

    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    candidate = propose(initial_state, max_steps)
    if candidate is None:
        return CertificateSearchResult(
            status=ReachabilityStatus.UNKNOWN,
            certificate=None,
            reason="constructive-planner-found-no-candidate",
        )
    verified = verify_accepting_certificate(
        initial_state,
        candidate,
        transition=transition,
        is_accepting=is_accepting,
        max_steps=max_steps,
    )
    if verified.certificate is None:
        return CertificateSearchResult(
            status=ReachabilityStatus.UNKNOWN,
            certificate=None,
            reason=f"invalid-constructive-candidate:{verified.failure_reason}",
        )
    return CertificateSearchResult(
        status=ReachabilityStatus.REACHABLE,
        certificate=verified.certificate,
    )


def certificate_result_from_exact_search(
    initial_state: StateT,
    result: BoundedReachabilityResult[StateT],
    *,
    max_steps: int,
    transition: TokenTransition[StateT],
    is_accepting: AcceptingPredicate[StateT],
) -> CertificateSearchResult[StateT]:
    """Re-verify an exact-search witness and preserve its tri-state result."""

    if result.status is ReachabilityStatus.REACHABLE:
        if result.witness is None or result.accepting_state is None:
            raise AssertionError("reachable exact result omitted its witness")
        verified = verify_accepting_certificate(
            initial_state,
            result.witness,
            transition=transition,
            is_accepting=is_accepting,
            max_steps=max_steps,
            expected_final_state=result.accepting_state,
        )
        if verified.certificate is None:
            raise AssertionError(
                "exact-search witness failed independent certificate replay: "
                f"{verified.failure_reason}"
            )
        return CertificateSearchResult(
            status=ReachabilityStatus.REACHABLE,
            certificate=verified.certificate,
        )
    if result.status is ReachabilityStatus.UNREACHABLE:
        return CertificateSearchResult(
            status=ReachabilityStatus.UNREACHABLE,
            certificate=None,
            reason="complete-exact-search-exhausted",
        )
    return CertificateSearchResult(
        status=ReachabilityStatus.UNKNOWN,
        certificate=None,
        reason=result.cutoff_reason or "exact-search-resource-cutoff",
    )


__all__ = [
    "AcceptingPredicate",
    "CertificateProposer",
    "CertificatePlanner",
    "CertificateSearchResult",
    "CertificateVerificationResult",
    "PrefixCertificateStore",
    "TokenTransition",
    "VerifiedAcceptingCertificate",
    "WitnessCarryingController",
    "WitnessCarryingProcessorStats",
    "WitnessDecisionKind",
    "WitnessTokenDecision",
    "certificate_result_from_exact_search",
    "certify_constructive_plan",
    "verify_accepting_certificate",
]
