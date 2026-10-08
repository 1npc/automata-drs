"""Exact finite-horizon reachability for small audit state spaces.

Production BF/WF uses a cheap constructive positive certificate because the
reachable DRS state graph under a 248k-token vocabulary is far too large to
enumerate. This module supplies the deliberately expensive reference: breadth-
first exploration over a complete finite token alphabet. It is exact only
when exploration finishes inside its explicit state/transition ceilings;
resource exhaustion returns ``UNKNOWN``, never ``UNREACHABLE``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Hashable, TypeVar


StateT = TypeVar("StateT", bound=Hashable)


class ReachabilityStatus(str, Enum):
    """Three-valued result for a resource-bounded exact search."""

    REACHABLE = "reachable"
    UNREACHABLE = "unreachable"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class BoundedReachabilityResult(Generic[StateT]):
    status: ReachabilityStatus
    witness: tuple[int, ...] | None
    accepting_state: StateT | None
    explored_states: int
    tested_transitions: int
    depth_reached: int
    cutoff_reason: str | None = None

    @property
    def distance(self) -> int | None:
        return None if self.witness is None else len(self.witness)


def exact_bounded_reachability(
    initial_state: StateT,
    *,
    max_steps: int,
    token_ids: Callable[[StateT], Iterable[int]],
    transition: Callable[[StateT, int], StateT | None],
    is_accepting: Callable[[StateT], bool],
    max_states: int = 100_000,
    max_transitions: int = 1_000_000,
) -> BoundedReachabilityResult[StateT]:
    """Return a shortest accepting witness or an honest three-valued result.

    ``token_ids`` must enumerate the complete finite token alphabet applicable
    to each state. ``transition`` must be deterministic, returning ``None`` for
    a dead successor. Under those assumptions, an exhausted BFS proves that no
    accepting sequence of length at most ``max_steps`` exists.

    The state space itself need not be globally finite: a finite vocabulary and
    finite horizon induce a finite unfolding. In practice that unfolding can be
    exponential, so ``max_states`` and ``max_transitions`` are mandatory safety
    ceilings. Hitting either ceiling yields ``UNKNOWN``.
    """

    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if max_states <= 0:
        raise ValueError("max_states must be positive")
    if max_transitions < 0:
        raise ValueError("max_transitions must be non-negative")
    if is_accepting(initial_state):
        return BoundedReachabilityResult(
            status=ReachabilityStatus.REACHABLE,
            witness=(),
            accepting_state=initial_state,
            explored_states=1,
            tested_transitions=0,
            depth_reached=0,
        )

    visited = {initial_state}
    frontier: dict[StateT, tuple[int, ...]] = {initial_state: ()}
    tested = 0
    depth_reached = 0
    for depth in range(max_steps):
        next_frontier: dict[StateT, tuple[int, ...]] = {}
        for state, prefix in frontier.items():
            candidates = sorted({int(token_id) for token_id in token_ids(state)})
            for token_id in candidates:
                if tested >= max_transitions:
                    return BoundedReachabilityResult(
                        status=ReachabilityStatus.UNKNOWN,
                        witness=None,
                        accepting_state=None,
                        explored_states=len(visited),
                        tested_transitions=tested,
                        depth_reached=depth_reached,
                        cutoff_reason="max-transitions",
                    )
                tested += 1
                successor = transition(state, token_id)
                if successor is None:
                    continue
                witness = prefix + (token_id,)
                if is_accepting(successor):
                    return BoundedReachabilityResult(
                        status=ReachabilityStatus.REACHABLE,
                        witness=witness,
                        accepting_state=successor,
                        explored_states=(
                            len(visited) + (0 if successor in visited else 1)
                        ),
                        tested_transitions=tested,
                        depth_reached=depth + 1,
                    )
                if successor in visited:
                    continue
                if len(visited) >= max_states:
                    return BoundedReachabilityResult(
                        status=ReachabilityStatus.UNKNOWN,
                        witness=None,
                        accepting_state=None,
                        explored_states=len(visited),
                        tested_transitions=tested,
                        depth_reached=depth_reached,
                        cutoff_reason="max-states",
                    )
                visited.add(successor)
                next_frontier[successor] = witness
        depth_reached = depth + 1
        if not next_frontier:
            return BoundedReachabilityResult(
                status=ReachabilityStatus.UNREACHABLE,
                witness=None,
                accepting_state=None,
                explored_states=len(visited),
                tested_transitions=tested,
                depth_reached=depth_reached,
            )
        frontier = next_frontier

    return BoundedReachabilityResult(
        status=ReachabilityStatus.UNREACHABLE,
        witness=None,
        accepting_state=None,
        explored_states=len(visited),
        tested_transitions=tested,
        depth_reached=depth_reached,
    )


__all__ = [
    "BoundedReachabilityResult",
    "ReachabilityStatus",
    "exact_bounded_reachability",
]
