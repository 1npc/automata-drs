"""Referee-aligned scope, accessibility, and connectedness checks for SBN."""

from __future__ import annotations

from dataclasses import dataclass

from .clf import sbn_graph_to_clf
from .graph import SBNGraph
from .state import SBNConstraintState


@dataclass(frozen=True, slots=True)
class SBNWellFormednessResult:
    valid: bool
    reason: str | None
    family: str | None
    box_aliases: tuple[int, ...]
    direct_subordination: frozenset[tuple[int, int]]
    accessibility_subordination: frozenset[tuple[int, int]]
    subordination: frozenset[tuple[int, int]]
    disconnected_boxes: frozenset[int] = frozenset()


def _transitive_closure(
    relation: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    closure = set(relation)
    while True:
        before = set(closure)
        closure.update(
            (left, right)
            for left, middle in before
            for next_middle, right in before
            if middle == next_middle
        )
        if closure == before:
            return closure


def _referee_closure(
    initial: set[tuple[int, int]],
    nested_direct: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Mirror PMB Referee's transitive and connectivity closure."""
    subordination = set(initial)
    while True:
        before = set(subordination)
        transitive = _transitive_closure(before)
        closed = set(transitive)
        # If b0 directly contains b1 and another b reaches b1, Referee can
        # place b above b0 as long as b0 does not already reach b.
        for outer, inner in nested_direct:
            for candidate, reached in transitive:
                if (
                    reached == inner
                    and candidate != outer
                    and (outer, candidate) not in transitive
                ):
                    closed.add((candidate, outer))
        subordination = closed
        if subordination == before:
            return subordination


def _disconnected(
    boxes: set[int], relation: set[tuple[int, int]]
) -> set[int]:
    if len(boxes) <= 1:
        return set()
    start = next(iter(boxes))
    reached = {start}
    while True:
        before = set(reached)
        for left, right in relation:
            if left in reached:
                reached.add(right)
            if right in reached:
                reached.add(left)
        if reached == before:
            return boxes - reached


def check_sbn_graph_wellformedness(graph: SBNGraph) -> SBNWellFormednessResult:
    """Check the semantic graph with the same closure used by PMB Referee.

    The SBN parser already guarantees that every concept reference resolves.
    Since each SBN concept introduces its own discourse referent, Referee's
    remaining binding work is to infer a box-subordination edge for every
    cross-box use. Cycles and disconnected boxes are then checked after the
    same closure rules as the upstream implementation.
    """
    aliases = sbn_graph_to_clf(graph).box_aliases

    def box(index: int) -> int:
        return aliases[index]

    concept_boxes = {concept.index: box(concept.box) for concept in graph.concepts}
    boxes = set(aliases)
    nested_direct: set[tuple[int, int]] = set()
    discourse_direct: set[tuple[int, int]] = set()

    for edge in graph.proposition_edges:
        nested_direct.add((concept_boxes[edge.source], box(edge.target_box)))

    for edge in graph.box_edges:
        if edge.label == "CONJUNCTION":
            continue
        pair = (box(edge.source), box(edge.target))
        if edge.label in {"NEGATION", "POSSIBILITY", "NECESSITY"}:
            nested_direct.add(pair)
        else:
            # Upstream Referee assumes the first discourse box subordinates
            # the second one for every floating discourse relation.
            discourse_direct.add(pair)

    accessibility: set[tuple[int, int]] = set()
    for edge in graph.concept_edges:
        use_box = concept_boxes[edge.source]
        introduction_box = concept_boxes[edge.target]
        if introduction_box != use_box:
            accessibility.add((introduction_box, use_box))

    initial = nested_direct | discourse_direct | accessibility
    subordination = _referee_closure(initial, nested_direct)
    loop = next(((left, right) for left, right in subordination if left == right), None)
    if loop is not None:
        return SBNWellFormednessResult(
            False,
            f"subordinate relation has a loop at box {loop[0]}",
            "subordination-loop",
            aliases,
            frozenset(nested_direct | discourse_direct),
            frozenset(accessibility),
            frozenset(subordination),
        )

    disconnected = _disconnected(boxes, subordination)
    if disconnected:
        return SBNWellFormednessResult(
            False,
            "boxes are disconnected: "
            + ", ".join(str(index) for index in sorted(disconnected)),
            "box-connectedness",
            aliases,
            frozenset(nested_direct | discourse_direct),
            frozenset(accessibility),
            frozenset(subordination),
            frozenset(disconnected),
        )

    return SBNWellFormednessResult(
        True,
        None,
        None,
        aliases,
        frozenset(nested_direct | discourse_direct),
        frozenset(accessibility),
        frozenset(subordination),
    )


def check_sbn_state_wellformedness(
    state: SBNConstraintState,
) -> SBNWellFormednessResult:
    return check_sbn_graph_wellformedness(SBNGraph.from_state(state))


__all__ = [
    "SBNWellFormednessResult",
    "check_sbn_graph_wellformedness",
    "check_sbn_state_wellformedness",
]
