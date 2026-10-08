"""Lossless typed-graph projection of an accepted SBN constraint state."""

from __future__ import annotations

from dataclasses import dataclass

from .state import SBNConstraintState


@dataclass(frozen=True, slots=True, order=True)
class ConceptNode:
    index: int
    label: str
    box: int


@dataclass(frozen=True, slots=True, order=True)
class ConceptEdge:
    source: int
    label: str
    target: int


@dataclass(frozen=True, slots=True, order=True)
class ConstantEdge:
    source: int
    label: str
    target: str


@dataclass(frozen=True, slots=True, order=True)
class BoxEdge:
    source: int
    label: str
    target: int


@dataclass(frozen=True, slots=True, order=True)
class PropositionEdge:
    source: int
    target_box: int


@dataclass(frozen=True, slots=True)
class SBNGraph:
    """A tokenizer-independent graph reconstructed during SBN decoding.

    Concept indices and box indices are the absolute registers used by the
    incremental state. Relative surface references have already been resolved.
    The projection is lossless for graph semantics while deliberately omitting
    serialization details such as the original relative offsets.
    """

    box_count: int
    concepts: tuple[ConceptNode, ...]
    concept_edges: tuple[ConceptEdge, ...]
    constant_edges: tuple[ConstantEdge, ...]
    box_edges: tuple[BoxEdge, ...]
    proposition_edges: tuple[PropositionEdge, ...]

    @classmethod
    def from_state(
        cls,
        state: SBNConstraintState,
        *,
        require_accepting: bool = True,
    ) -> "SBNGraph":
        if require_accepting and not state.is_accepting:
            raise ValueError("an accepting SBN state is required for graph projection")
        if len(state.concept_labels) != len(state.concept_boxes):
            raise ValueError("concept labels and membership registers are inconsistent")
        return cls(
            box_count=state.box_count,
            concepts=tuple(
                ConceptNode(index, label, state.concept_boxes[index])
                for index, label in enumerate(state.concept_labels)
            ),
            concept_edges=tuple(
                sorted(
                    ConceptEdge(source, label, target)
                    for source, target, label in state.node_edges
                )
            ),
            constant_edges=tuple(
                ConstantEdge(source, label, target)
                for source, label, target in state.constant_edges
            ),
            box_edges=tuple(
                BoxEdge(source, label, target)
                for source, target, label in state.box_edges
            ),
            proposition_edges=tuple(
                PropositionEdge(source, target)
                for source, target in state.proposition_targets
            ),
        )


__all__ = [
    "BoxEdge",
    "ConceptEdge",
    "ConceptNode",
    "ConstantEdge",
    "PropositionEdge",
    "SBNGraph",
]
