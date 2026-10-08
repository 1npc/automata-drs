"""Projection from an accepted SBN graph to PMB-style DRS clauses (CLF).

The conversion follows Bos's SBN interpretation: every concept introduces a
referent in its member context, hooks and anchors become conditions in the
source concept's context, and box connectors determine scope/discourse links.
``CONJUNCTION`` aliases its new context to the connected context because it
continues the current DRS instead of introducing a nested DRS.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from .graph import SBNGraph
from .state import SBNConstraintState


Clause = tuple[str, ...]

_UNARY_DRS_CONNECTORS = {
    "NEGATION": "NOT",
    "POSSIBILITY": "POS",
    "NECESSITY": "NEC",
}


@dataclass(frozen=True, slots=True)
class CLFProjection:
    clauses: tuple[Clause, ...]
    # One canonical CLF box index per original SBN context index.
    box_aliases: tuple[int, ...]

    def render(self) -> str:
        return "\n".join(" ".join(clause) for clause in self.clauses)


@dataclass(frozen=True, slots=True)
class PMBRefereeResult:
    valid: bool
    reason: str | None
    projection: CLFProjection
    subordination: frozenset[tuple[str, str]] = frozenset()
    direct_subordination: frozenset[tuple[str, str]] = frozenset()


class PMBRefereeOracle:
    """Reusable adapter around one loaded upstream PMB Referee module."""

    def __init__(
        self,
        referee_path: str | Path,
        *,
        signature_path: str | Path | None = None,
    ) -> None:
        try:
            self._referee = _load_referee(
                referee_path,
                allow_missing_yaml=signature_path is None,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "yaml":
                raise RuntimeError(
                    "PyYAML is required when an official Referee signature is supplied"
                ) from exc
            raise
        self._signature = (
            self._referee.get_signature(str(signature_path))
            if signature_path is not None
            else {}
        )

    def check(self, projection: CLFProjection) -> PMBRefereeResult:
        try:
            _, subordination, direct_subordination, _ = self._referee.check_clf(
                list(projection.clauses), self._signature
            )
        except Exception as exc:  # upstream intentionally reports via RuntimeError
            return PMBRefereeResult(False, str(exc), projection)
        return PMBRefereeResult(
            True,
            None,
            projection,
            frozenset(subordination),
            frozenset(direct_subordination),
        )


def _constant_to_clf(token: str) -> str:
    """Return one whitespace-free, quoted CLF constant token."""
    if token.startswith('"') and token.endswith('"'):
        content = token[1:-1]
    else:
        content = token
    # PMB CLF serializes multi-token names with `~`; lower-casing is the
    # standard PMB normalization and does not affect Referee well-formedness.
    content = content.replace(" ", "~").lower()
    content = content.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{content}"'


def _split_synset(label: str) -> tuple[str, str]:
    try:
        lemma, pos, sense = label.rsplit(".", 2)
    except ValueError as exc:  # defensive: accepted states already validate this
        raise ValueError(f"invalid SBN synset label {label!r}") from exc
    return lemma, f'"{pos}.{sense}"'


def _context_aliases(graph: SBNGraph) -> tuple[int, ...]:
    parent = list(range(graph.box_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        canonical = min(left_root, right_root)
        parent[left_root] = canonical
        parent[right_root] = canonical

    for edge in graph.box_edges:
        if edge.label == "CONJUNCTION":
            union(edge.source, edge.target)
    return tuple(find(index) for index in range(graph.box_count))


def sbn_graph_to_clf(graph: SBNGraph) -> CLFProjection:
    """Translate a complete typed SBN graph to Referee-compatible clauses."""
    aliases = _context_aliases(graph)

    def box(index: int) -> str:
        return f"b{aliases[index]}"

    def ref(index: int) -> str:
        return f"x{index}"

    concept_boxes = {concept.index: concept.box for concept in graph.concepts}
    clauses: list[Clause] = []

    for concept in graph.concepts:
        concept_box = box(concept.box)
        concept_ref = ref(concept.index)
        lemma, pos_sense = _split_synset(concept.label)
        clauses.append((concept_box, "REF", concept_ref))
        clauses.append((concept_box, lemma, pos_sense, concept_ref))

    for edge in graph.concept_edges:
        clauses.append(
            (
                box(concept_boxes[edge.source]),
                edge.label,
                ref(edge.source),
                ref(edge.target),
            )
        )

    for edge in graph.constant_edges:
        clauses.append(
            (
                box(concept_boxes[edge.source]),
                edge.label,
                ref(edge.source),
                _constant_to_clf(edge.target),
            )
        )

    for edge in graph.proposition_edges:
        clauses.append(
            (
                box(concept_boxes[edge.source]),
                "Proposition",
                ref(edge.source),
                box(edge.target_box),
            )
        )

    for edge in graph.box_edges:
        if edge.label == "CONJUNCTION":
            continue
        label = _UNARY_DRS_CONNECTORS.get(edge.label, edge.label)
        clauses.append((box(edge.source), label, box(edge.target)))

    return CLFProjection(tuple(clauses), aliases)


def sbn_state_to_clf(state: SBNConstraintState) -> CLFProjection:
    return sbn_graph_to_clf(SBNGraph.from_state(state))


def _load_referee(
    referee_path: str | Path,
    *,
    allow_missing_yaml: bool,
) -> ModuleType:
    path = Path(referee_path)
    if not path.is_file():
        raise FileNotFoundError(f"PMB Referee module not found: {path}")
    module_spec = importlib.util.spec_from_file_location("automata_drs_pmb_referee", path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load PMB Referee module: {path}")
    module = importlib.util.module_from_spec(module_spec)
    yaml_stub: ModuleType | None = None
    if allow_missing_yaml and importlib.util.find_spec("yaml") is None:
        # Upstream imports PyYAML at module load time even though it is only
        # used by get_signature(). Its format/semantic checks work with the
        # built-in fallback signature, so keep that dependency optional.
        yaml_stub = ModuleType("yaml")
        sys.modules["yaml"] = yaml_stub
    try:
        module_spec.loader.exec_module(module)
    finally:
        if yaml_stub is not None:
            sys.modules.pop("yaml", None)
    return module


def check_with_pmb_referee(
    projection: CLFProjection,
    referee_path: str | Path,
    *,
    signature_path: str | Path | None = None,
) -> PMBRefereeResult:
    """Run the upstream PMB Referee against a projected CLF.

    With no signature, Referee still performs clause typing, covert
    subordination recovery, loop freedom, connectedness, and binding checks.
    Passing the official YAML additionally checks its closed operator/role
    inventory and requires the Referee's optional PyYAML dependency.
    """
    return PMBRefereeOracle(
        referee_path,
        signature_path=signature_path,
    ).check(projection)


__all__ = [
    "CLFProjection",
    "Clause",
    "PMBRefereeResult",
    "PMBRefereeOracle",
    "check_with_pmb_referee",
    "sbn_graph_to_clf",
    "sbn_state_to_clf",
]
