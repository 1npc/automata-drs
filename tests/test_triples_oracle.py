from __future__ import annotations

import unittest

from automata_drs import (
    ByteLevelTokenConstraintController,
    ByteLevelVocabulary,
    CanonicalTripleConstraintState,
    validate_canonical_triples,
    validate_canonical_triples_batch,
)


def _weakly_connected(
    node_ids: tuple[str, ...],
    edges: tuple[tuple[str, str, str], ...],
) -> bool:
    reached = {node_ids[0]}
    changed = True
    while changed:
        changed = False
        for source, _, target in edges:
            if source in reached and target not in reached:
                reached.add(target)
                changed = True
            elif target in reached and source not in reached:
                reached.add(source)
                changed = True
    return reached == set(node_ids)


def _surface(
    node_ids: tuple[str, ...],
    edges: tuple[tuple[str, str, str], ...],
) -> str:
    lines = [f"NODE {identifier} Type" for identifier in node_ids]
    lines.extend(
        f"EDGE {source} {relation} {target}"
        for source, relation, target in edges
    )
    return "\n".join(lines)


class CanonicalTripleIndependentOracleTest(unittest.TestCase):
    def test_bounded_graph_space_has_no_incremental_or_batch_disagreement(self):
        accepted = 0
        for node_count in range(1, 4):
            node_ids = tuple("abc"[:node_count])
            possible_edges = tuple(
                (source, "r", target)
                for source in node_ids
                for target in node_ids
            )
            for bits in range(1 << len(possible_edges)):
                edges = tuple(
                    edge
                    for index, edge in enumerate(possible_edges)
                    if bits & (1 << index)
                )
                if not _weakly_connected(node_ids, edges):
                    continue
                accepted += 1
                surface = _surface(node_ids, edges)
                with self.subTest(nodes=node_count, bits=bits):
                    incremental = validate_canonical_triples(surface)
                    batch = validate_canonical_triples_batch(surface)
                    self.assertTrue(incremental.valid, incremental.reason)
                    self.assertTrue(batch.valid, batch.reason)
                    self.assertEqual(batch.nodes, incremental.state.nodes)
                    self.assertEqual(batch.edges, incremental.state.edges)

                    state = CanonicalTripleConstraintState.initial()
                    for char in surface:
                        state = state.advance_text(char)
                        self.assertTrue(state.is_extendable, state.dead_reason)
                    self.assertTrue(state.finish().is_accepting)

                    encoded = surface.encode("utf-8")
                    first = len(encoded) // 3
                    second = 2 * len(encoded) // 3
                    vocabulary = ByteLevelVocabulary(
                        fragments={
                            0: encoded[:first],
                            1: encoded[first:second],
                            2: encoded[second:],
                        },
                        eos_token_ids=frozenset({3}),
                        byte_token_ids={},
                    )
                    controller = ByteLevelTokenConstraintController(vocabulary)
                    lifted = controller.initial(
                        CanonicalTripleConstraintState.initial()
                    )
                    for token_id in (0, 1, 2, 3):
                        lifted = controller.transition(lifted, token_id)  # type: ignore[assignment,arg-type]
                        self.assertIsNotNone(lifted)
                    self.assertTrue(lifted.is_accepting)  # type: ignore[union-attr]

        self.assertEqual(446, accepted)

    def test_registered_corruptions_are_rejected_by_both_paths(self):
        corruptions = (
            "",
            "EDGE a r a",
            "NODE b Type\nNODE a Type",
            "NODE a Type\nNODE b Type",
            "NODE a Type\nEDGE a r z",
            "NODE a Type\nNODE b Type\nEDGE a r b\nEDGE a r b",
            "NODE a Type\nEDGE a r a\nNODE b Type",
            "NODE a Type\n",
            "NODE  a Type",
            "NODE a Type\r\n",
        )
        for surface in corruptions:
            with self.subTest(surface=surface):
                incremental = validate_canonical_triples(surface)
                batch = validate_canonical_triples_batch(surface)
                self.assertFalse(incremental.valid)
                self.assertFalse(batch.valid)


if __name__ == "__main__":
    unittest.main()
