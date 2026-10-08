"""Strict readers for sentence-to-AMR corpora stored as paired blocks.

The local AMR files are not ordinary PENMAN corpora: each graph is preceded by
its source sentence as a separate blank-line-delimited block.  Treating every
block as a graph silently doubles the record count, while guessing from a
leading ``("` is incorrect because legitimate sentences such as ``( End )``
also begin with a parenthesis.  This module therefore assigns meaning by the
documented sentence/graph position and fails closed on incomplete pairs.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path


class AMRCorpusFormatError(ValueError):
    """Raised when a paired sentence/PENMAN corpus is structurally ambiguous."""


@dataclass(frozen=True, slots=True)
class AMRCorpusRecord:
    """One source sentence paired with one multiline PENMAN graph."""

    ordinal: int
    sentence_start_line: int
    graph_start_line: int
    sentence: str
    graph: str


@dataclass(frozen=True, slots=True)
class _TextBlock:
    start_line: int
    text: str


def _iter_nonempty_blocks(lines: Iterable[str]) -> Iterator[_TextBlock]:
    block: list[str] = []
    start_line: int | None = None

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r\n")
        if line.strip():
            if start_line is None:
                start_line = line_number
            block.append(line)
            continue
        if block:
            assert start_line is not None
            yield _TextBlock(start_line=start_line, text="\n".join(block).strip())
            block = []
            start_line = None

    if block:
        assert start_line is not None
        yield _TextBlock(start_line=start_line, text="\n".join(block).strip())


def iter_amr_records(lines: Iterable[str]) -> Iterator[AMRCorpusRecord]:
    """Yield alternating sentence/PENMAN blocks and reject malformed tails.

    A graph block must begin with ``(``, as AMR instance graphs do.  No shape
    test is applied to sentence blocks: their position is authoritative, so a
    sentence such as ``( End )`` cannot be mistaken for a graph.
    """

    blocks = iter(_iter_nonempty_blocks(lines))
    ordinal = 0
    while True:
        try:
            sentence = next(blocks)
        except StopIteration:
            return
        try:
            graph = next(blocks)
        except StopIteration as error:
            raise AMRCorpusFormatError(
                "unpaired sentence block at line "
                f"{sentence.start_line}: expected a following PENMAN graph"
            ) from error

        ordinal += 1
        if not graph.text.startswith("("):
            raise AMRCorpusFormatError(
                f"record {ordinal} graph block at line {graph.start_line} "
                "does not begin with '('"
            )
        yield AMRCorpusRecord(
            ordinal=ordinal,
            sentence_start_line=sentence.start_line,
            graph_start_line=graph.start_line,
            sentence=sentence.text,
            graph=graph.text,
        )


def load_amr_records(path: Path) -> list[AMRCorpusRecord]:
    """Load a local paired-block AMR file with strict structural validation."""

    with path.open(encoding="utf-8") as source:
        return list(iter_amr_records(source))


def select_amr_records_by_ordinal(
    records: Iterable[AMRCorpusRecord], ordinals: Sequence[int]
) -> list[AMRCorpusRecord]:
    """Select unique positive record ordinals in the requested order."""

    requested = [int(ordinal) for ordinal in ordinals]
    if not requested:
        raise ValueError("ordinals must not be empty")
    if any(ordinal <= 0 for ordinal in requested):
        raise ValueError("ordinals must be positive")
    if len(set(requested)) != len(requested):
        raise ValueError("ordinals must not contain duplicates")
    wanted = set(requested)
    available: dict[int, AMRCorpusRecord] = {}
    for record in records:
        if record.ordinal in wanted:
            available[record.ordinal] = record
    missing = [ordinal for ordinal in requested if ordinal not in available]
    if missing:
        raise ValueError(
            "requested AMR record ordinals are missing: "
            + ", ".join(str(ordinal) for ordinal in missing)
        )
    return [available[ordinal] for ordinal in requested]
