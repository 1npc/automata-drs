"""Prompt and supervision contract for decoder-only AMR fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .amr_data import AMRCorpusRecord, iter_amr_records
from .causal_lm import encode_supervised_pair


DEFAULT_AMR_INSTRUCTION = (
    "Generate an Abstract Meaning Representation (AMR) graph in PENMAN format "
    "for the text."
)

VIAMR_QWEN3_INSTRUCTION = (
    "Convert the following English sentence into its Abstract Meaning "
    "Representation (AMR):"
)

AMR_PROMPT_TEMPLATES = (
    "frozen-plain",
    "viamr-qwen3",
    "viamr-qwen3-graph-prefill",
)
AMR_OUTPUT_ENVELOPES = ("none", "viamr-amr-tag")


@dataclass(frozen=True, slots=True)
class AMRTaskRecord:
    """Generation-facing AMR pair with a stable corpus-record identifier."""

    identifier: str
    source: str
    target: str


def iter_amr_task_records(
    path: Path, *, start: int = 0, limit: int = -1
) -> Iterator[AMRTaskRecord]:
    """Read paired-block AMR records using record ordinals, not line guesses."""

    if start < 0:
        raise ValueError("start must be non-negative")
    if limit == 0 or limit < -1:
        raise ValueError("limit must be -1 or positive")
    emitted = 0
    with path.open(encoding="utf-8") as source:
        for record in iter_amr_records(source):
            if record.ordinal <= start:
                continue
            yield AMRTaskRecord(
                identifier=f"{path}:record:{record.ordinal}",
                source=record.sentence,
                target=record.graph,
            )
            emitted += 1
            if limit > 0 and emitted >= limit:
                return


def build_amr_prompt(
    sentence: str, *, instruction: str = DEFAULT_AMR_INSTRUCTION
) -> str:
    """Return the frozen plain-text AMR generation prompt."""

    if not sentence.strip():
        raise ValueError("AMR source sentence must not be empty")
    return f"{instruction}\nText: {sentence}\nAMR:\n"


def build_amr_generation_prompt(sentence: str, *, template: str) -> str:
    """Build a generation prompt under an explicitly named contract.

    ``frozen-plain`` preserves the registered training/evaluation prompt used by
    this repository. ``viamr-qwen3`` reproduces the public checkpoint's sample
    prompt byte for byte, so an external-checkpoint probe does not confound the
    model comparison with an undocumented prompt change.
    """

    if not sentence.strip():
        raise ValueError("AMR source sentence must not be empty")
    if template == "frozen-plain":
        return build_amr_prompt(sentence)
    if template == "viamr-qwen3":
        return (
            "<|im_start|>user\n"
            f"{VIAMR_QWEN3_INSTRUCTION}\n\n"
            f"<sentence>{sentence}</sentence><|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if template == "viamr-qwen3-graph-prefill":
        return (
            "<|im_start|>user\n"
            f"{VIAMR_QWEN3_INSTRUCTION}\n\n"
            f"<sentence>{sentence}</sentence><|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
            "<amr>\n"
        )
    raise ValueError(f"unsupported AMR prompt template: {template}")


def remove_amr_output_envelope(text: str, *, envelope: str) -> tuple[str, bool]:
    """Remove only a registered deterministic transport envelope.

    This is deliberately narrower than graph extraction: it never searches for
    a parenthesized substring and never repairs a malformed payload.
    """

    if envelope == "none":
        return text, False
    if envelope == "viamr-amr-tag":
        right_trimmed = text.rstrip()
        closing_tag = "</amr>"
        if right_trimmed.endswith(closing_tag):
            return right_trimmed[: -len(closing_tag)].rstrip(), True
        return text, False
    raise ValueError(f"unsupported AMR output envelope: {envelope}")


def extract_first_penman_graph(text: str) -> str:
    """Extract one balanced graph for postprocessed diagnostics.

    Raw validity remains a separate metric. Parentheses inside escaped PENMAN
    strings do not affect the nesting count.
    """

    start = text.find("(")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
            if depth < 0:
                return ""
    return ""


def encode_amr_supervised_record(
    tokenizer: Any,
    record: AMRCorpusRecord,
    *,
    corpus_path: Path,
    max_length: int,
    instruction: str = DEFAULT_AMR_INSTRUCTION,
) -> dict[str, Any]:
    """Tokenize one paired-block AMR record without flattening its graph."""

    identifier = f"{corpus_path}:record:{record.ordinal}"
    return encode_supervised_pair(
        tokenizer,
        identifier=identifier,
        prompt=build_amr_prompt(record.sentence, instruction=instruction),
        target=record.graph,
        max_length=max_length,
    )
