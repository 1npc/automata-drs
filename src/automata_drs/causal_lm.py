"""Shared data handling for decoder-only SBN fine-tuning.

The functions in this module intentionally do not import Torch or Transformers.
That keeps the corpus/prompt contract testable on machines which do not have the
model-training extras installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


DEFAULT_INSTRUCTION = (
    "Translate the text into one-line PMB Sequence Box Notation (SBN)."
)


@dataclass(frozen=True)
class SBNRecord:
    """One source/SBN pair with a stable corpus identifier."""

    identifier: str
    source: str
    target: str


def iter_sbn_records(
    path: Path, *, start: int = 0, limit: int = -1
) -> Iterator[SBNRecord]:
    """Read the repository's ``source<TAB>SBN`` corpus format.

    ``start`` and ``limit`` count non-empty records.  Empty source or target
    fields are rejected instead of entering a training run silently.
    """

    if start < 0:
        raise ValueError("start must be non-negative")
    if limit == 0 or limit < -1:
        raise ValueError("limit must be -1 or positive")

    available = 0
    emitted = 0
    with path.open(encoding="utf-8") as source_file:
        for line_number, raw_line in enumerate(source_file, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            if "\t" not in line:
                raise ValueError(
                    f"expected source<TAB>SBN at {path}:{line_number}"
                )
            source, target = line.split("\t", 1)
            if not source or not target:
                raise ValueError(f"empty source or SBN at {path}:{line_number}")
            if available < start:
                available += 1
                continue
            yield SBNRecord(
                identifier=f"{path}:{line_number}",
                source=source,
                target=target,
            )
            emitted += 1
            if limit > 0 and emitted >= limit:
                return


def build_sbn_prompt(source: str, *, instruction: str = DEFAULT_INSTRUCTION) -> str:
    """Return the model-family-neutral plain-text generation prompt."""

    if "\n" in source or "\r" in source:
        raise ValueError("SBN source text must occupy one line")
    return f"{instruction}\nText: {source}\nSBN:\n"


def select_sbn_records_by_line(
    records: Iterable[SBNRecord], line_numbers: Sequence[int]
) -> list[SBNRecord]:
    """Select stable physical TSV lines in the requested order.

    Conditioned intervention experiments use this instead of materializing a
    second corpus file. Missing, non-positive, or duplicate requests fail
    closed so the generated manifest remains an exact description of the
    evaluated population.
    """

    requested = [int(line_number) for line_number in line_numbers]
    if not requested:
        raise ValueError("line_numbers must not be empty")
    if any(line_number <= 0 for line_number in requested):
        raise ValueError("line_numbers must be positive")
    if len(set(requested)) != len(requested):
        raise ValueError("line_numbers must not contain duplicates")

    available: dict[int, SBNRecord] = {}
    for record in records:
        try:
            line_number = int(record.identifier.rsplit(":", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(
                f"record identifier has no physical line suffix: {record.identifier}"
            ) from error
        if line_number in available:
            raise ValueError(f"duplicate record line identifier: {line_number}")
        available[line_number] = record

    missing = [line_number for line_number in requested if line_number not in available]
    if missing:
        raise ValueError(
            "requested TSV lines are not non-empty records: "
            + ", ".join(str(line_number) for line_number in missing)
        )
    return [available[line_number] for line_number in requested]


def scheduled_generation_identities(
    records: Sequence[SBNRecord],
    configurations: Sequence[str],
    *,
    schedule: str,
) -> list[tuple[str, str]]:
    """Return the deterministic JSONL cell order for causal generation."""

    names = [str(configuration) for configuration in configurations]
    if not names or len(set(names)) != len(names):
        raise ValueError("configurations must be non-empty and unique")
    if schedule not in {"rotate", "fixed"}:
        raise ValueError(f"unsupported generation schedule: {schedule}")
    identities: list[tuple[str, str]] = []
    for record_index, record in enumerate(records):
        if schedule == "rotate":
            shift = record_index % len(names)
            ordered = names[shift:] + names[:shift]
        else:
            ordered = names
        identities.extend((record.identifier, name) for name in ordered)
    return identities


def validate_generation_resume_prefix(
    rows: Sequence[dict[str, Any]],
    expected: Sequence[tuple[str, str]],
    *,
    num_beams: int,
) -> int:
    """Validate that existing prediction rows are an exact scheduled prefix."""

    if len(rows) > len(expected):
        raise ValueError(
            f"resume output has {len(rows)} rows but only {len(expected)} are expected"
        )
    expected_strategy = "greedy" if num_beams == 1 else "beam"
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"resume row {index + 1} is not a JSON object")
        identity = (str(row.get("id", "")), str(row.get("configuration", "")))
        if identity in seen:
            raise ValueError(f"duplicate resume cell at row {index + 1}: {identity}")
        seen.add(identity)
        if identity != tuple(expected[index]):
            raise ValueError(
                f"resume row {index + 1} is out of schedule: "
                f"found {identity}, expected {tuple(expected[index])}"
            )
        decode = row.get("decode", {})
        if (
            decode.get("strategy") != expected_strategy
            or decode.get("num_beams") != num_beams
        ):
            raise ValueError(
                f"resume row {index + 1} has incompatible decoding metadata"
            )
    return len(rows)


def _token_ids(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("expected one tokenized example")
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def encode_supervised_pair(
    tokenizer: Any,
    *,
    identifier: str,
    prompt: str,
    target: str,
    max_length: int,
) -> dict[str, Any]:
    """Tokenize a prompt/target pair and supervise only target tokens plus EOS."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    if not target:
        raise ValueError(f"{identifier} has an empty target")

    prompt_ids = _token_ids(tokenizer, prompt, add_special_tokens=True)
    target_ids = _token_ids(tokenizer, target, add_special_tokens=False)
    if not target_ids or target_ids[-1] != int(eos_token_id):
        target_ids.append(int(eos_token_id))

    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_length:
        raise ValueError(
            f"{identifier} requires {len(input_ids)} tokens, "
            f"exceeding max_length={max_length}"
        )
    return {
        "id": identifier,
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(target_ids),
    }


def encode_supervised_record(
    tokenizer: Any,
    record: SBNRecord,
    *,
    max_length: int,
    instruction: str = DEFAULT_INSTRUCTION,
) -> dict[str, Any]:
    """Tokenize one causal-LM example and mask all prompt labels.

    Prompt and target are encoded separately on purpose: the exact token
    sequence presented at training time is therefore the same prompt prefix
    used at inference time.  A terminal EOS is always supervised.
    """

    prompt = build_sbn_prompt(record.source, instruction=instruction)
    return encode_supervised_pair(
        tokenizer,
        identifier=record.identifier,
        prompt=prompt,
        target=record.target,
        max_length=max_length,
    )


def pad_supervised_batch(
    examples: Sequence[dict[str, Any]],
    *,
    pad_token_id: int,
    pad_to_multiple_of: int = 8,
) -> dict[str, list[list[int]]]:
    """Right-pad tokenized examples without depending on a tensor library."""

    if not examples:
        raise ValueError("cannot pad an empty batch")
    if pad_to_multiple_of <= 0:
        raise ValueError("pad_to_multiple_of must be positive")
    width = max(len(example["input_ids"]) for example in examples)
    width = (
        (width + pad_to_multiple_of - 1) // pad_to_multiple_of
    ) * pad_to_multiple_of

    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for example in examples:
        padding = width - len(example["input_ids"])
        batch["input_ids"].append(
            list(example["input_ids"]) + [int(pad_token_id)] * padding
        )
        batch["attention_mask"].append(
            list(example["attention_mask"]) + [0] * padding
        )
        batch["labels"].append(list(example["labels"]) + [-100] * padding)
    return batch


def extract_one_line_sbn(generated_text: str) -> str:
    """Extract the first non-empty generated line as the SBN prediction."""

    for line in generated_text.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    return ""
