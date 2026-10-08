"""Lexical inventories and predicates for PMB Sequence Box Notation.

The closed inventories follow the PMB 5-era SBN conversion code used by the
project and are checked against the strict gold corpus in this repository.
Lexical predicates are intentionally dependency-free so they can later be
compiled or cached for a model tokenizer.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import AbstractSet


SEPARATORS: frozenset[str] = frozenset(
    {
        "ALTERNATION",
        "ATTRIBUTION",
        "COMMENTARY",
        "CONDITION",
        "CONJUNCTION",
        "CONSEQUENCE",
        "CONTINUATION",
        "CONTRAST",
        "ELABORATION",
        "EXPLANATION",
        "NECESSITY",
        "NEGATION",
        "POSSIBILITY",
        "PRECONDITION",
        "RESULT",
        "SOURCE",
    }
)

DRS_OPERATORS: frozenset[str] = frozenset(
    {
        "ANA",
        "APX",
        "BOT",
        "EPR",
        "EQU",
        "ESU",
        "LEQ",
        "LES",
        "MOR",
        "NEQ",
        "STI",
        "STO",
        "SXP",
        "SXN",
        "SXY",
        "SY1",
        "SY2",
        "SZN",
        "SZP",
        "TAB",
        "TCT",
        "TIN",
        "TOP",
        "TPR",
        "TSU",
    }
)

ROLES: frozenset[str] = frozenset(
    {
        "Affectee",
        "Affector",
        "Agent",
        "Asset",
        "Attribute",
        "AttributeOf",
        "Bearer",
        "Beneficiary",
        "Cause",
        "Causer",
        "ClockTime",
        "Co-Agent",
        "Co-Patient",
        "Co-Theme",
        "Colour",
        "ColourOf",
        "Consumer",
        "Content",
        "ContentOf",
        "Context",
        "Creator",
        "DayOfMonth",
        "DayOfWeek",
        "Decade",
        "Degree",
        "Destination",
        "Duration",
        "Equal",
        "Experiencer",
        "Extent",
        "FeatureOf",
        "Finish",
        "Frequency",
        "Goal",
        "Instance",
        "InstanceOf",
        "Instrument",
        "Location",
        "MadeOf",
        "Manner",
        "Material",
        "Measure",
        "MonthOfYear",
        "Name",
        "Of",
        "Operand",
        "Order",
        "Owner",
        "Part",
        "Participant",
        "PartOf",
        "Path",
        "Patient",
        "Pivot",
        "Player",
        "Precondition",
        "Product",
        "Proposition",
        "Quantity",
        "QuantityOf",
        "Recipient",
        "Result",
        "Role",
        "Source",
        "Start",
        "Stimulus",
        "Sub",
        "SubOf",
        "Theme",
        "Time",
        "Title",
        "Topic",
        "Unit",
        "User",
        "Value",
        "YearOfCentury",
    }
)

ATTACHMENT_LABELS: frozenset[str] = ROLES | DRS_OPERATORS

SPECIAL_CONSTANTS: frozenset[str] = frozenset(
    {
        "here",
        "hearer",
        "now",
        "speaker",
        "unknown_ref",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "+",
        "-",
        "?",
    }
)

_SYNSET_RE = re.compile(r"^(.+)\.([nvarx])\.(\d{2})$")
_NODE_REFERENCE_RE = re.compile(r"^([+-])(0|[1-9]\d*)$")
_BOX_REFERENCE_RE = re.compile(r"^([<>])(0|[1-9]\d*)$")
_NUMBER_RE = re.compile(r"^(?:0|[1-9]\d*)(?:\.\d+)?$")
_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
_FRACTION_RE = re.compile(r"^[1-9]\d*/[1-9]\d*$")
_SCORE_RE = re.compile(r"^(?:0|[1-9]\d*)-(?:0|[1-9]\d*)$")
_YEAR_RE = re.compile(r"^'[\dX]{4}'$")


def _finite_prefix(prefix: str, values: AbstractSet[str]) -> bool:
    if isinstance(values, frozenset):
        return prefix in _finite_prefix_table(values)
    return any(value.startswith(prefix) for value in values)


@lru_cache(maxsize=None)
def _finite_prefix_table(values: frozenset[str]) -> frozenset[str]:
    """Compile a closed inventory into its trie-equivalent prefix language."""
    return frozenset(
        value[:length]
        for value in values
        for length in range(len(value) + 1)
    )


def is_synset(token: str, *, strict_lemma_case: bool = True) -> bool:
    match = _SYNSET_RE.fullmatch(token)
    if match is None:
        return False
    lemma = match.group(1)
    if any(
        char.isspace() or char == '"' or not char.isprintable()
        for char in lemma
    ):
        return False
    return not strict_lemma_case or lemma == lemma.lower()


def is_synset_prefix(prefix: str, *, strict_lemma_case: bool = True) -> bool:
    if not prefix or any(
        char.isspace() or char == '"' or not char.isprintable()
        for char in prefix
    ):
        return False
    if strict_lemma_case and prefix != prefix.lower():
        return False
    # SBN lemmas are open class and may themselves contain punctuation and
    # dots. Any lowercase, non-space lemma prefix can still be completed with
    # a POS and sense suffix.
    return True


def is_node_reference(token: str) -> bool:
    return _NODE_REFERENCE_RE.fullmatch(token) is not None


def is_node_reference_prefix(prefix: str) -> bool:
    if not prefix or prefix[0] not in "+-":
        return False
    digits = prefix[1:]
    if not digits:
        return True
    if not digits.isdigit():
        return False
    return digits == "0" or not digits.startswith("0")


def is_box_reference(token: str) -> bool:
    return _BOX_REFERENCE_RE.fullmatch(token) is not None


def is_box_reference_prefix(prefix: str, *, backward_only: bool = False) -> bool:
    allowed_signs = "<" if backward_only else "<>"
    if not prefix or prefix[0] not in allowed_signs:
        return False
    digits = prefix[1:]
    if not digits:
        return True
    if not digits.isdigit():
        return False
    return digits == "0" or not digits.startswith("0")


def parse_node_reference(token: str) -> int:
    if not is_node_reference(token):
        raise ValueError(f"Not a canonical node reference: {token!r}")
    return int(token)


def parse_box_reference(token: str) -> int:
    if not is_box_reference(token):
        raise ValueError(f"Not a canonical box reference: {token!r}")
    magnitude = int(token[1:])
    return magnitude if token[0] == ">" else -magnitude


def _quoted_prefix(prefix: str) -> tuple[bool, bool]:
    """Return ``(is_prefix, is_complete)`` for a double-quoted constant."""
    if not prefix.startswith('"'):
        return False, False
    escaped = False
    content_length = 0
    for index, char in enumerate(prefix[1:], start=1):
        if escaped:
            escaped = False
            content_length += 1
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            return index == len(prefix) - 1, index == len(prefix) - 1 and content_length > 0
        if not char.isprintable():
            return False, False
        content_length += 1
    return True, False


def quoted_string_is_open(prefix: str) -> bool:
    is_prefix, is_complete = _quoted_prefix(prefix)
    return is_prefix and not is_complete


def is_quoted_constant(token: str) -> bool:
    return _quoted_prefix(token) == (True, True)


def is_name_constant(token: str) -> bool:
    """SBN `Name` accepts a quoted string or the gold-data unknown marker."""
    return token == "?" or is_quoted_constant(token)


def is_name_constant_prefix(prefix: str) -> bool:
    quoted_prefix, _ = _quoted_prefix(prefix)
    return "?".startswith(prefix) or quoted_prefix


def _integer_prefix(text: str, *, positive_only: bool = False) -> bool:
    if not text or not text.isdigit():
        return False
    if text.startswith("0") and text != "0":
        return False
    return not positive_only or text != "0"


def _number_prefix(prefix: str) -> bool:
    if "." not in prefix:
        return _integer_prefix(prefix)
    if prefix.count(".") != 1:
        return False
    integer, fraction = prefix.split(".")
    return _integer_prefix(integer) and (not fraction or fraction.isdigit())


def _time_prefix(prefix: str) -> bool:
    if ":" not in prefix:
        return prefix.isdigit() and len(prefix) <= 2
    if prefix.count(":") != 1:
        return False
    hour, minute = prefix.split(":")
    if len(hour) != 2 or not hour.isdigit() or int(hour) > 23:
        return False
    if not minute:
        return True
    return minute.isdigit() and len(minute) <= 2 and int(minute) <= 59


def _binary_number_prefix(prefix: str, separator: str) -> bool:
    if separator not in prefix:
        return _integer_prefix(prefix, positive_only=separator == "/")
    if prefix.count(separator) != 1:
        return False
    left, right = prefix.split(separator)
    if not _integer_prefix(left, positive_only=separator == "/"):
        return False
    if not right:
        return True
    return _integer_prefix(right, positive_only=separator == "/")


def _year_prefix(prefix: str) -> bool:
    if not prefix.startswith("'"):
        return False
    body = prefix[1:]
    if body.endswith("'"):
        body = body[:-1]
        return len(body) == 4 and all(char.isdigit() or char == "X" for char in body)
    return len(body) <= 4 and all(char.isdigit() or char == "X" for char in body)


def is_constant(token: str) -> bool:
    return (
        token in SPECIAL_CONSTANTS
        or is_quoted_constant(token)
        or _NUMBER_RE.fullmatch(token) is not None
        or _TIME_RE.fullmatch(token) is not None
        or _FRACTION_RE.fullmatch(token) is not None
        or _SCORE_RE.fullmatch(token) is not None
        or _YEAR_RE.fullmatch(token) is not None
        # Single-letter symbolic constants occur in the strict corpus (`i`)
        # and the SBN specification also uses mathematical constants such as
        # pi as examples. Multi-letter bare constants remain closed-class.
        or (len(token) == 1 and token.isalpha())
    )


def is_constant_prefix(prefix: str) -> bool:
    quoted_prefix, _ = _quoted_prefix(prefix)
    return (
        _finite_prefix(prefix, SPECIAL_CONSTANTS)
        or quoted_prefix
        or _number_prefix(prefix)
        or _time_prefix(prefix)
        or _binary_number_prefix(prefix, "/")
        or _binary_number_prefix(prefix, "-")
        or _year_prefix(prefix)
        or (len(prefix) == 1 and prefix.isalpha())
    )


def finite_token_is_prefix(prefix: str, values: AbstractSet[str]) -> bool:
    """Public wrapper used by the incremental syntax state."""
    return _finite_prefix(prefix, values)
