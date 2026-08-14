"""Strict serialization helpers for Eraser4RAG knowledge triples.

The helpers preserve entity/relation content while canonicalizing internal
 whitespace. They do not lowercase, fuzzy-match, or silently repair malformed
 triples because those operations would change the reference sets used by the
 paper's reward.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import re
from typing import List, Optional


NEUTRAL = "neutral"
PUBLIC = "public"
PRIVATE = "private"

SPECIAL_TOKENS = (
    "<csubj>",
    "<crel>",
    "<cobj>",
    "<ce>",
    "<rsubj>",
    "<rrel>",
    "<robj>",
    "<re>",
)

_SERIALIZATION_TAGS = {
    NEUTRAL: ("<subj>", "<rel>", "<obj>", "<e>"),
    PUBLIC: ("<csubj>", "<crel>", "<cobj>", "<ce>"),
    PRIVATE: ("<rsubj>", "<rrel>", "<robj>", "<re>"),
}
_ALL_TAGS = frozenset(tag for tags in _SERIALIZATION_TAGS.values() for tag in tags)


class TripleFormatError(ValueError):
    """Raised when a triple or serialized triple is malformed."""


def _get_tags(serialization: str) -> tuple[str, str, str, str]:
    try:
        return _SERIALIZATION_TAGS[serialization]
    except KeyError as exc:
        supported = ", ".join(sorted(_SERIALIZATION_TAGS))
        raise TripleFormatError(
            f"Unknown serialization {serialization!r}; expected one of: {supported}"
        ) from exc


def _coerce_raw_triple(triple: Sequence[str]) -> List[str]:
    if isinstance(triple, (str, bytes)) or not isinstance(triple, Sequence):
        raise TripleFormatError("A raw triple must be a three-item sequence of strings")
    if len(triple) != 3:
        raise TripleFormatError(f"A triple must contain exactly three items; got {len(triple)}")

    # Coreferee/ReLiK can return an entity span containing a line break. Keep
    # the entity text but make raw and tagged representations compare equally.
    normalized = [
        " ".join(component.split()) if isinstance(component, str) else component
        for component in triple
    ]
    for index, component in enumerate(normalized):
        if not isinstance(component, str):
            raise TripleFormatError(
                f"Triple component {index} must be a string; got {type(component).__name__}"
            )
        if not component or not component.strip():
            raise TripleFormatError(f"Triple component {index} must not be empty")
        embedded = next((tag for tag in _ALL_TAGS if tag in component), None)
        if embedded is not None:
            raise TripleFormatError(
                f"Triple component {index} contains reserved tag {embedded!r}"
            )
    return normalized


def serialize_triple(triple: Sequence[str], serialization: str = NEUTRAL) -> str:
    """Serialize ``[head, relation, tail]`` with the selected tag family."""

    head, relation, tail = _coerce_raw_triple(triple)
    subject_tag, relation_tag, object_tag, end_tag = _get_tags(serialization)
    return (
        f"{subject_tag}{head}{relation_tag}{relation}"
        f"{object_tag}{tail}{end_tag}"
    )


def serialize_neutral_triple(triple: Sequence[str]) -> str:
    return serialize_triple(triple, NEUTRAL)


def serialize_public_prompt_triple(triple: Sequence[str]) -> str:
    return serialize_triple(triple, PUBLIC)


def serialize_private_prompt_triple(triple: Sequence[str]) -> str:
    return serialize_triple(triple, PRIVATE)


def _pattern_for(serialization: str) -> re.Pattern[str]:
    subject_tag, relation_tag, object_tag, end_tag = _get_tags(serialization)
    return re.compile(
        rf"{re.escape(subject_tag)}(.+?){re.escape(relation_tag)}"
        rf"(.+?){re.escape(object_tag)}(.+?){re.escape(end_tag)}"
    )


def parse_serialized_triple(
    text: str, serialization: Optional[str] = None
) -> List[str]:
    """Strictly parse one complete tagged triple.

    When ``serialization`` is omitted, the tag family is auto-detected.  The
    full input must match exactly; leading/trailing text and concatenated
    triples are rejected.
    """

    if not isinstance(text, str):
        raise TripleFormatError(
            f"Serialized triple must be a string; got {type(text).__name__}"
        )

    candidates = (
        (serialization,) if serialization is not None else (NEUTRAL, PUBLIC, PRIVATE)
    )
    matches: list[List[str]] = []
    for candidate in candidates:
        match = _pattern_for(candidate).fullmatch(text)
        if match is not None:
            matches.append(_coerce_raw_triple(list(match.groups())))

    if len(matches) != 1:
        expected = serialization if serialization is not None else "a supported tag family"
        raise TripleFormatError(
            f"Expected exactly one complete triple using {expected}; got {text!r}"
        )
    return matches[0]


def is_serialized_triple(text: object, serialization: Optional[str] = None) -> bool:
    if not isinstance(text, str):
        return False
    try:
        parse_serialized_triple(text, serialization)
    except TripleFormatError:
        return False
    return True


def normalize_triple_collection(
    items: object, serialization: Optional[str] = None
) -> list[List[str]]:
    """Normalize raw and tagged triples to ``list[list[str]]``.

    Duplicate triples and input order are deliberately preserved.  A single
    raw or serialized triple is accepted as a convenience.  ``None`` denotes
    an empty collection.
    """

    if items is None:
        return []
    if isinstance(items, str):
        return [parse_serialized_triple(items, serialization)]
    if isinstance(items, Mapping):
        raise TripleFormatError("A mapping is not a triple collection")

    if isinstance(items, Sequence) and len(items) == 3 and all(
        isinstance(component, str) for component in items
    ):
        # Three tagged strings are a collection; three ordinary strings are a
        # single raw triple.
        if not all(is_serialized_triple(component, serialization) for component in items):
            return [_coerce_raw_triple(items)]

    if not isinstance(items, Iterable):
        raise TripleFormatError("Triple collection must be iterable")

    normalized: list[List[str]] = []
    for item in items:
        if isinstance(item, str):
            normalized.append(parse_serialized_triple(item, serialization))
        else:
            normalized.append(_coerce_raw_triple(item))
    return normalized


__all__ = [
    "NEUTRAL",
    "PUBLIC",
    "PRIVATE",
    "SPECIAL_TOKENS",
    "TripleFormatError",
    "is_serialized_triple",
    "normalize_triple_collection",
    "parse_serialized_triple",
    "serialize_neutral_triple",
    "serialize_private_prompt_triple",
    "serialize_public_prompt_triple",
    "serialize_triple",
]
