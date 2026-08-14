import pytest

from utils.triple_utils import (
    NEUTRAL,
    PRIVATE,
    PUBLIC,
    SPECIAL_TOKENS,
    TripleFormatError,
    is_serialized_triple,
    normalize_triple_collection,
    parse_serialized_triple,
    serialize_private_prompt_triple,
    serialize_public_prompt_triple,
    serialize_triple,
)


TRIPLE = ["Marie Curie", "collaborated_with", "Pierre Curie"]


def test_special_tokens_match_author_training_tags():
    assert SPECIAL_TOKENS == (
        "<csubj>",
        "<crel>",
        "<cobj>",
        "<ce>",
        "<rsubj>",
        "<rrel>",
        "<robj>",
        "<re>",
    )


@pytest.mark.parametrize(
    ("serialization", "expected"),
    [
        (
            NEUTRAL,
            "<subj>Marie Curie<rel>collaborated_with<obj>Pierre Curie<e>",
        ),
        (
            PUBLIC,
            "<csubj>Marie Curie<crel>collaborated_with<cobj>Pierre Curie<ce>",
        ),
        (
            PRIVATE,
            "<rsubj>Marie Curie<rrel>collaborated_with<robj>Pierre Curie<re>",
        ),
    ],
)
def test_roundtrip_for_each_serialization(serialization, expected):
    serialized = serialize_triple(TRIPLE, serialization)
    assert serialized == expected
    assert parse_serialized_triple(serialized) == TRIPLE
    assert parse_serialized_triple(serialized, serialization) == TRIPLE


def test_named_public_and_private_serializers():
    assert serialize_public_prompt_triple(TRIPLE).startswith("<csubj>")
    assert serialize_private_prompt_triple(TRIPLE).startswith("<rsubj>")


@pytest.mark.parametrize(
    "malformed",
    [
        "",
        "<subj>a<rel>b<obj>c",
        "prefix<subj>a<rel>b<obj>c<e>",
        "<subj>a<rel>b<obj>c<e>suffix",
        "<subj>a<rel>b<obj>c<e><subj>d<rel>e<obj>f<e>",
        "<subj><rel>b<obj>c<e>",
    ],
)
def test_strict_parser_rejects_malformed_input(malformed):
    with pytest.raises(TripleFormatError):
        parse_serialized_triple(malformed)


def test_parser_rejects_wrong_explicit_tag_family():
    public = serialize_public_prompt_triple(TRIPLE)
    with pytest.raises(TripleFormatError):
        parse_serialized_triple(public, PRIVATE)


@pytest.mark.parametrize(
    "raw",
    [
        ["head", "relation"],
        ["head", "relation", "tail", "extra"],
        ["head", 3, "tail"],
        ["head", " ", "tail"],
        ["head<rel>injected", "relation", "tail"],
    ],
)
def test_serializer_rejects_invalid_raw_triples(raw):
    with pytest.raises(TripleFormatError):
        serialize_triple(raw)


def test_normalize_accepts_mixed_formats_and_preserves_duplicates():
    neutral = serialize_triple(TRIPLE)
    public = serialize_public_prompt_triple(["Polonium", "named_after", "Poland"])
    normalized = normalize_triple_collection([TRIPLE, neutral, public, TRIPLE])
    assert normalized == [
        TRIPLE,
        TRIPLE,
        ["Polonium", "named_after", "Poland"],
        TRIPLE,
    ]


def test_normalize_accepts_single_raw_or_serialized_triple():
    assert normalize_triple_collection(TRIPLE) == [TRIPLE]
    assert normalize_triple_collection(serialize_triple(TRIPLE)) == [TRIPLE]
    assert normalize_triple_collection(None) == []


def test_three_serialized_strings_are_treated_as_collection():
    collection = [
        serialize_triple(["h1", "r1", "t1"]),
        serialize_triple(["h2", "r2", "t2"]),
        serialize_triple(["h3", "r3", "t3"]),
    ]
    assert normalize_triple_collection(collection) == [
        ["h1", "r1", "t1"],
        ["h2", "r2", "t2"],
        ["h3", "r3", "t3"],
    ]


def test_is_serialized_triple_never_raises_for_bad_input():
    assert is_serialized_triple(serialize_triple(TRIPLE))
    assert not is_serialized_triple("not a triple")
    assert not is_serialized_triple(123)


def test_whitespace_in_components_is_canonicalized_for_tagged_roundtrip():
    raw = ["North Carolina\n Films", "country", "American"]
    serialized = serialize_triple(raw)
    assert serialized == "<subj>North Carolina Films<rel>country<obj>American<e>"
    assert parse_serialized_triple(serialized) == [
        "North Carolina Films",
        "country",
        "American",
    ]
