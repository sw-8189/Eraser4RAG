from __future__ import annotations

import json

import pytest

from scripts.prepare_reduced_retrieval_data import (
    build_hotpot_record,
    build_popqa_record,
    hotpot_has_top_k_contexts,
    normalize_answers,
    parse_hotpot_context,
    parse_popqa_document,
    select_unique_indices,
)


def test_normalize_answers_parses_json_and_deduplicates_casefolded_values():
    assert normalize_answers('["Paris", "paris", "City of Paris"]') == [
        "Paris",
        "City of Paris",
    ]


def test_parse_popqa_document_supports_structured_and_title_content_text():
    assert parse_popqa_document(
        {"title": "Alpha", "text": "Alpha text"}, doc_id="x"
    ) == {"id": "x", "title": "Alpha", "text": "Alpha text"}
    assert parse_popqa_document(
        "Title: Beta\nContent: Beta text", doc_id="y"
    ) == {"id": "y", "title": "Beta", "text": "Beta text"}


def test_parse_popqa_document_preserves_body_only_single_line():
    assert parse_popqa_document("body only", doc_id="x") == {
        "id": "x",
        "title": "[untitled retrieved passage]",
        "text": "body only",
    }


def test_hotpot_context_supports_dict_of_lists_and_stable_ids():
    contexts = parse_hotpot_context(
        {"title": ["A", "B"], "sentences": [["One.", "Two."], ["Three."]]},
        record_id="q1",
    )
    assert contexts == [
        {"id": "hotpotqa:q1:0", "title": "A", "text": "One. Two."},
        {"id": "hotpotqa:q1:1", "title": "B", "text": "Three."},
    ]


def test_hotpot_eligibility_requires_matched_top_k_contexts():
    assert hotpot_has_top_k_contexts(
        {"context": {"title": ["A", "B"], "sentences": [["A"], ["B"]]}},
        top_k=2,
    )
    assert not hotpot_has_top_k_contexts(
        {"context": {"title": ["A"], "sentences": [["A"]]}}, top_k=2
    )
    assert not hotpot_has_top_k_contexts(
        {"context": {"title": ["A", "B"], "sentences": [["A"]]}}, top_k=2
    )


def test_builders_enforce_top_k_and_record_source_metadata():
    pop_row = {
        "id": 7,
        "question": "Question?",
        "possible_answers": '["Answer"]',
        "retrieved_docs": [
            {"title": f"Title {index}", "text": f"Text {index}"}
            for index in range(3)
        ],
    }
    pop = build_popqa_record(pop_row, source_index=11, top_k=2)
    assert len(pop["ctxs"]) == 2
    assert pop["metadata"]["source_index"] == 11
    assert pop["ctxs"][0]["id"] == "popqa:7:0"

    with pytest.raises(ValueError, match="fewer than 4"):
        build_popqa_record(pop_row, source_index=11, top_k=4)

    hot_row = {
        "id": "h1",
        "question": "Hot question?",
        "answer": "Hot answer",
        "context": {
            "title": ["A", "B"],
            "sentences": [["A text"], ["B text"]],
        },
    }
    hot = build_hotpot_record(
        hot_row, source_index=2, top_k=2, source_split="train"
    )
    assert hot["answers"] == ["Hot answer"]
    assert hot["metadata"]["source_split"] == "train"


def test_selection_is_deterministic_unique_and_respects_exclusions():
    rows = [
        (0, {"question": "Same question"}),
        (1, {"question": " same   QUESTION "}),
        (2, {"question": "Second"}),
        (3, {"question": "Third"}),
    ]
    first = select_unique_indices(rows, size=2, seed=42)
    second = select_unique_indices(rows, size=2, seed=42)
    assert first == second
    assert len(first[0]) == len(set(first[0])) == 2
    assert len(first[1]) == len(set(first[1])) == 2

    excluded = select_unique_indices(
        rows, size=1, seed=42, excluded_questions={"same question", "second"}
    )
    assert excluded[1] == ["third"]


def test_selection_fails_when_unique_pool_is_too_small():
    rows = [(0, {"question": "Q"}), (1, {"question": " q "})]
    with pytest.raises(ValueError, match="only 1 unique"):
        select_unique_indices(rows, size=2, seed=42)
