import json
from types import SimpleNamespace

import pytest

from scripts.validate_relik_consistency import (
    choose_sample_indices,
    evaluate_records,
    load_or_create_sample_indices,
    relik_output_to_triples,
)
from utils.triple_utils import serialize_triple


def _triple(head, relation, tail):
    return SimpleNamespace(
        subject=SimpleNamespace(text=head),
        label=relation,
        object=SimpleNamespace(text=tail),
    )


def test_sample_indices_are_fixed_by_seed():
    assert choose_sample_indices(total_records=100, sample_size=10, seed=42) == choose_sample_indices(
        total_records=100, sample_size=10, seed=42
    )


def test_relik_conversion_preserves_author_entity_pair_deduplication():
    output = SimpleNamespace(
        triplets=[
            _triple("Alice", "knows", "Bob"),
            _triple("Alice", "works_with", "Bob"),
            _triple("Alice", "self", "Alice"),
        ]
    )
    result = relik_output_to_triples(output)
    assert result == {serialize_triple(["Alice", "knows", "Bob"])}


def test_consistency_metrics_use_original_local_references():
    public = serialize_triple(["Alice", "works_at", "Example Lab"])
    private = serialize_triple(["Alice", "lives_in", "Secret Town"])
    records = [
        (
            7,
            {
                "text": "Alice works at Example Lab and lives in Secret Town.",
                "anonymized_text": "This field must not be used.",
                "o_public": [public],
                "o_private": [private],
            },
        )
    ]

    def extractor(_):
        return SimpleNamespace(
            triplets=[
                _triple("Alice", "works_at", "Example Lab"),
                _triple("Alice", "lives_in", "Secret Town"),
            ]
        )

    metrics, mismatches = evaluate_records(records, extractor)

    assert metrics["public_micro_recall"] == 1.0
    assert metrics["private_micro_recall"] == 1.0
    assert metrics["zero_match_ratio"] == 0.0
    assert mismatches == []


def test_sample_manifest_detects_content_or_index_tampering(tmp_path):
    data = tmp_path / "data.jsonl"
    output = tmp_path / "sample_indices.json"
    data.write_text("{}\n{}\n{}\n", encoding="utf-8")
    expected = load_or_create_sample_indices(
        data, output, total_records=3, sample_size=2, seed=42
    )
    assert len(expected) == 2

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["indices"] = [expected[0], expected[0]]
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates"):
        load_or_create_sample_indices(
            data, output, total_records=3, sample_size=2, seed=42
        )

    output.unlink()
    load_or_create_sample_indices(data, output, total_records=3, sample_size=2, seed=42)
    data.write_text("{}\n[]\n{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="data_sha256"):
        load_or_create_sample_indices(
            data, output, total_records=3, sample_size=2, seed=42
        )
