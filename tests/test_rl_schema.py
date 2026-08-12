import json

import pytest

from scripts.build_popqa_rl_from_sft import build_smoke_dataset
from scripts.validate_rl_dataset import validate_rl_dataset
from utils.triple_utils import serialize_triple


def _rl_record():
    public = [["Alice", "works_at", "Example Lab"]]
    private = [["Alice", "lives_in", "Secret Town"]]
    return {
        "question": "Where does Alice work?",
        "answers": ["Example Lab"],
        "public": public,
        "privacy": private,
        "ctxs": [
            {
                "id": "doc-1",
                "text": "Alice works at Example Lab and lives in Secret Town.",
                "public": [serialize_triple(public[0])],
                "private": [serialize_triple(private[0])],
            }
        ],
    }


def _write(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_valid_formal_rl_schema(tmp_path):
    path = tmp_path / "rl.jsonl"
    _write(path, [_rl_record()])

    report = validate_rl_dataset(path, require_qa=True)

    assert report["valid"] is True
    assert report["counts"]["query_count"] == 1
    assert report["counts"]["document_count"] == 1


def test_rl_local_subset_failure(tmp_path):
    path = tmp_path / "invalid.jsonl"
    record = _rl_record()
    record["ctxs"][0]["private"] = [serialize_triple(["X", "knows", "Y"])]
    _write(path, [record])

    report = validate_rl_dataset(path)

    assert report["valid"] is False
    assert any("absent from global privacy" in item["message"] for item in report["errors"])


def test_rl_failure_is_not_hidden_when_error_examples_are_disabled(tmp_path):
    path = tmp_path / "invalid-without-examples.jsonl"
    record = _rl_record()
    record["ctxs"][0]["private"] = [serialize_triple(["X", "knows", "Y"])]
    _write(path, [record])

    report = validate_rl_dataset(path, max_error_examples=0)

    assert report["valid"] is False
    assert report["counts"]["records_with_errors"] == 1
    assert report["errors"] == []


def test_regrouped_sft_is_marked_smoke_only_and_validates(tmp_path):
    sft = tmp_path / "sft.jsonl"
    output = tmp_path / "smoke.jsonl"
    global_public = [["Alice", "works_at", "Example Lab"]]
    global_private = [["Alice", "lives_in", "Secret Town"]]
    source_rows = [
        {
            "public": global_public,
            "private": global_private,
            "text": f"Document {index}",
            "anonymized_text": f"Rewritten {index}",
            "o_public": [serialize_triple(global_public[0])],
            "o_private": [serialize_triple(global_private[0])],
        }
        for index in range(2)
    ]
    _write(sft, source_rows)

    stats = build_smoke_dataset(sft, output)
    result = json.loads(output.read_text(encoding="utf-8").splitlines()[0])

    assert stats["groups_written"] == 1
    assert result["metadata"]["source"] == "regrouped_author_sft"
    assert result["metadata"]["purpose"] == "ppo_engineering_smoke_only"
    assert "anonymized_text" not in result["ctxs"][0]
    assert validate_rl_dataset(output)["valid"] is True


def test_regrouping_rejects_an_empty_sft_dataset(tmp_path):
    sft = tmp_path / "empty.jsonl"
    output = tmp_path / "smoke.jsonl"
    sft.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no PPO smoke groups"):
        build_smoke_dataset(sft, output)

    assert not output.exists()


def test_formal_qa_requirement_rejects_reconstructed_empty_question(tmp_path):
    path = tmp_path / "engineering.jsonl"
    record = _rl_record()
    record["question"] = ""
    record["answers"] = []
    _write(path, [record])

    report = validate_rl_dataset(path, require_qa=True)

    assert report["valid"] is False


def test_formal_schema_rejects_empty_answers_and_contexts(tmp_path):
    path = tmp_path / "empty-formal.jsonl"
    record = _rl_record()
    record["answers"] = []
    record["ctxs"] = []
    _write(path, [record])

    report = validate_rl_dataset(path, require_qa=True)

    assert report["valid"] is False
    assert {item["location"] for item in report["errors"]} >= {"answers", "ctxs"}


def test_rl_raw_triples_use_shared_strict_validation(tmp_path):
    path = tmp_path / "invalid-raw-triple.jsonl"
    record = _rl_record()
    record["public"] = [["", "works_at", "Example Lab"]]
    record["ctxs"][0]["public"] = []
    _write(path, [record])

    report = validate_rl_dataset(path)

    assert report["valid"] is False
    assert any("must not be empty" in item["message"] for item in report["errors"])
