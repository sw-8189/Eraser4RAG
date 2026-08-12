import json

from scripts.validate_sft_dataset import validate_sft_dataset
from utils.triple_utils import serialize_triple


def _record(public=None, private=None):
    public = public or [["Alice", "works_at", "Example Lab"]]
    private = private or [["Alice", "lives_in", "Secret Town"]]
    return {
        "public": public,
        "private": private,
        "text": "Alice works at Example Lab and lives in Secret Town.",
        "anonymized_text": "Alice works at Example Lab.",
        "o_public": [serialize_triple(public[0])],
        "o_private": [serialize_triple(private[0])],
    }


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_valid_sft_schema_and_grouping(tmp_path):
    path = tmp_path / "sft.jsonl"
    first = _record()
    second = dict(first, text="A second document in the same retrieval group.")
    third = _record(
        public=[["Bob", "works_at", "Other Lab"]],
        private=[["Bob", "lives_in", "Other Secret"]],
    )
    _write_jsonl(path, [first, second, third])

    report = validate_sft_dataset(path)

    assert report["valid"] is True
    assert report["counts"]["records_total"] == 3
    assert report["groups"]["unique_fingerprint_count"] == 2
    assert report["groups"]["consecutive_segment_count"] == 2
    assert report["groups"]["documents_per_segment"]["max"] == 2


def test_sft_local_triple_must_belong_to_global_set(tmp_path):
    path = tmp_path / "invalid.jsonl"
    record = _record()
    record["o_private"] = [serialize_triple(["Mallory", "knows", "Secret"])]
    _write_jsonl(path, [record])

    report = validate_sft_dataset(path)

    assert report["valid"] is False
    assert any(error["field"] == "o_private" for error in report["errors"])


def test_sft_failure_is_not_hidden_when_error_examples_are_disabled(tmp_path):
    path = tmp_path / "invalid-without-examples.jsonl"
    record = _record()
    record["o_private"] = [serialize_triple(["Mallory", "knows", "Secret"])]
    _write_jsonl(path, [record])

    report = validate_sft_dataset(path, max_error_examples=0)

    assert report["valid"] is False
    assert report["counts"]["records_with_errors"] == 1
    assert report["errors"] == []


def test_sft_non_contiguous_group_is_reported(tmp_path):
    path = tmp_path / "groups.jsonl"
    first = _record()
    middle = _record(
        public=[["Bob", "works_at", "Other Lab"]],
        private=[["Bob", "lives_in", "Other Secret"]],
    )
    last = dict(first, text="The first graph appears again later.")
    _write_jsonl(path, [first, middle, last])

    report = validate_sft_dataset(path)

    assert report["valid"] is True
    assert report["groups"]["non_contiguous_fingerprint_count"] == 1


def test_sft_raw_triples_use_shared_strict_validation(tmp_path):
    path = tmp_path / "invalid-raw-triple.jsonl"
    record = _record()
    record["public"] = [["", "works_at", "Example Lab"]]
    record["o_public"] = []
    _write_jsonl(path, [record])

    report = validate_sft_dataset(path)

    assert report["valid"] is False
    assert any("must not be empty" in item["message"] for item in report["errors"])
