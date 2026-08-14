from __future__ import annotations

import json

from scripts.validate_retrieved_data import validate_retrieved_data


def _record(*, duplicate_id: bool = False, context_count: int = 10) -> dict:
    contexts = [
        {"id": f"doc-{index}", "title": f"Title {index}", "text": f"Text {index}"}
        for index in range(context_count)
    ]
    if duplicate_id and len(contexts) > 1:
        contexts[1]["id"] = contexts[0]["id"]
    return {"question": "Question?", "answers": ["Answer"], "ctxs": contexts}


def test_validate_retrieved_data_accepts_exact_top_k(tmp_path):
    path = tmp_path / "valid.jsonl"
    path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")

    report = validate_retrieved_data(path)

    assert report["valid"]
    assert report["counts"] == {"records": 1, "contexts": 10}


def test_validate_retrieved_data_reports_context_count_and_duplicate_ids(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text(
        "\n".join(
            [json.dumps(_record(context_count=9)), json.dumps(_record(duplicate_id=True))]
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_retrieved_data(path)

    assert not report["valid"]
    assert report["counts"]["invalid_records"] == 2
    assert any(item["field"] == "ctxs" for item in report["errors"])
    assert any(item["field"] == "ctxs[1].id" for item in report["errors"])
