import json

import pytest

from scripts.validate_popqa_sft_only_eval import EXPECTED, validate


def write_jsonl(path, records):
    path.write_text("{}\n" * records, encoding="utf-8")


def metric_payload(label, kind, records):
    if kind == "retention":
        metrics = {"records": records, "documents": records * 10, "r_pub": 0.5, "r_pri": 0.1}
        evaluator = "test_special.py"
    else:
        metrics = {
            "records": records,
            "private_triples": records * 2,
            "connected_private_triples": records,
            "r_connect_macro": 0.5,
            "r_connect_micro": 0.5,
        }
        evaluator = "test_inferattack.py"
    return {
        "schema": "eraser4rag-evaluation-v1",
        "evaluator": evaluator,
        "data_path": f"/generated/{label}.jsonl",
        "metrics": metrics,
    }


def build_artifacts(tmp_path):
    rewrites = tmp_path / "rewritten"
    metrics = tmp_path / "metrics"
    rewrites.mkdir()
    metrics.mkdir()
    for label, (kind, records) in EXPECTED.items():
        write_jsonl(rewrites / f"{label}.jsonl", records)
        (metrics / f"{label}.json").write_text(
            json.dumps(metric_payload(label, kind, records)) + "\n", encoding="utf-8"
        )
    return rewrites, metrics


def test_popqa_sft_only_validation_accepts_complete_artifacts(tmp_path):
    rewrites, metrics = build_artifacts(tmp_path)
    report = validate(rewrites, metrics)
    assert report["schema"] == "eraser4rag-popqa-sft-only-validation-v1"
    assert set(report["artifacts"]) == set(EXPECTED)


def test_popqa_sft_only_validation_rejects_wrong_count(tmp_path):
    rewrites, metrics = build_artifacts(tmp_path)
    path = rewrites / "popqa_special.jsonl"
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="rewrite records 1 != 954"):
        validate(rewrites, metrics)
