import json

import pytest

from scripts.validate_two_dataset_eval import EXPECTED, validate


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
            json.dumps(metric_payload(label, kind, records)) + "\n",
            encoding="utf-8",
        )
    return rewrites, metrics


def test_six_metric_validation_passes_complete_artifacts(tmp_path):
    rewrites, metrics = build_artifacts(tmp_path)
    report = validate(rewrites, metrics)
    assert report["schema"] == "eraser4rag-evaluation-validation-v1"
    assert set(report["artifacts"]) == set(EXPECTED)


def test_validation_rejects_zero_retention_denominator(tmp_path):
    rewrites, metrics = build_artifacts(tmp_path)
    path = metrics / "popqa_eval.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metrics"]["documents"] = 0
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="denominator is empty"):
        validate(rewrites, metrics)
