"""Validate the six final-policy outputs before marking evaluation complete."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED = {
    "popqa_eval": ("retention", 1000),
    "hotpotqa_eval": ("retention", 1000),
    "popqa_special": ("retention", 954),
    "hotpotqa_special": ("retention", 862),
    "popqa_inferattack": ("connectivity", 167),
    "hotpotqa_inferattack": ("connectivity", 45),
}


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def bounded(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= value <= 1.0


def validate(
    rewrite_root: Path,
    metric_root: Path,
    expected: Mapping[str, tuple[str, int]] = EXPECTED,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for label, (kind, expected_records) in expected.items():
        rewrite_path = rewrite_root / f"{label}.jsonl"
        metric_path = metric_root / f"{label}.json"
        if not rewrite_path.is_file() or not metric_path.is_file():
            raise FileNotFoundError(f"missing evaluation artifacts for {label}")
        rewrite_records = count_jsonl(rewrite_path)
        if rewrite_records != expected_records:
            raise ValueError(
                f"{label}: rewrite records {rewrite_records} != {expected_records}"
            )

        payload = json.loads(metric_path.read_text(encoding="utf-8"))
        expected_evaluator = (
            "test_special.py" if kind == "retention" else "test_inferattack.py"
        )
        if payload.get("schema") != "eraser4rag-evaluation-v1":
            raise ValueError(f"{label}: invalid metric schema")
        if payload.get("evaluator") != expected_evaluator:
            raise ValueError(f"{label}: invalid evaluator")
        if Path(payload.get("data_path", "")).name != rewrite_path.name:
            raise ValueError(f"{label}: metric data path does not match rewrite")
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict) or metrics.get("records") != expected_records:
            raise ValueError(f"{label}: invalid evaluated record count")

        if kind == "retention":
            if not isinstance(metrics.get("documents"), int) or metrics["documents"] <= 0:
                raise ValueError(f"{label}: retention denominator is empty")
            values = [metrics.get("r_pub"), metrics.get("r_pri")]
        else:
            private = metrics.get("private_triples")
            connected = metrics.get("connected_private_triples")
            if not isinstance(private, int) or private <= 0:
                raise ValueError(f"{label}: connectivity denominator is empty")
            if not isinstance(connected, int) or not 0 <= connected <= private:
                raise ValueError(f"{label}: invalid connected triple count")
            values = [metrics.get("r_connect_macro"), metrics.get("r_connect_micro")]
        if not all(bounded(value) for value in values):
            raise ValueError(f"{label}: metric outside [0, 1] or non-finite")
        artifacts[label] = {
            "kind": kind,
            "rewrite_records": rewrite_records,
            "metrics": metrics,
        }
    return {"schema": "eraser4rag-evaluation-validation-v1", "artifacts": artifacts}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewrite-root", required=True, type=Path)
    parser.add_argument("--metric-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = validate(args.rewrite_root, args.metric_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
