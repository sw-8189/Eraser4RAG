"""Validate the retrieval JSONL contract before Coreferee/ReLiK extraction."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_retrieved_data(
    data_path: str | Path, *, expected_contexts: int = 10
) -> dict[str, Any]:
    if expected_contexts <= 0:
        raise ValueError("expected_contexts must be positive")
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"retrieval data not found: {path}")

    counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []

    def error(line: int, field: str, message: str) -> None:
        counts["invalid_records"] += 1
        if len(errors) < 100:
            errors.append({"line": line, "field": field, "message": message})

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            counts["records"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                error(line_number, "record", f"invalid JSON: {exc.msg}")
                continue
            if not isinstance(record, dict):
                error(line_number, "record", "must be an object")
                continue
            if not _non_empty_text(record.get("question")):
                error(line_number, "question", "must be a non-empty string")
            answers = record.get("answers")
            if not isinstance(answers, list) or not answers or not all(
                _non_empty_text(answer) for answer in answers
            ):
                error(line_number, "answers", "must be a non-empty list of strings")
            contexts = record.get("ctxs")
            if not isinstance(contexts, list) or len(contexts) != expected_contexts:
                actual = len(contexts) if isinstance(contexts, list) else type(contexts).__name__
                error(
                    line_number,
                    "ctxs",
                    f"must contain exactly {expected_contexts} contexts; got {actual}",
                )
                continue
            context_ids: set[str] = set()
            for context_index, context in enumerate(contexts):
                prefix = f"ctxs[{context_index}]"
                if not isinstance(context, dict):
                    error(line_number, prefix, "must be an object")
                    continue
                for field in ("id", "title", "text"):
                    if not _non_empty_text(context.get(field)):
                        error(line_number, f"{prefix}.{field}", "must be a non-empty string")
                context_id = context.get("id")
                if isinstance(context_id, str) and context_id in context_ids:
                    error(line_number, f"{prefix}.id", "must be unique within a record")
                elif isinstance(context_id, str):
                    context_ids.add(context_id)
            counts["contexts"] += len(contexts)

    return {
        "schema": "eraser4rag-retrieved-data-validation-v1",
        "data_path": str(path.resolve()),
        "expected_contexts": expected_contexts,
        "valid": not errors,
        "counts": dict(counts),
        "errors": errors,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--expected-contexts", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    reports = [
        validate_retrieved_data(path, expected_contexts=args.expected_contexts)
        for path in args.data
    ]
    result = {
        "schema": "eraser4rag-retrieved-data-validation-batch-v1",
        "valid": all(report["valid"] for report in reports),
        "reports": reports,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
