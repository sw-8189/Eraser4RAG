"""Validate formal or reconstructed Eraser4RAG PPO JSONL data."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.triple_utils import (
    TripleFormatError,
    parse_serialized_triple,
    serialize_triple,
)


def _summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def _records(path: Path) -> Iterator[tuple[int, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as exc:
                    yield line_number, exc
        return
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("a .json RL dataset must contain a top-level list")
        for index, record in enumerate(payload, start=1):
            yield index, record
        return
    raise ValueError(f"unsupported dataset suffix: {path.suffix}")


def _raw_triples(value: Any, location: str, add_error) -> list[tuple[str, str, str]]:
    if not isinstance(value, list):
        add_error(location, "must be a list")
        return []
    result: list[tuple[str, str, str]] = []
    for index, triple in enumerate(value):
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            add_error(f"{location}[{index}]", "must contain exactly three items")
            continue
        if not all(isinstance(part, str) for part in triple):
            add_error(f"{location}[{index}]", "triple items must be strings")
            continue
        normalized = (triple[0], triple[1], triple[2])
        try:
            # Reuse the exact strictness applied by the PPO prompt builder so
            # data cannot pass validation and then fail at training time.
            serialize_triple(normalized)
        except TripleFormatError as exc:
            add_error(f"{location}[{index}]", str(exc))
            continue
        result.append(normalized)
    return result


def _local_triples(value: Any, location: str, add_error) -> list[tuple[str, str, str]]:
    if not isinstance(value, list):
        add_error(location, "must be a list")
        return []
    result: list[tuple[str, str, str]] = []
    for index, item in enumerate(value):
        try:
            if isinstance(item, str):
                parsed = tuple(parse_serialized_triple(item))
                if serialize_triple(parsed) != item:
                    raise ValueError("non-canonical neutral serialization")
            elif isinstance(item, (list, tuple)) and len(item) == 3:
                parsed = tuple(item)
                serialize_triple(parsed)
            else:
                raise ValueError("expected a neutral tagged string or raw triple")
        except (TypeError, ValueError) as exc:
            add_error(f"{location}[{index}]", str(exc))
            continue
        if not all(isinstance(part, str) for part in parsed):
            add_error(f"{location}[{index}]", "triple items must be strings")
            continue
        result.append((parsed[0], parsed[1], parsed[2]))
    return result


def validate_rl_dataset(
    data_path: str | Path,
    *,
    require_qa: bool = False,
    max_records: int | None = None,
    max_error_examples: int = 100,
) -> dict[str, Any]:
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"RL dataset not found: {path}")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive when provided")
    if max_error_examples < 0:
        raise ValueError("max_error_examples must be non-negative")

    errors: list[dict[str, Any]] = []
    error_records: set[int] = set()
    counts: Counter[str] = Counter()
    contexts_per_query: list[int] = []
    local_public_lengths: list[int] = []
    local_private_lengths: list[int] = []

    def add_error(record_number: int, location: str, message: str) -> None:
        error_records.add(record_number)
        if len(errors) < max_error_examples:
            errors.append(
                {"record": record_number, "location": location, "message": message}
            )

    for record_number, record in _records(path):
        if max_records is not None and counts["query_count"] >= max_records:
            counts["truncated_by_max_records"] = 1
            break
        counts["query_count"] += 1
        if isinstance(record, json.JSONDecodeError):
            add_error(record_number, "<record>", f"invalid JSON: {record.msg}")
            continue
        if not isinstance(record, dict):
            add_error(record_number, "<record>", "record must be an object")
            continue
        for field in ("public", "privacy", "ctxs"):
            if field not in record:
                add_error(record_number, field, "missing required field")
        if not all(field in record for field in ("public", "privacy", "ctxs")):
            continue

        if require_qa:
            if not isinstance(record.get("question"), str) or not record.get("question", "").strip():
                add_error(record_number, "question", "formal RL data require a non-empty question")
            answers = record.get("answers")
            if (
                not isinstance(answers, list)
                or not answers
                or not all(isinstance(item, str) and item.strip() for item in answers)
            ):
                add_error(
                    record_number,
                    "answers",
                    "formal RL data require a non-empty list of non-empty strings",
                )

        def record_error(location: str, message: str) -> None:
            add_error(record_number, location, message)

        global_public = _raw_triples(record["public"], "public", record_error)
        global_private = _raw_triples(record["privacy"], "privacy", record_error)
        public_set = set(global_public)
        private_set = set(global_private)
        if public_set & private_set:
            record_error("public/privacy", "global public and private triples overlap")
        if not isinstance(record["ctxs"], list):
            record_error("ctxs", "must be a list")
            continue
        if not record["ctxs"]:
            record_error("ctxs", "must contain at least one document")
            continue

        contexts_per_query.append(len(record["ctxs"]))
        counts["document_count"] += len(record["ctxs"])
        seen_ids: set[str] = set()
        for ctx_index, ctx in enumerate(record["ctxs"]):
            prefix = f"ctxs[{ctx_index}]"
            if not isinstance(ctx, dict):
                record_error(prefix, "must be an object")
                continue
            if not isinstance(ctx.get("text"), str) or not ctx.get("text", "").strip():
                record_error(f"{prefix}.text", "must be a non-empty string")
            if "id" in ctx:
                normalized_id = str(ctx["id"])
                if normalized_id in seen_ids:
                    record_error(f"{prefix}.id", "duplicate context ID within query")
                seen_ids.add(normalized_id)
            local_public = _local_triples(ctx.get("public"), f"{prefix}.public", record_error)
            local_private = _local_triples(ctx.get("private"), f"{prefix}.private", record_error)
            local_public_lengths.append(len(local_public))
            local_private_lengths.append(len(local_private))
            if not local_public:
                counts["empty_local_public"] += 1
            if not local_private:
                counts["empty_local_private"] += 1
            missing_public = set(local_public).difference(public_set)
            missing_private = set(local_private).difference(private_set)
            if missing_public:
                record_error(
                    f"{prefix}.public",
                    f"{len(missing_public)} local triples are absent from global public",
                )
            if missing_private:
                record_error(
                    f"{prefix}.private",
                    f"{len(missing_private)} local triples are absent from global privacy",
                )

    counts["records_with_errors"] = len(error_records)
    document_count = counts["document_count"]
    report = {
        "schema": "eraser4rag-rl-validation-v1",
        "data_path": str(path.resolve()),
        # The displayed errors are capped, but validity must reflect every
        # failing record (including when max_error_examples is zero).
        "valid": not error_records,
        "require_qa": require_qa,
        "counts": dict(counts),
        "statistics": {
            "contexts_per_query": _summary(contexts_per_query),
            "local_public_triples_per_document": _summary(local_public_lengths),
            "local_private_triples_per_document": _summary(local_private_lengths),
            "empty_local_public_ratio": (
                counts["empty_local_public"] / document_count if document_count else 0.0
            ),
            "empty_local_private_ratio": (
                counts["empty_local_private"] / document_count if document_count else 0.0
            ),
            "malformed_record_ratio": (
                len(error_records) / counts["query_count"] if counts["query_count"] else 0.0
            ),
        },
        "errors": errors,
    }
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RL Dataset Validation Report",
        "",
        f"- Status: **{'PASS' if report['valid'] else 'FAIL'}**",
        f"- Data: `{report['data_path']}`",
        f"- Formal QA fields required: {report['require_qa']}",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(report["counts"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Statistics",
        "",
        "```json",
        json.dumps(report["statistics"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Error Examples",
        "",
    ]
    if report["errors"]:
        lines.extend(
            f"- record {item['record']} `{item['location']}`: {item['message']}"
            for item in report["errors"]
        )
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = Path(report["data_path"]).stem
    json_path = directory / f"{stem}_rl_dataset_report.json"
    markdown_path = directory / f"{stem}_rl_dataset_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs/validation")
    parser.add_argument("--require-qa", action="store_true")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-error-examples", type=int, default=100)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_rl_dataset(
        args.data,
        require_qa=args.require_qa,
        max_records=args.max_records,
        max_error_examples=args.max_error_examples,
    )
    json_path, markdown_path = write_report(report, args.output_dir)
    print(f"RL validation: {'PASS' if report['valid'] else 'FAIL'}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
