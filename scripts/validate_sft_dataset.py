"""Validate the author-provided flattened SFT JSONL dataset.

This validator is intentionally model-free.  It checks the six-field schema,
triple encodings, local/global membership, and consecutive global-graph
grouping used by the PPO engineering smoke-data builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.triple_utils import TripleFormatError, parse_serialized_triple, serialize_triple


REQUIRED_FIELDS = {
    "public",
    "private",
    "text",
    "anonymized_text",
    "o_public",
    "o_private",
}


def _fingerprint(public: Any, private: Any) -> str:
    payload = json.dumps(
        [public, private], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _raw_triples(
    value: Any, field: str, line_number: int, add_error
) -> list[tuple[str, str, str]]:
    if not isinstance(value, list):
        add_error(line_number, field, "must be a list")
        return []
    triples: list[tuple[str, str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            add_error(line_number, f"{field}[{index}]", "must contain exactly 3 items")
            continue
        if not all(isinstance(part, str) for part in item):
            add_error(line_number, f"{field}[{index}]", "all triple parts must be strings")
            continue
        normalized = (item[0], item[1], item[2])
        try:
            serialize_triple(normalized)
        except TripleFormatError as exc:
            add_error(line_number, f"{field}[{index}]", str(exc))
            continue
        triples.append(normalized)
    return triples


def _tagged_triples(
    value: Any, field: str, line_number: int, add_error
) -> list[tuple[str, str, str]]:
    if not isinstance(value, list):
        add_error(line_number, field, "must be a list")
        return []
    triples: list[tuple[str, str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            add_error(line_number, f"{field}[{index}]", "must be a tagged string")
            continue
        try:
            parsed = tuple(parse_serialized_triple(item))
        except (TypeError, ValueError) as exc:
            add_error(line_number, f"{field}[{index}]", f"malformed tagged triple: {exc}")
            continue
        if len(parsed) != 3 or not all(isinstance(part, str) for part in parsed):
            add_error(line_number, f"{field}[{index}]", "parsed triple is invalid")
            continue
        if serialize_triple(parsed) != item:
            add_error(line_number, f"{field}[{index}]", "tagged triple is not canonical")
            continue
        triples.append((parsed[0], parsed[1], parsed[2]))
    return triples


def validate_sft_dataset(
    data_path: str | Path,
    *,
    max_records: int | None = None,
    max_error_examples: int = 100,
) -> dict[str, Any]:
    """Return a JSON-serializable validation report."""

    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"SFT dataset not found: {path}")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive when provided")
    if max_error_examples < 0:
        raise ValueError("max_error_examples must be non-negative")

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    error_lines: set[int] = set()

    def add_error(line: int, field: str, message: str) -> None:
        error_lines.add(line)
        if len(errors) < max_error_examples:
            errors.append({"line": line, "field": field, "message": message})

    counts: Counter[str] = Counter()
    text_lengths: list[int] = []
    anonymized_lengths: list[int] = []
    global_public_lengths: list[int] = []
    global_private_lengths: list[int] = []
    local_public_lengths: list[int] = []
    local_private_lengths: list[int] = []

    last_fingerprint: str | None = None
    current_segment_size = 0
    group_segment_sizes: list[int] = []
    closed_groups: set[str] = set()
    unique_groups: set[str] = set()
    non_contiguous_groups: set[str] = set()

    with path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if max_records is not None and counts["records_total"] >= max_records:
                counts["truncated_by_max_records"] = 1
                break
            counts["records_total"] += 1
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                add_error(line_number, "<record>", f"invalid JSON: {exc.msg}")
                continue
            if not isinstance(record, dict):
                add_error(line_number, "<record>", "record must be an object")
                continue

            missing = REQUIRED_FIELDS.difference(record)
            if missing:
                add_error(line_number, "<record>", f"missing fields: {sorted(missing)}")
                continue
            extra = set(record).difference(REQUIRED_FIELDS)
            if extra and len(warnings) < max_error_examples:
                warnings.append(
                    {"line": line_number, "field": "<record>", "message": f"extra fields: {sorted(extra)}"}
                )

            public = _raw_triples(record["public"], "public", line_number, add_error)
            private = _raw_triples(record["private"], "private", line_number, add_error)
            local_public = _tagged_triples(record["o_public"], "o_public", line_number, add_error)
            local_private = _tagged_triples(record["o_private"], "o_private", line_number, add_error)

            for text_field in ("text", "anonymized_text"):
                if not isinstance(record[text_field], str):
                    add_error(line_number, text_field, "must be a string")
                elif not record[text_field].strip():
                    add_error(line_number, text_field, "must not be empty")

            public_set = set(public)
            private_set = set(private)
            local_public_set = set(local_public)
            local_private_set = set(local_private)
            if public_set & private_set:
                add_error(line_number, "public/private", "global public and private triples overlap")
            if local_public_set & local_private_set:
                add_error(line_number, "o_public/o_private", "local public and private triples overlap")
            missing_public = local_public_set.difference(public_set)
            missing_private = local_private_set.difference(private_set)
            if missing_public:
                add_error(
                    line_number,
                    "o_public",
                    f"{len(missing_public)} local public triples are absent from global public",
                )
            if missing_private:
                add_error(
                    line_number,
                    "o_private",
                    f"{len(missing_private)} local private triples are absent from global private",
                )

            counts["duplicate_global_public"] += len(public) - len(public_set)
            counts["duplicate_global_private"] += len(private) - len(private_set)
            counts["duplicate_local_public"] += len(local_public) - len(local_public_set)
            counts["duplicate_local_private"] += len(local_private) - len(local_private_set)
            for field, collection in (
                ("empty_global_public", public),
                ("empty_global_private", private),
                ("empty_local_public", local_public),
                ("empty_local_private", local_private),
            ):
                if not collection:
                    counts[field] += 1

            if isinstance(record["text"], str):
                text_lengths.append(len(record["text"]))
            if isinstance(record["anonymized_text"], str):
                anonymized_lengths.append(len(record["anonymized_text"]))
            global_public_lengths.append(len(public))
            global_private_lengths.append(len(private))
            local_public_lengths.append(len(local_public))
            local_private_lengths.append(len(local_private))

            fingerprint = _fingerprint(record["public"], record["private"])
            unique_groups.add(fingerprint)
            if last_fingerprint is None:
                last_fingerprint = fingerprint
                current_segment_size = 1
            elif fingerprint == last_fingerprint:
                current_segment_size += 1
            else:
                group_segment_sizes.append(current_segment_size)
                closed_groups.add(last_fingerprint)
                if fingerprint in closed_groups:
                    non_contiguous_groups.add(fingerprint)
                last_fingerprint = fingerprint
                current_segment_size = 1

    if last_fingerprint is not None:
        group_segment_sizes.append(current_segment_size)

    counts["records_with_errors"] = len(error_lines)
    counts["records_valid"] = counts["records_total"] - counts["records_with_errors"]
    report: dict[str, Any] = {
        "schema": "eraser4rag-sft-validation-v1",
        "data_path": str(path.resolve()),
        "file_size_bytes": path.stat().st_size,
        # Validity is based on all failing records, not on the truncated list
        # of examples retained for the report.  In particular,
        # ``max_error_examples=0`` must never turn a failing dataset into PASS.
        "valid": not error_lines,
        "counts": dict(counts),
        "groups": {
            "unique_fingerprint_count": len(unique_groups),
            "consecutive_segment_count": len(group_segment_sizes),
            "non_contiguous_fingerprint_count": len(non_contiguous_groups),
            "documents_per_segment": _summary(group_segment_sizes),
        },
        "lengths": {
            "text_chars": _summary(text_lengths),
            "anonymized_text_chars": _summary(anonymized_lengths),
            "global_public_triples": _summary(global_public_lengths),
            "global_private_triples": _summary(global_private_lengths),
            "local_public_triples": _summary(local_public_lengths),
            "local_private_triples": _summary(local_private_lengths),
        },
        "errors": errors,
        "warnings": warnings,
        "notes": [
            "Groups are inferred from consecutive, order-sensitive (public, private) fingerprints.",
            "This flattened dataset has no authoritative query/group identifier.",
        ],
    }
    return report


def _markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    groups = report["groups"]
    status = "PASS" if report["valid"] else "FAIL"
    lines = [
        "# SFT Dataset Validation Report",
        "",
        f"- Status: **{status}**",
        f"- Data: `{report['data_path']}`",
        f"- Records: {counts.get('records_total', 0)}",
        f"- Valid records: {counts.get('records_valid', 0)}",
        f"- Records with errors: {counts.get('records_with_errors', 0)}",
        f"- Unique global-graph fingerprints: {groups['unique_fingerprint_count']}",
        f"- Consecutive group segments: {groups['consecutive_segment_count']}",
        f"- Non-contiguous repeated fingerprints: {groups['non_contiguous_fingerprint_count']}",
        "",
        "## Empty and Duplicate Counts",
        "",
        "```json",
        json.dumps(counts, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Length Statistics",
        "",
        "```json",
        json.dumps(report["lengths"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Error Examples",
        "",
    ]
    if report["errors"]:
        for error in report["errors"]:
            lines.append(f"- line {error['line']} `{error['field']}`: {error['message']}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The inferred grouping is an engineering reconstruction based on repeated global graphs; it is not an author-provided query ID.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "sft_dataset_report.json"
    markdown_path = directory / "sft_dataset_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="dataset/constructed_dataset/popqa_10_25_filtered_new.jsonl",
        help="Author SFT JSONL file.",
    )
    parser.add_argument("--output-dir", default="outputs/validation")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-error-examples", type=int, default=100)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_sft_dataset(
        args.data,
        max_records=args.max_records,
        max_error_examples=args.max_error_examples,
    )
    json_path, markdown_path = write_report(report, args.output_dir)
    print(f"SFT validation: {'PASS' if report['valid'] else 'FAIL'}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
