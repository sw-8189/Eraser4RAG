"""Measure current ReLiK recovery of the author's original local triples.

This is the mandatory engineering gate before full PPO.  The fixed references
are ``o_public`` and ``o_private`` extracted from each original ``text``.  The
rewritten teacher text and global graphs are deliberately not used as fixed
references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.triple_utils import parse_serialized_triple, serialize_triple


def count_jsonl_records(path: Path) -> int:
    with path.open("r", encoding="utf-8") as source:
        return sum(1 for _ in source)


def choose_sample_indices(
    *, total_records: int, sample_size: int, seed: int
) -> list[int]:
    if total_records <= 0:
        raise ValueError("dataset is empty")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    actual_size = min(sample_size, total_records)
    return sorted(random.Random(seed).sample(range(total_records), actual_size))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_or_create_sample_indices(
    data_path: Path,
    output_path: Path,
    *,
    total_records: int,
    sample_size: int,
    seed: int,
    resample: bool = False,
) -> list[int]:
    data_sha256 = sha256_file(data_path)
    if output_path.exists() and not resample:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid sample index file: {output_path}")
        indices = payload.get("indices")
        if not isinstance(indices, list) or not all(isinstance(index, int) for index in indices):
            raise ValueError(f"invalid sample index file: {output_path}")
        expected_size = min(sample_size, total_records)
        if len(indices) != expected_size:
            raise ValueError(
                f"existing sample index file has {len(indices)} entries, expected {expected_size}; "
                "use a different output directory or --resample"
            )
        if any(index < 0 or index >= total_records for index in indices):
            raise ValueError("existing sample indices are outside the current dataset")
        if len(set(indices)) != len(indices):
            raise ValueError("existing sample indices contain duplicates")
        expected_metadata = {
            "data_path": str(data_path.resolve()),
            "data_size_bytes": data_path.stat().st_size,
            "data_sha256": data_sha256,
            "total_records": total_records,
            "requested_sample_size": sample_size,
            "actual_sample_size": expected_size,
            "seed": seed,
        }
        for key, expected in expected_metadata.items():
            if payload.get(key) != expected:
                raise ValueError(
                    f"sample index manifest field {key!r} does not match the current run; "
                    "use a different output directory or --resample"
                )
        return sorted(indices)

    indices = choose_sample_indices(
        total_records=total_records, sample_size=sample_size, seed=seed
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "eraser4rag-relik-sample-indices-v1",
        "data_path": str(data_path.resolve()),
        "data_size_bytes": data_path.stat().st_size,
        "data_sha256": data_sha256,
        "total_records": total_records,
        "requested_sample_size": sample_size,
        "actual_sample_size": len(indices),
        "seed": seed,
        "indices": indices,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return indices


def load_sampled_records(path: Path, indices: list[int]) -> list[tuple[int, dict[str, Any]]]:
    wanted = set(indices)
    records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as source:
        for zero_index, line in enumerate(source):
            if zero_index not in wanted:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {zero_index + 1}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"record at line {zero_index + 1} is not an object")
            records.append((zero_index, record))
            if len(records) == len(indices):
                break
    if len(records) != len(indices):
        raise ValueError("could not load every saved sample index")
    return records


def _canonical_reference(items: Any, field: str, index: int) -> set[str]:
    if not isinstance(items, list):
        raise ValueError(f"sample {index} field {field!r} must be a list")
    result: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            raise ValueError(f"sample {index} field {field!r} contains a non-string")
        parsed = tuple(parse_serialized_triple(item))
        canonical = serialize_triple(parsed)
        if canonical != item:
            raise ValueError(f"sample {index} field {field!r} contains non-canonical data")
        result.add(canonical)
    return result


def relik_output_to_triples(output: Any, *, deduplicate_entity_pairs: bool = True) -> set[str]:
    """Convert one ReLiK output to author-compatible neutral strings."""

    if isinstance(output, (list, tuple)):
        if len(output) != 1:
            raise ValueError("expected exactly one ReLiK output for one text")
        output = output[0]
    triplets = getattr(output, "triplets", None)
    if triplets is None:
        raise TypeError("ReLiK output has no 'triplets' attribute")
    result: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for triple in triplets:
        subject_obj = getattr(triple, "subject", None)
        object_obj = getattr(triple, "object", None)
        subject = getattr(subject_obj, "text", subject_obj)
        object_ = getattr(object_obj, "text", object_obj)
        relation = getattr(triple, "label", None)
        if not all(isinstance(value, str) for value in (subject, relation, object_)):
            continue
        if subject == object_:
            continue
        pair = (subject, object_)
        reverse_pair = (object_, subject)
        if deduplicate_entity_pairs and (pair in seen_pairs or reverse_pair in seen_pairs):
            continue
        seen_pairs.add(pair)
        result.add(serialize_triple((subject, relation, object_)))
    return result


def evaluate_records(
    records: list[tuple[int, dict[str, Any]]],
    extractor: Callable[[str], Any],
    *,
    mismatch_text_chars: int = 400,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    public_hits = public_total = private_hits = private_total = 0
    public_macro: list[float] = []
    private_macro: list[float] = []
    nonempty_reference_docs = zero_match_docs = fully_recovered_docs = 0
    mismatches: list[dict[str, Any]] = []

    for dataset_index, record in records:
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError(f"sample {dataset_index} has no string 'text'")
        public = _canonical_reference(record.get("o_public"), "o_public", dataset_index)
        private = _canonical_reference(record.get("o_private"), "o_private", dataset_index)
        predicted = relik_output_to_triples(extractor(text))

        hit_public = public & predicted
        hit_private = private & predicted
        public_hits += len(hit_public)
        private_hits += len(hit_private)
        public_total += len(public)
        private_total += len(private)
        if public:
            public_macro.append(len(hit_public) / len(public))
        if private:
            private_macro.append(len(hit_private) / len(private))

        reference_union = public | private
        hits_union = hit_public | hit_private
        if reference_union:
            nonempty_reference_docs += 1
            if not hits_union:
                zero_match_docs += 1
            if reference_union.issubset(predicted):
                fully_recovered_docs += 1

        missing_public = sorted(public - predicted)
        missing_private = sorted(private - predicted)
        extra_predicted = sorted(predicted - reference_union)
        if missing_public or missing_private:
            mismatches.append(
                {
                    "dataset_index": dataset_index,
                    "line_number": dataset_index + 1,
                    "text_excerpt": text[:mismatch_text_chars],
                    "reference_public_count": len(public),
                    "reference_private_count": len(private),
                    "predicted_count": len(predicted),
                    "missing_public": missing_public,
                    "missing_private": missing_private,
                    "extra_predicted": extra_predicted,
                }
            )

    metrics = {
        "sample_count": len(records),
        "public_reference_count": public_total,
        "private_reference_count": private_total,
        "public_hit_count": public_hits,
        "private_hit_count": private_hits,
        "public_micro_recall": public_hits / public_total if public_total else None,
        "private_micro_recall": private_hits / private_total if private_total else None,
        "public_macro_recall": statistics.fmean(public_macro) if public_macro else None,
        "private_macro_recall": statistics.fmean(private_macro) if private_macro else None,
        "macro_public_document_count": len(public_macro),
        "macro_private_document_count": len(private_macro),
        "nonempty_reference_document_count": nonempty_reference_docs,
        "zero_match_document_count": zero_match_docs,
        "zero_match_ratio": (
            zero_match_docs / nonempty_reference_docs if nonempty_reference_docs else None
        ),
        "fully_recovered_reference_document_count": fully_recovered_docs,
        "mismatch_document_count": len(mismatches),
    }
    mismatches.sort(
        key=lambda item: (len(item["missing_public"]) + len(item["missing_private"])),
        reverse=True,
    )
    return metrics, mismatches


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_relik(model: str, *, device: str, use_nme: bool):
    try:
        from relik import Relik
    except ImportError as exc:
        raise RuntimeError(
            "ReLiK is not installed. Run this gate in the pinned eraser-main environment."
        ) from exc
    return Relik.from_pretrained(model, device=device, use_nme=use_nme)


def _markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# ReLiK Consistency Report",
        "",
        f"- Gate status: **{report['gate_status'].upper()}**",
        f"- Model: `{report['relik_model']}`",
        f"- Device: `{report['device']}`",
        f"- Sample size: {metrics['sample_count']}",
        f"- Seed: {report['seed']}",
        f"- Engineering warning threshold: {report['engineering_warning_threshold']}",
        "",
        "> The threshold is an engineering warning threshold, not a paper-defined threshold.",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Gate Interpretation",
        "",
    ]
    if report["gate_status"] == "pass":
        lines.append("All defined public/private micro and macro recall metrics meet the engineering threshold.")
    else:
        lines.append("At least one defined recall metric is below the engineering threshold. Do not start full PPO until reviewed.")
    lines.extend(
        [
            "",
            "## Top Mismatch Examples",
            "",
        ]
    )
    if report["top_mismatch_examples"]:
        for item in report["top_mismatch_examples"]:
            lines.append(
                f"- dataset index {item['dataset_index']}: missing public={len(item['missing_public'])}, "
                f"missing private={len(item['missing_private'])}, extra predicted={len(item['extra_predicted'])}"
            )
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="dataset/constructed_dataset/popqa_10_25_filtered_new.jsonl",
    )
    parser.add_argument(
        "--relik-model", default="relik-ie/relik-relation-extraction-small"
    )
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--use-nme", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-dir", default="outputs/relik_consistency")
    parser.add_argument("--warning-threshold", type=float, default=0.80)
    parser.add_argument("--top-mismatches", type=int, default=20)
    parser.add_argument("--mismatch-text-chars", type=int, default=400)
    parser.add_argument("--resample", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.warning_threshold <= 1.0:
        raise ValueError("warning_threshold must be between 0 and 1")
    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"SFT dataset not found: {data_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_records = count_jsonl_records(data_path)
    indices = load_or_create_sample_indices(
        data_path,
        output_dir / "sample_indices.json",
        total_records=total_records,
        sample_size=args.sample_size,
        seed=args.seed,
        resample=args.resample,
    )
    records = load_sampled_records(data_path, indices)
    device = _resolve_device(args.device)
    relik = load_relik(args.relik_model, device=device, use_nme=args.use_nme)
    metrics, mismatches = evaluate_records(
        records, relik, mismatch_text_chars=args.mismatch_text_chars
    )
    recalls = [
        metrics[key]
        for key in (
            "public_micro_recall",
            "private_micro_recall",
            "public_macro_recall",
            "private_macro_recall",
        )
        if metrics[key] is not None
    ]
    gate_status = (
        "pass"
        if len(recalls) == 4 and all(value >= args.warning_threshold for value in recalls)
        else "warning"
    )
    report = {
        "schema": "eraser4rag-relik-consistency-v1",
        "data_path": str(data_path.resolve()),
        "relik_model": args.relik_model,
        "device": device,
        "use_nme": args.use_nme,
        "seed": args.seed,
        "sample_indices_path": str((output_dir / "sample_indices.json").resolve()),
        "engineering_warning_threshold": args.warning_threshold,
        "gate_status": gate_status,
        "metrics": metrics,
        "top_mismatch_examples": mismatches[: args.top_mismatches],
        "notes": [
            "0.80 is an engineering warning threshold, not a paper threshold.",
            "Fixed references are o_public/o_private from original text.",
            "Extra predicted triples can be legitimate triples excluded from the private/public partition.",
            "Current predictions use the released unordered entity-pair first-relation duplicate rule.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    with (output_dir / "mismatches.jsonl").open("w", encoding="utf-8", newline="\n") as output:
        for mismatch in mismatches:
            output.write(json.dumps(mismatch, ensure_ascii=False) + "\n")
    print(f"ReLiK consistency gate: {gate_status.upper()}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if gate_status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
