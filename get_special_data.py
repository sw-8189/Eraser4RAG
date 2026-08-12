"""Construct the D_special subset from document-local triple references."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from utils.triple_utils import parse_serialized_triple


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _safe_parent(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def iter_data(data_path: str, max_samples: int | None = None) -> Iterator[dict[str, Any]]:
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative or None")
    suffix = Path(data_path).suffix.lower()
    if suffix == ".json":
        with open(data_path, "r", encoding="utf-8") as fin:
            data = json.load(fin)
        if not isinstance(data, list):
            raise TypeError(f"Expected a JSON array in {data_path}")
        for index, record in enumerate(data):
            if max_samples is not None and index >= max_samples:
                break
            if not isinstance(record, dict):
                raise TypeError(f"Record {index} in {data_path} is not an object")
            yield record
        return
    if suffix == ".jsonl":
        with open(data_path, "r", encoding="utf-8") as fin:
            emitted = 0
            for line_number, line in enumerate(fin, start=1):
                if not line.strip():
                    continue
                if max_samples is not None and emitted >= max_samples:
                    break
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError(f"Line {line_number} in {data_path} is not an object")
                yield record
                emitted += 1
        return
    raise ValueError(f"Unsupported input format: {data_path}; expected .json or .jsonl")


def load_data(data_path: str, max_samples: int | None = None) -> list[dict[str, Any]]:
    return list(iter_data(data_path, max_samples=max_samples))


def trp2list(triple_text: str) -> list[str]:
    """Backward-compatible wrapper around the shared strict parser."""

    return list(parse_serialized_triple(triple_text))


def _anonymized_key(record: dict[str, Any]) -> str | None:
    if "anonymized_text" in record:
        return "anonymized_text"
    if "anonymized_ctxs" in record:
        return "anonymized_ctxs"
    return None


def select_special_ctx(each_data: dict[str, Any]) -> tuple[dict[str, Any] | None, int]:
    contexts = each_data.get("ctxs")
    if not isinstance(contexts, list):
        raise TypeError("Record has no list-valued 'ctxs'")

    anonymized_key = _anonymized_key(each_data)
    anonymized_contexts: list[Any] | None = None
    if anonymized_key is not None:
        value = each_data[anonymized_key]
        if not isinstance(value, list) or len(value) != len(contexts):
            return None, 0
        anonymized_contexts = value

    new_contexts: list[dict[str, Any]] = []
    new_anonymized_contexts: list[Any] = []
    for index, context in enumerate(contexts):
        if not isinstance(context, dict):
            raise TypeError(f"Context {index} is not an object")
        private_items = context.get("private")
        public_items = context.get("public")
        if not isinstance(private_items, list) or not isinstance(public_items, list):
            raise TypeError(f"Context {index} must contain list-valued public/private")

        private_by_tail: dict[str, set[tuple[str, str]]] = {}
        for item in private_items:
            head, relation, tail = parse_serialized_triple(item)
            private_by_tail.setdefault(tail, set()).add((head, relation))

        is_special = False
        for item in public_items:
            head, relation, tail = parse_serialized_triple(item)
            if tail in private_by_tail and (head, relation) not in private_by_tail[tail]:
                is_special = True
                break

        if is_special:
            new_contexts.append(context)
            if anonymized_contexts is not None:
                new_anonymized_contexts.append(anonymized_contexts[index])

    each_data["ctxs"] = new_contexts
    if anonymized_key is not None:
        each_data[anonymized_key] = new_anonymized_contexts
    return each_data, len(new_contexts)


def main(args: argparse.Namespace) -> dict[str, int]:
    if args.min_special_contexts < 1:
        raise ValueError("min_special_contexts must be at least 1")
    if os.path.abspath(args.input_path) == os.path.abspath(args.output_path):
        raise ValueError("Input and output paths must be different")

    totals: Counter[str] = Counter()
    _safe_parent(args.output_path)
    with open(args.output_path, "w", encoding="utf-8") as fout:
        for each_data in iter_data(args.input_path, max_samples=args.max_samples):
            totals["records_seen"] += 1
            new_data, count = select_special_ctx(each_data)
            if new_data is None:
                totals["alignment_mismatch"] += 1
                continue
            totals["special_contexts"] += count
            if count >= args.min_special_contexts:
                json.dump(new_data, fout, ensure_ascii=False)
                fout.write("\n")
                totals["records_written"] += 1

    logger.info("D_special construction statistics: %s", dict(sorted(totals.items())))
    logger.info("Saved results to %s", args.output_path)
    return dict(totals)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "--input_file",
        dest="input_path",
        default="./dataset/sample_privacy/popqa_10_25_trp.jsonl",
    )
    parser.add_argument(
        "--output",
        "--output_dir",
        dest="output_path",
        default="./dataset/special_data/popqa_10_25_special.jsonl",
    )
    parser.add_argument("--min_special_contexts", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
