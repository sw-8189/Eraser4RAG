"""Attach document-local public/private references to sampled RL records."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import ChainMap, Counter
from collections.abc import Iterator, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

try:
    from utils.triple_utils import (
        normalize_triple_collection,
        parse_serialized_triple,
        serialize_triple,
    )
except ModuleNotFoundError as exc:  # Support ``python utils/add_sampled_data.py``.
    if exc.name != "utils":
        raise
    from triple_utils import (  # type: ignore[no-redef]
        normalize_triple_collection,
        parse_serialized_triple,
        serialize_triple,
    )


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


def _canonical_serialized(item: Any) -> str:
    if isinstance(item, str):
        return serialize_triple(parse_serialized_triple(item))
    return serialize_triple(item)


def _serialized_triples(items: Sequence[Any]) -> list[str]:
    normalized = normalize_triple_collection(items)
    serialized: list[str] = []
    seen: set[str] = set()
    for item in normalized:
        value = _canonical_serialized(item)
        if value not in seen:
            serialized.append(value)
            seen.add(value)
    return serialized


def process_private_trps(items: Sequence[Any]) -> dict[str, str]:
    """Compatibility replacement for the missing released-code helper.

    The caller needs canonical membership keys and serialized string values so
    that ``ctx['public/private']`` remains hashable in ``RL_train.py``.
    """

    return {value: value for value in _serialized_triples(items)}


def organize_triplets(data_each: dict[str, Any]) -> dict[str, Sequence[Any]]:
    triplet_maps = data_each.get("triplets")
    if not isinstance(triplet_maps, list):
        raise TypeError("The extraction record must contain list-valued 'triplets'")
    merged = dict(ChainMap(*triplet_maps)) if triplet_maps else {}
    return {str(doc_id): triples for doc_id, triples in merged.items()}


def match_triplets(
    triplets_dict: dict[str, Sequence[Any]],
    public_items: Sequence[Any],
    private_items: Sequence[Any],
    contexts: list[dict[str, Any]],
    stats: Counter[str] | None = None,
) -> list[dict[str, Any]]:
    stats = stats if stats is not None else Counter()
    private_keys = set(process_private_trps(private_items))
    public_keys = set(process_private_trps(public_items))

    for context_index, context in enumerate(contexts):
        if "id" not in context:
            raise KeyError(f"Context {context_index} has no id")
        doc_id = str(context["id"])
        if doc_id not in triplets_dict:
            raise KeyError(f"No extracted triples found for context id {doc_id!r}")

        # [RECONSTRUCTION] Preserve the extraction order.  The released code
        # used set intersections, making output order depend on hash seeding.
        document_triples = _serialized_triples(triplets_dict[doc_id])
        context["private"] = [
            triple for triple in document_triples if triple in private_keys
        ]
        context["public"] = [
            triple for triple in document_triples if triple in public_keys
        ]
        stats["contexts"] += 1
        stats["local_private"] += len(context["private"])
        stats["local_public"] += len(context["public"])
        if not context["private"]:
            stats["empty_local_private"] += 1
        if not context["public"]:
            stats["empty_local_public"] += 1
    return contexts


def _context_ids(record: dict[str, Any]) -> list[str]:
    contexts = record.get("ctxs")
    if not isinstance(contexts, list):
        raise TypeError("Record has no list-valued 'ctxs'")
    return [str(context["id"]) for context in contexts if "id" in context]


def _validate_alignment(
    extraction_record: dict[str, Any], sampled_record: dict[str, Any], record_number: int
) -> None:
    for field in ("question", "answers"):
        if field in extraction_record and field in sampled_record:
            if extraction_record[field] != sampled_record[field]:
                raise ValueError(
                    f"Record {record_number} is misaligned: field {field!r} differs"
                )
    sampled_ids = _context_ids(sampled_record)
    extraction_ids = set(_context_ids(extraction_record))
    missing = [doc_id for doc_id in sampled_ids if doc_id not in extraction_ids]
    if missing:
        raise ValueError(
            f"Record {record_number} is misaligned; missing context IDs: {missing[:5]}"
        )


def main(args: argparse.Namespace) -> dict[str, int]:
    paths = [args.triples_input, args.sampled_input, args.output_path]
    if os.path.abspath(paths[2]) in {os.path.abspath(paths[0]), os.path.abspath(paths[1])}:
        raise ValueError("Output path must differ from both input paths")

    extraction_iter = iter_data(args.triples_input, max_samples=args.max_samples)
    sampled_iter = iter_data(args.sampled_input, max_samples=args.max_samples)
    sentinel = object()
    totals: Counter[str] = Counter()
    _safe_parent(args.output_path)

    with open(args.output_path, "w", encoding="utf-8") as fout:
        for record_number, pair in enumerate(
            zip_longest(extraction_iter, sampled_iter, fillvalue=sentinel), start=1
        ):
            extraction_record, sampled_record = pair
            if extraction_record is sentinel or sampled_record is sentinel:
                raise ValueError(
                    "Extraction and sampled inputs have different record counts"
                )
            assert isinstance(extraction_record, dict)
            assert isinstance(sampled_record, dict)
            _validate_alignment(extraction_record, sampled_record, record_number)

            triplets_dict = organize_triplets(extraction_record)
            public_items = sampled_record.get("public")
            private_items = sampled_record.get("privacy")
            contexts = sampled_record.get("ctxs")
            if not isinstance(public_items, list) or not isinstance(private_items, list):
                raise TypeError(
                    f"Record {record_number} must contain list-valued public/privacy"
                )
            if not isinstance(contexts, list):
                raise TypeError(f"Record {record_number} has no list-valued ctxs")

            sampled_record["ctxs"] = match_triplets(
                triplets_dict,
                public_items,
                private_items,
                contexts,
                stats=totals,
            )
            json.dump(sampled_record, fout, ensure_ascii=False)
            fout.write("\n")
            totals["records"] += 1

    logger.info("Local triple mapping statistics: %s", dict(sorted(totals.items())))
    logger.info("Saved results to %s", args.output_path)
    return dict(totals)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--triples_input",
        "--data_0",
        dest="triples_input",
        default="./dataset/to_triplets/hotpotqa_with_triplets.jsonl",
    )
    parser.add_argument(
        "--sampled_input",
        "--data_1",
        dest="sampled_input",
        default="./dataset/sample_privacy/hotpotqa_10_25_new.jsonl",
    )
    parser.add_argument(
        "--output",
        "--output_dir",
        dest="output_path",
        default="./dataset/sample_privacy/hotpotqa_10_25_trp.jsonl",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
