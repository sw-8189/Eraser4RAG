"""Extract per-document relation triples with ReLiK."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from relik import Relik


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
    """Backward-compatible materializing wrapper."""

    return list(iter_data(data_path, max_samples=max_samples))


def _batched(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _predict_batch(relik: Any, texts: list[str]) -> list[Any]:
    """Run ReLiK on one chunk while retaining the original single-text path."""

    if not texts:
        return []
    raw_outputs = relik(texts[0]) if len(texts) == 1 else relik(texts)
    if len(texts) == 1 and hasattr(raw_outputs, "triplets"):
        return [raw_outputs]
    outputs = list(raw_outputs)
    if len(outputs) != len(texts):
        raise RuntimeError(
            f"ReLiK returned {len(outputs)} outputs for a batch of {len(texts)} texts"
        )
    return outputs


def _filter_triplets(output: Any) -> tuple[list[tuple[str, str, str]], Counter[str]]:
    """Apply the released author's exact duplicate/unordered-pair rules."""

    true_triplets: list[tuple[str, str, str]] = []
    pairs: list[tuple[str, str]] = []
    stats: Counter[str] = Counter(raw=len(output.triplets))

    # [AUTHOR-CODE] Do not alter these semantics without a recorded deviation:
    # self loops are removed and only the first relation for an unordered entity
    # pair is retained.
    for triplet in output.triplets:
        triple = (triplet.subject.text, triplet.label, triplet.object.text)
        pair = (triple[0], triple[2])
        reverse_pair = (triple[2], triple[0])
        if triple in true_triplets:
            stats["dropped_exact_duplicate"] += 1
            continue
        if triple[0] == triple[2]:
            stats["dropped_self_loop"] += 1
            continue
        if pair in pairs or reverse_pair in pairs:
            stats["dropped_unordered_pair"] += 1
            continue
        true_triplets.append(triple)
        pairs.append(pair)

    stats["kept"] = len(true_triplets)
    return true_triplets, stats


def main(args: argparse.Namespace) -> dict[str, int]:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if os.path.abspath(args.input_path) == os.path.abspath(args.output_path):
        raise ValueError("Input and output paths must be different")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info(
        "Loading ReLiK model=%s device=%s use_nme=%s batch_size=%d",
        args.relik_model,
        args.device,
        args.use_nme,
        args.batch_size,
    )
    relik = Relik.from_pretrained(
        args.relik_model,
        device=args.device,
        use_nme=args.use_nme,
    )

    _safe_parent(args.output_path)
    totals: Counter[str] = Counter()
    with open(args.output_path, "w", encoding="utf-8") as fout, torch.no_grad():
        for record_number, data_each in enumerate(
            iter_data(args.input_path, max_samples=args.max_samples), start=1
        ):
            contexts = data_each.get("ctxs")
            if not isinstance(contexts, list):
                raise TypeError(f"Record {record_number} has no list-valued 'ctxs'")

            entries: list[tuple[str, str]] = []
            for context_index, document in enumerate(contexts):
                totals["documents_seen"] += 1
                if not isinstance(document, dict):
                    raise TypeError(
                        f"Record {record_number} ctx {context_index} is not an object"
                    )
                if "id" not in document:
                    totals["documents_without_id"] += 1
                    continue
                text = document.get("text")
                if not isinstance(text, str):
                    raise TypeError(
                        f"Record {record_number} ctx {context_index} has non-string text"
                    )
                # JSON object keys are strings; normalize proactively so the
                # downstream ctx-id lookup cannot diverge for integer IDs.
                entries.append((str(document["id"]), text))

            triplets_each: list[dict[str, list[tuple[str, str, str]]]] = []
            for chunk in _batched(entries, args.batch_size):
                nonempty = [(doc_id, text) for doc_id, text in chunk if text != ""]
                predictions = _predict_batch(relik, [text for _, text in nonempty])
                prediction_iter = iter(predictions)

                for doc_id, text in chunk:
                    if text == "":
                        triplets_each.append({doc_id: []})
                        totals["empty_documents"] += 1
                        continue
                    filtered, filter_stats = _filter_triplets(next(prediction_iter))
                    triplets_each.append({doc_id: filtered})
                    totals.update(filter_stats)

            data_each["triplets"] = triplets_each
            json.dump(data_each, fout, ensure_ascii=False)
            fout.write("\n")
            totals["records"] += 1
            if record_number % 20 == 0:
                logger.info("Processed %d records", record_number)
            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("ReLiK extraction statistics: %s", dict(sorted(totals.items())))
    logger.info("Saved results to %s", args.output_path)
    return dict(totals)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "--data",
        dest="input_path",
        default="./dataset/retrieved_data/popqa_retrieved_clean.jsonl",
    )
    parser.add_argument(
        "--output",
        "--output_dir",
        dest="output_path",
        default="./dataset/to_triplets/popqa_with_triplets.jsonl",
    )
    parser.add_argument(
        "--relik_model",
        default="relik-ie/relik-relation-extraction-small",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--use_nme",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
