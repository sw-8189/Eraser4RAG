"""Evaluate private/public triple retention on D_special rewrites."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import torch
    from relik import Relik
except ModuleNotFoundError:
    torch = None
    Relik = None

from utils.triple_utils import (
    normalize_triple_collection,
    parse_serialized_triple,
    serialize_triple,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> list[dict[str, Any]]:
    suffix = Path(data_path).suffix.lower()
    if suffix == ".json":
        with open(data_path, "r", encoding="utf-8") as fin:
            data = json.load(fin)
        if not isinstance(data, list):
            raise TypeError(f"Expected a JSON array in {data_path}")
        return data
    if suffix == ".jsonl":
        data: list[dict[str, Any]] = []
        with open(data_path, "r", encoding="utf-8") as fin:
            for line_number, line in enumerate(fin, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError(f"Line {line_number} in {data_path} is not an object")
                data.append(record)
        return data
    raise ValueError(f"Unsupported input format: {data_path}; expected .json or .jsonl")


def _canonical_references(items: list[Any]) -> set[str]:
    references: set[str] = set()
    for item in normalize_triple_collection(items):
        if isinstance(item, str):
            item = parse_serialized_triple(item)
        references.add(serialize_triple(item))
    return references


def _predict_batch(relik: Any, texts: list[str]) -> list[Any]:
    if not texts:
        return []
    raw_outputs = relik(texts[0]) if len(texts) == 1 else relik(texts)
    if len(texts) == 1 and hasattr(raw_outputs, "triplets"):
        return [raw_outputs]
    outputs = list(raw_outputs)
    if len(outputs) != len(texts):
        raise RuntimeError(
            f"ReLiK returned {len(outputs)} outputs for {len(texts)} texts"
        )
    return outputs


def _predicted_triples(output: Any) -> set[str]:
    predicted: set[str] = set()
    for triplet in output.triplets:
        # [AUTHOR-CODE] This evaluator removes only exact duplicates/self-loops;
        # unlike the extraction script it does not collapse unordered pairs.
        if triplet.subject.text == triplet.object.text:
            continue
        predicted.add(
            serialize_triple(
                (triplet.subject.text, triplet.label, triplet.object.text)
            )
        )
    return predicted


def _anonymized_contexts(record: dict[str, Any]) -> list[str] | None:
    value = record.get("anonymized_text", record.get("anonymized_ctxs"))
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not all(isinstance(text, str) for text in value):
        raise TypeError("anonymized_text/anonymized_ctxs must be a string list")
    return value


def main(args: argparse.Namespace) -> dict[str, float | int]:
    if torch is None or Relik is None:
        raise RuntimeError("Torch/ReLiK are unavailable; use eraser-main")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError("max_samples must be non-negative or None")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    relik = Relik.from_pretrained(
        args.relik_model,
        device=args.device,
        use_nme=args.use_nme,
    )
    data = load_data(args.data_path)
    if args.max_samples is not None and args.max_samples < len(data):
        data = random.Random(args.seed).sample(data, args.max_samples)

    totals: Counter[str] = Counter()
    sum_public_ratio = 0.0
    sum_private_ratio = 0.0
    with torch.no_grad():
        for record_number, record in enumerate(data, start=1):
            contexts = record.get("ctxs")
            anonymized_contexts = _anonymized_contexts(record)
            if not isinstance(contexts, list) or anonymized_contexts is None:
                totals["records_skipped"] += 1
                continue
            if len(anonymized_contexts) != len(contexts):
                totals["alignment_mismatch"] += 1
                continue

            for start in range(0, len(contexts), args.batch_size):
                context_batch = contexts[start : start + args.batch_size]
                text_batch = anonymized_contexts[start : start + args.batch_size]
                outputs = _predict_batch(relik, [text for text in text_batch if text != ""])
                output_iter = iter(outputs)
                for relative_index, (context, text) in enumerate(
                    zip(context_batch, text_batch)
                ):
                    offset = start + relative_index
                    if not isinstance(context, dict):
                        raise TypeError(
                            f"Record {record_number} ctx {offset} is not an object"
                        )
                    private_items = context.get("private")
                    public_items = context.get("public")
                    if not isinstance(private_items, list) or not isinstance(
                        public_items, list
                    ):
                        raise TypeError(
                            f"Record {record_number} ctx {offset} has invalid references"
                        )

                    predicted = (
                        set() if text == "" else _predicted_triples(next(output_iter))
                    )
                    private = _canonical_references(private_items)
                    public = _canonical_references(public_items)
                    private_ratio = (
                        len(predicted & private) / len(private) if private else 0.0
                    )
                    public_ratio = (
                        len(predicted & public) / len(public) if public else 1.0
                    )
                    sum_private_ratio += private_ratio
                    sum_public_ratio += public_ratio
                    totals["documents"] += 1
                    if not private:
                        totals["empty_private_reference"] += 1
                    if not public:
                        totals["empty_public_reference"] += 1
                    logger.debug(
                        "record=%d ctx=%d r_pub=%.6f r_pri=%.6f",
                        record_number,
                        offset,
                        public_ratio,
                        private_ratio,
                    )

            totals["records"] += 1
            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    denominator = totals["documents"]
    if denominator == 0:
        logger.warning("No aligned documents were evaluated")
    result: dict[str, float | int] = dict(sorted(totals.items()))
    result["r_pub"] = sum_public_ratio / denominator if denominator else 0.0
    result["r_pri"] = sum_private_ratio / denominator if denominator else 0.0
    logger.info("D_special evaluation: %s", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        "--data_path",
        dest="data_path",
        default="./outputs/rewritten/popqa_10_25_special.jsonl",
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
