"""Evaluate cross-document private-entity connectivity after rewriting."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict, deque
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

try:
    import torch
    from relik import Relik
except ModuleNotFoundError:
    torch = None
    Relik = None

from utils.triple_utils import normalize_triple_collection, parse_serialized_triple


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_adjacency_list(edges: Sequence[Sequence[str]]) -> defaultdict[str, list[str]]:
    adjacency_list: defaultdict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if len(edge) != 2:
            raise ValueError(f"Malformed edge: {edge!r}")
        u, v = edge
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
    return adjacency_list


def are_nodes_connected(
    adjacency_list: defaultdict[str, list[str]], start: str, end: str
) -> bool:
    visited: set[str] = set()
    queue: deque[str] = deque([start])
    while queue:
        current = queue.popleft()
        # [AUTHOR-CODE] Preserve released exact-or-substring entity matching.
        if current == end or end in current or current in end:
            return True
        if current not in visited:
            visited.add(current)
            for neighbor in adjacency_list.get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)
    return False


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
    return list(parse_serialized_triple(triple_text))


def _as_triple(item: Any) -> tuple[str, str, str]:
    if isinstance(item, str):
        return tuple(parse_serialized_triple(item))  # type: ignore[return-value]
    if len(item) != 3 or not all(isinstance(value, str) for value in item):
        raise TypeError(f"Malformed triple: {item!r}")
    return item[0], item[1], item[2]


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
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    relik = Relik.from_pretrained(
        args.relik_model,
        device=args.device,
        use_nme=args.use_nme,
    )

    totals: Counter[str] = Counter()
    sum_connection_ratio = 0.0
    with torch.no_grad():
        for record_number, record in enumerate(
            iter_data(args.data_path, max_samples=args.max_samples), start=1
        ):
            anonymized_contexts = _anonymized_contexts(record)
            if anonymized_contexts is None:
                totals["records_without_rewrites"] += 1
                continue
            privacy_items = record.get("privacy")
            if not isinstance(privacy_items, list):
                raise TypeError(f"Record {record_number} has invalid privacy triples")
            privacy = [
                _as_triple(item)
                for item in normalize_triple_collection(privacy_items)
            ]
            if not privacy:
                # r_connect has no defined denominator when G_private is empty.
                totals["records_empty_privacy"] += 1
                continue

            total_edges: list[list[str]] = []
            for start in range(0, len(anonymized_contexts), args.batch_size):
                text_batch = anonymized_contexts[start : start + args.batch_size]
                outputs = _predict_batch(
                    relik, [text for text in text_batch if text != ""]
                )
                for output in outputs:
                    for triplet in output.triplets:
                        edge = [triplet.subject.text, triplet.object.text]
                        # [AUTHOR-CODE] Preserve exact oriented-pair deduplication.
                        if edge not in total_edges and edge[0] != edge[1]:
                            total_edges.append(edge)

            graph = build_adjacency_list(total_edges)
            connected = sum(
                are_nodes_connected(graph, triple[0], triple[2])
                for triple in privacy
            )
            connection_ratio = connected / len(privacy)
            sum_connection_ratio += connection_ratio
            totals["records"] += 1
            totals["private_triples"] += len(privacy)
            totals["connected_private_triples"] += connected
            logger.debug(
                "record=%d r_connect=%.6f", record_number, connection_ratio
            )
            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    record_count = totals["records"]
    private_count = totals["private_triples"]
    result: dict[str, float | int] = dict(sorted(totals.items()))
    result["r_connect_macro"] = (
        sum_connection_ratio / record_count if record_count else 0.0
    )
    result["r_connect_micro"] = (
        totals["connected_private_triples"] / private_count if private_count else 0.0
    )
    if record_count == 0:
        logger.warning("No records with non-empty privacy references were evaluated")
    logger.info("Inference-attack evaluation: %s", result)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "eraser4rag-evaluation-v1",
            "evaluator": "test_inferattack.py",
            "data_path": str(Path(args.data_path).resolve()),
            "relik_model": args.relik_model,
            "device": args.device,
            "use_nme": args.use_nme,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "max_samples": args.max_samples,
            "metrics": result,
        }
        with output_path.open("w", encoding="utf-8") as fout:
            json.dump(payload, fout, ensure_ascii=False, indent=2, sort_keys=True)
            fout.write("\n")
        logger.info("Saved evaluation JSON to %s", output_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        "--data_path",
        dest="data_path",
        default="./outputs/rewritten/popqa_10_25_inferattack.jsonl",
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
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum input records; replaces the released fixed record cap",
    )
    parser.add_argument(
        "--output_json",
        "--output-json",
        dest="output_json",
        default=None,
        help="Optional machine-readable metrics output path",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
