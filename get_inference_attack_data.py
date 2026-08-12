"""Construct records susceptible to cross-document inference attacks."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import ChainMap, Counter, defaultdict, deque
from collections.abc import Iterator, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

from tqdm import tqdm

from utils.triple_utils import normalize_triple_collection, parse_serialized_triple


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _safe_parent(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def iter_data(data_path: str) -> Iterator[dict[str, Any]]:
    suffix = Path(data_path).suffix.lower()
    if suffix == ".json":
        with open(data_path, "r", encoding="utf-8") as fin:
            data = json.load(fin)
        if not isinstance(data, list):
            raise TypeError(f"Expected a JSON array in {data_path}")
        for index, record in enumerate(data):
            if not isinstance(record, dict):
                raise TypeError(f"Record {index} in {data_path} is not an object")
            yield record
        return
    if suffix == ".jsonl":
        with open(data_path, "r", encoding="utf-8") as fin:
            for line_number, line in enumerate(fin, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError(f"Line {line_number} in {data_path} is not an object")
                yield record
        return
    raise ValueError(f"Unsupported input format: {data_path}; expected .json or .jsonl")


def load_data(data_path: str) -> list[dict[str, Any]]:
    return list(iter_data(data_path))


def trp2list(triple_text: str) -> list[str]:
    return list(parse_serialized_triple(triple_text))


def organize_triplets(data_each: dict[str, Any]) -> dict[str, Sequence[Any]]:
    triplet_maps = data_each.get("triplets")
    if not isinstance(triplet_maps, list):
        raise TypeError("Extraction record must contain list-valued 'triplets'")
    merged = dict(ChainMap(*triplet_maps)) if triplet_maps else {}
    return {str(doc_id): triples for doc_id, triples in merged.items()}


def _as_triple(item: Any) -> tuple[str, str, str]:
    if isinstance(item, str):
        return tuple(parse_serialized_triple(item))  # type: ignore[return-value]
    if len(item) != 3 or not all(isinstance(value, str) for value in item):
        raise TypeError(f"Malformed triple: {item!r}")
    return item[0], item[1], item[2]


def to_edge(triples: Sequence[Any]) -> list[list[str]]:
    return [
        [triple[0], triple[2]]
        for triple in (_as_triple(item) for item in normalize_triple_collection(triples))
    ]


def supplement(left: Sequence[list[str]], right: Sequence[list[str]]) -> list[list[str]]:
    return [edge for edge in left if edge not in right]


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


def _context_ids(record: dict[str, Any]) -> set[str]:
    contexts = record.get("ctxs")
    if not isinstance(contexts, list):
        raise TypeError("Record has no list-valued 'ctxs'")
    return {str(context["id"]) for context in contexts if "id" in context}


def _validate_alignment(
    extraction_record: dict[str, Any], sampled_record: dict[str, Any], record_number: int
) -> None:
    for field in ("question", "answers"):
        if field in extraction_record and field in sampled_record:
            if extraction_record[field] != sampled_record[field]:
                raise ValueError(
                    f"Record {record_number} is misaligned: field {field!r} differs"
                )
    missing = _context_ids(sampled_record) - _context_ids(extraction_record)
    if missing:
        raise ValueError(
            f"Record {record_number} is missing extracted context IDs: {sorted(missing)[:5]}"
        )


def _resolve_output_path(args: argparse.Namespace) -> str:
    return args.output_path or os.path.join(
        args.output_dir, "popqa_10_25_inferattack.jsonl"
    )


def main(args: argparse.Namespace) -> dict[str, int]:
    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError("max_samples must be non-negative or None")
    output_path = _resolve_output_path(args)
    input_paths = {os.path.abspath(args.triples_input), os.path.abspath(args.sampled_input)}
    if os.path.abspath(output_path) in input_paths:
        raise ValueError("Output path must differ from both input paths")

    extraction_iter = iter_data(args.triples_input)
    sampled_iter = iter_data(args.sampled_input)
    sentinel = object()
    totals: Counter[str] = Counter()
    _safe_parent(output_path)

    paired_records = zip_longest(extraction_iter, sampled_iter, fillvalue=sentinel)
    with open(output_path, "w", encoding="utf-8") as fout:
        for record_number, pair in enumerate(tqdm(paired_records), start=1):
            if args.max_samples is not None and totals["records_written"] >= args.max_samples:
                break
            extraction_record, sampled_record = pair
            if extraction_record is sentinel or sampled_record is sentinel:
                raise ValueError("Input files have different record counts")
            assert isinstance(extraction_record, dict)
            assert isinstance(sampled_record, dict)
            totals["records_seen"] += 1
            _validate_alignment(extraction_record, sampled_record, record_number)

            contexts = sampled_record.get("ctxs")
            if not isinstance(contexts, list):
                raise TypeError(f"Record {record_number} has no list-valued ctxs")
            for key in ("anonymized_text", "anonymized_ctxs"):
                if key in sampled_record:
                    anonymized = sampled_record[key]
                    if not isinstance(anonymized, list) or len(anonymized) != len(contexts):
                        totals["alignment_mismatch"] += 1
                        contexts = []
                    break
            if not contexts:
                continue

            triplet_payload = sampled_record.get("triplets")
            if not isinstance(triplet_payload, dict) or not isinstance(
                triplet_payload.get("trps"), list
            ):
                raise TypeError(f"Record {record_number} has invalid global triplets")
            privacy_items = sampled_record.get("privacy")
            if not isinstance(privacy_items, list):
                raise TypeError(f"Record {record_number} has invalid privacy triples")

            total_edges = to_edge(triplet_payload["trps"])
            privacy_edges = to_edge(privacy_items)
            unprivate_edges = supplement(total_edges, privacy_edges)
            triplets_dict = organize_triplets(extraction_record)
            selected = False

            for context_index, context in enumerate(contexts):
                if "id" not in context:
                    raise KeyError(f"Record {record_number} ctx {context_index} has no id")
                doc_id = str(context["id"])
                if doc_id not in triplets_dict:
                    raise KeyError(f"No extracted triples found for context id {doc_id!r}")
                local_private_items = context.get("private")
                if not isinstance(local_private_items, list):
                    raise TypeError(
                        f"Record {record_number} ctx {context_index} has invalid private triples"
                    )
                local_private_edges = to_edge(local_private_items)
                if not local_private_edges:
                    continue
                local_total_edges = to_edge(triplets_dict[doc_id])
                local_unprivate_edges = supplement(local_total_edges, local_private_edges)
                other_unprivate_graph = build_adjacency_list(
                    supplement(unprivate_edges, local_unprivate_edges)
                )
                other_private_graph = build_adjacency_list(
                    supplement(privacy_edges, local_private_edges)
                )

                for private_edge in local_private_edges:
                    disconnected_via_other_private = not are_nodes_connected(
                        other_private_graph, private_edge[0], private_edge[1]
                    )
                    connected_via_other_public = are_nodes_connected(
                        other_unprivate_graph, private_edge[0], private_edge[1]
                    )
                    if disconnected_via_other_private and connected_via_other_public:
                        json.dump(sampled_record, fout, ensure_ascii=False)
                        fout.write("\n")
                        totals["records_written"] += 1
                        selected = True
                        break
                if selected:
                    break

    logger.info("Inference-attack construction statistics: %s", dict(sorted(totals.items())))
    logger.info("Saved results to %s", output_path)
    return dict(totals)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--triples_input",
        "--data_0",
        dest="triples_input",
        default="./dataset/to_triplets/popqa_with_triplets.jsonl",
    )
    parser.add_argument(
        "--sampled_input",
        "--data_1",
        dest="sampled_input",
        default="./dataset/sample_privacy/popqa_10_25_trp.jsonl",
    )
    parser.add_argument("--output", dest="output_path", default=None)
    parser.add_argument("--output_dir", default="./dataset/inference_attack/")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of selected output records; None means no limit",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
