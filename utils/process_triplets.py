"""Build global triples and split them into public/private graphs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import ChainMap, Counter, defaultdict, deque
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import spacy


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

Triple = list[str] | tuple[str, str, str]
Edge = list[str]


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


def build_adjacency_list(edges: Sequence[Sequence[str]]) -> defaultdict[str, list[str]]:
    adjacency_list: defaultdict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if len(edge) != 2:
            raise ValueError(f"Expected a two-node edge, got {edge!r}")
        u, v = edge
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
    return adjacency_list


def _author_entity_match(left: str, right: str) -> bool:
    """Released-code entity heuristic; intentionally case-sensitive."""

    # [AUTHOR-CODE] Preserve exact-or-substring matching for reproduction.
    return left == right or right in left or left in right


def are_nodes_connected(
    adjacency_list: defaultdict[str, list[str]], start: str, end: str
) -> bool:
    visited: set[str] = set()
    queue: deque[str] = deque([start])

    while queue:
        current = queue.popleft()
        if _author_entity_match(current, end):
            return True
        if current not in visited:
            visited.add(current)
            for neighbor in adjacency_list.get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)
    return False


def _validate_triple(triple: Sequence[Any], where: str) -> tuple[str, str, str]:
    if len(triple) != 3 or not all(isinstance(value, str) for value in triple):
        raise TypeError(f"Malformed triple in {where}: {triple!r}")
    return triple[0], triple[1], triple[2]


def merge_triplets(
    data_each: dict[str, Any],
    n_retrieved: int,
    stats: Counter[str] | None = None,
) -> dict[str, Any]:
    """Merge per-document maps using the released duplicate rules."""

    if n_retrieved < 0:
        raise ValueError("n_retrieved must be non-negative")
    stats = stats if stats is not None else Counter()
    docs = data_each.get("ctxs")
    raw_triplet_maps = data_each.get("triplets")
    if not isinstance(docs, list) or not isinstance(raw_triplet_maps, list):
        raise TypeError("Each record must contain list-valued 'ctxs' and 'triplets'")

    # [AUTHOR-CODE] ChainMap gives the first duplicate document ID precedence.
    merged_map = dict(ChainMap(*raw_triplet_maps)) if raw_triplet_maps else {}
    triplets_by_id = {str(doc_id): triples for doc_id, triples in merged_map.items()}

    new_docs: list[dict[str, Any]] = []
    global_triplets: list[tuple[str, str, str]] = []
    head_relations: set[tuple[str, str]] = set()
    global_ids: list[str] = []

    for doc in docs:
        if len(new_docs) >= n_retrieved:
            break
        if not isinstance(doc, dict):
            raise TypeError(f"Context is not an object: {doc!r}")
        if "id" not in doc:
            stats["contexts_without_id"] += 1
            continue

        doc_id = str(doc["id"])
        if doc_id not in triplets_by_id:
            raise KeyError(f"No extracted triples found for context id {doc_id!r}")
        new_docs.append(doc)
        global_ids.append(doc_id)
        stats["contexts_retained"] += 1

        for raw_triple in triplets_by_id[doc_id]:
            triple = _validate_triple(raw_triple, f"document {doc_id}")
            stats["document_triples"] += 1
            # [AUTHOR-CODE] Preserve the asymmetric substring validity check.
            if triple[2] in triple[0]:
                stats["dropped_tail_in_head"] += 1
                continue
            if triple in global_triplets:
                stats["dropped_global_duplicate"] += 1
                continue
            head_relation = (triple[0], triple[1])
            if head_relation in head_relations:
                # [AUTHOR-CODE] Only the first tail for a (head, relation) pair
                # is retained.  Report it rather than silently changing it.
                stats["dropped_head_relation_collision"] += 1
                continue
            global_triplets.append(triple)
            head_relations.add(head_relation)

    data_each["triplets"] = {"ids": global_ids, "trps": global_triplets}
    data_each["ctxs"] = new_docs
    stats["global_triples"] += len(global_triplets)
    return data_each


def is_qa_harm(
    check_list: Sequence[Sequence[str]],
    private_edges: Sequence[Sequence[str]],
    new_edge: Sequence[str],
) -> bool:
    graph = build_adjacency_list([*private_edges, new_edge])
    return any(are_nodes_connected(graph, check[0], check[1]) for check in check_list)


def _relation_tail_conflict(
    triple: tuple[str, str, str],
    relation_tail_heads: dict[tuple[str, str], set[str]],
) -> bool:
    heads = relation_tail_heads.get((triple[1], triple[2]), set())
    return any(_author_entity_match(triple[0], existing_head) for existing_head in heads)


def sample_privacy(
    global_triplets: Sequence[Sequence[str]],
    privacy_rate: float,
    query_entity: Sequence[Any],
    answers: Sequence[str] | str,
    flag: int,
    rng: random.Random | None = None,
    return_stats: bool = False,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]], int] | tuple[
    list[tuple[str, str, str]], list[tuple[str, str, str]], int, dict[str, int]
]:
    """Split a global graph while protecting QA and graph-separation constraints."""

    if not 0.0 <= privacy_rate <= 1.0:
        raise ValueError("privacy_rate must be in [0, 1]")
    rng = rng or random.Random(42)
    triples = [
        _validate_triple(triple, "global graph") for triple in global_triplets
    ]
    query_entities = [
        entity.text if hasattr(entity, "text") else str(entity)
        for entity in query_entity
    ]
    answer_values = [answers] if isinstance(answers, str) else list(answers)
    qa_check_list = [
        [entity, answer]
        for entity in query_entities
        for answer in answer_values
        if entity and isinstance(answer, str) and answer
    ]

    sample_size = int(len(triples) * privacy_rate)
    sampled_indices = rng.sample(range(len(triples)), sample_size)
    accepted_private_indices: set[int] = set()
    privacy: list[tuple[str, str, str]] = []
    private_edges: list[Edge] = []
    relation_tail_heads: dict[tuple[str, str], set[str]] = defaultdict(set)
    qa_rejected_indices: set[int] = set()

    for index in sampled_indices:
        triple = triples[index]
        edge = [triple[0], triple[2]]
        if is_qa_harm(qa_check_list, private_edges, edge):
            qa_rejected_indices.add(index)
            flag += 1
            continue
        accepted_private_indices.add(index)
        privacy.append(triple)
        private_edges.append(edge)
        relation_tail_heads[(triple[1], triple[2])].add(triple[0])

    # [RECONSTRUCTION] A sampled QA-harming triple must return to the public
    # candidate pool instead of disappearing from both public and private sets.
    remaining: list[tuple[int, tuple[str, str, str]]] = [
        (index, triple)
        for index, triple in enumerate(triples)
        if index not in accepted_private_indices
    ]
    promoted_indices: set[int] = set()

    # [RECONSTRUCTION] Rebuild the private graph after every promotion and
    # revisit earlier candidates until a fixed point.  The released code used a
    # stale graph after extending privacy, which could leave inferable public
    # paths behind.
    changed = True
    while changed:
        changed = False
        next_remaining: list[tuple[int, tuple[str, str, str]]] = []
        for index, triple in remaining:
            private_graph = build_adjacency_list(private_edges)
            should_promote = _relation_tail_conflict(triple, relation_tail_heads) or are_nodes_connected(
                private_graph, triple[0], triple[2]
            )
            if not should_promote:
                next_remaining.append((index, triple))
                continue

            edge = [triple[0], triple[2]]
            if is_qa_harm(qa_check_list, private_edges, edge):
                if index not in qa_rejected_indices:
                    flag += 1
                    qa_rejected_indices.add(index)
                next_remaining.append((index, triple))
                continue

            privacy.append(triple)
            private_edges.append(edge)
            relation_tail_heads[(triple[1], triple[2])].add(triple[0])
            promoted_indices.add(index)
            changed = True
        remaining = next_remaining

    public = [triple for _, triple in remaining]
    public_edges = [[triple[0], triple[2]] for triple in public]
    public_graph = build_adjacency_list(public_edges)
    filtered_privacy = [
        triple
        for triple in privacy
        if not are_nodes_connected(public_graph, triple[0], triple[2])
    ]

    stats = {
        "sampled_requested": sample_size,
        "private_initial": len(accepted_private_indices),
        "qa_rejected": len(qa_rejected_indices),
        "private_promoted": len(promoted_indices),
        "private_filtered": len(privacy) - len(filtered_privacy),
        "private_final": len(filtered_privacy),
        "public_final": len(public),
    }
    if return_stats:
        return public, filtered_privacy, flag, stats
    return public, filtered_privacy, flag


def _resolve_output_path(args: argparse.Namespace) -> str:
    if args.output_path:
        return args.output_path
    percentage = int(args.privacy_rate * 100)
    filename = f"{args.dataset_name}_{args.top_k}_{percentage}_new.jsonl"
    return os.path.join(args.output_dir, filename)


def main(args: argparse.Namespace) -> dict[str, float | int]:
    output_path = _resolve_output_path(args)
    if os.path.abspath(args.input_path) == os.path.abspath(output_path):
        raise ValueError("Input and output paths must be different")

    logger.info("Loading spaCy model %s", args.spacy_model)
    nlp = spacy.load(args.spacy_model)
    rng = random.Random(args.seed)
    totals: Counter[str] = Counter()
    flag = 0
    _safe_parent(output_path)

    with open(output_path, "w", encoding="utf-8") as fout:
        for record_number, data_each in enumerate(
            iter_data(args.input_path, max_samples=args.max_samples), start=1
        ):
            data_each = merge_triplets(data_each, args.top_k, stats=totals)
            global_triplets = data_each["triplets"]["trps"]
            question = data_each.get("question")
            answers = data_each.get("answers")
            if not isinstance(question, str):
                raise TypeError(f"Record {record_number} has non-string question")
            if not isinstance(answers, (list, str)):
                raise TypeError(f"Record {record_number} has invalid answers")

            query_entities = nlp(question).ents
            public, privacy, flag, split_stats = sample_privacy(
                global_triplets,
                args.privacy_rate,
                query_entities,
                answers,
                flag,
                rng=rng,
                return_stats=True,
            )
            data_each["public"] = public
            data_each["privacy"] = privacy
            totals.update(split_stats)
            totals["records"] += 1
            json.dump(data_each, fout, ensure_ascii=False)
            fout.write("\n")

    global_count = totals["global_triples"]
    summary: dict[str, float | int] = dict(sorted(totals.items()))
    summary["actual_private_ratio"] = (
        totals["private_final"] / global_count if global_count else 0.0
    )
    summary["qa_safety_events"] = flag
    logger.info("Private/public split statistics: %s", summary)
    logger.info("Saved results to %s", output_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "--data_path",
        dest="input_path",
        default="./dataset/to_triplets/hotpotqa_with_triplets.jsonl",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=None,
        help="Explicit output JSONL path",
    )
    parser.add_argument("--output_dir", default="./dataset/sample_privacy/")
    parser.add_argument("--dataset_name", default="hotpotqa")
    parser.add_argument("--top_k", "--n_retrieved", dest="top_k", type=int, default=10)
    parser.add_argument("--privacy_rate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
