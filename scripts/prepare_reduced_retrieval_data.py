"""Prepare deterministic PopQA/HotpotQA retrieval subsets for reduced PPO.

This is an explicitly documented reconstruction path.  It converts pinned,
pre-retrieved sources into the JSONL schema consumed by the released
Coreferee/ReLiK pipeline.  It does not claim to recreate the paper's missing
Wikipedia/Contriever index.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Callable


POPQA_REVISION = "dcc3f4f72fab2f7bca386c51e5cf329109727919"
HOTPOTQA_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
SCHEMA = "eraser4rag-reduced-retrieval-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _question_key(question: str) -> str:
    return " ".join(question.casefold().split())


def _priority(seed: int, question_key: str, source_index: int) -> str:
    payload = f"{seed}\0{question_key}\0{source_index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def normalize_answers(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        parsed: Any = None
        if stripped.startswith(("[", "(")):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(stripped)
                    break
                except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                    continue
        value = parsed if parsed is not None else [stripped]
    if not isinstance(value, (list, tuple)):
        raise ValueError("answers must be a string or sequence of strings")
    answers: list[str] = []
    seen: set[str] = set()
    for item in value:
        answer = _require_text(item, "answer")
        key = answer.casefold()
        if key not in seen:
            answers.append(answer)
            seen.add(key)
    if not answers:
        raise ValueError("answers must not be empty")
    return answers


def _document(title: Any, text: Any, *, doc_id: str) -> dict[str, str]:
    normalized_title = _require_text(title, "document title")
    normalized_text = _require_text(text, "document text")
    return {"id": doc_id, "title": normalized_title, "text": normalized_text}


def parse_popqa_document(value: Any, *, doc_id: str) -> dict[str, str]:
    if isinstance(value, dict):
        title = value.get("title") or value.get("page_title")
        text = value.get("text") or value.get("content") or value.get("contents")
        return _document(title, text, doc_id=doc_id)
    if not isinstance(value, str):
        raise ValueError(f"unsupported PopQA document type: {type(value).__name__}")

    stripped = value.strip()
    if stripped.startswith("{"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict):
                return parse_popqa_document(parsed, doc_id=doc_id)

    match = re.fullmatch(
        r"\s*Title:\s*(?P<title>.+?)\r?\nContent:\s*(?P<text>.+)\s*",
        value,
        flags=re.DOTALL,
    )
    if match:
        return _document(match.group("title"), match.group("text"), doc_id=doc_id)

    lines = stripped.splitlines()
    if len(lines) >= 2:
        return _document(lines[0], "\n".join(lines[1:]), doc_id=doc_id)
    if stripped:
        # A small subset of the released pre-retrieved source contains body-only
        # passages. Preserve the passage instead of dropping a retrieved rank;
        # downstream extraction uses the text and does not require a real title.
        return _document("[untitled retrieved passage]", stripped, doc_id=doc_id)
    raise ValueError("PopQA document has no recognized title/text boundary")


def parse_hotpot_context(value: Any, *, record_id: str) -> list[dict[str, str]]:
    pairs: list[tuple[Any, Any]] = []
    if isinstance(value, dict):
        titles = value.get("title")
        sentences = value.get("sentences")
        if not isinstance(titles, list) or not isinstance(sentences, list):
            raise ValueError("HotpotQA context dict requires title/sentences lists")
        if len(titles) != len(sentences):
            raise ValueError("HotpotQA context title/sentences lengths differ")
        pairs = list(zip(titles, sentences))
    elif isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("HotpotQA context entries must be objects")
            pairs.append((item.get("title"), item.get("sentences")))
    else:
        raise ValueError("HotpotQA context must be a dict or list")

    documents: list[dict[str, str]] = []
    for rank, (title, sentences) in enumerate(pairs):
        if isinstance(sentences, str):
            text = sentences
        elif isinstance(sentences, list) and all(isinstance(item, str) for item in sentences):
            text = " ".join(item.strip() for item in sentences if item.strip())
        else:
            raise ValueError("HotpotQA sentences must be a string or list of strings")
        documents.append(_document(title, text, doc_id=f"hotpotqa:{record_id}:{rank}"))
    return documents


def build_popqa_record(
    row: dict[str, Any], *, source_index: int, top_k: int
) -> dict[str, Any]:
    question = _require_text(row.get("question"), "question")
    answers = normalize_answers(row.get("possible_answers", row.get("answers")))
    source_id = str(row.get("id", source_index))
    raw_documents = row.get("retrieved_docs")
    if not isinstance(raw_documents, (list, tuple)) or len(raw_documents) < top_k:
        raise ValueError(f"PopQA record {source_id} has fewer than {top_k} documents")
    contexts = [
        parse_popqa_document(value, doc_id=f"popqa:{source_id}:{rank}")
        for rank, value in enumerate(raw_documents[:top_k])
    ]
    return {
        "question": question,
        "answers": answers,
        "ctxs": contexts,
        "metadata": {
            "schema": SCHEMA,
            "dataset": "popqa",
            "source_index": source_index,
            "source_id": source_id,
            "source_revision": POPQA_REVISION,
            "retrieval_top_k": top_k,
            "retrieval_deviation": "third_party_pre_retrieved_source; retriever/index revision undisclosed",
        },
    }


def build_hotpot_record(
    row: dict[str, Any], *, source_index: int, top_k: int, source_split: str
) -> dict[str, Any]:
    question = _require_text(row.get("question"), "question")
    answers = normalize_answers(row.get("answer", row.get("answers")))
    source_id = str(row.get("id", source_index))
    contexts = parse_hotpot_context(row.get("context"), record_id=source_id)
    if len(contexts) < top_k:
        raise ValueError(f"HotpotQA record {source_id} has fewer than {top_k} documents")
    return {
        "question": question,
        "answers": answers,
        "ctxs": contexts[:top_k],
        "metadata": {
            "schema": SCHEMA,
            "dataset": "hotpotqa",
            "source_index": source_index,
            "source_id": source_id,
            "source_split": source_split,
            "source_revision": HOTPOTQA_REVISION,
            "retrieval_top_k": top_k,
            "retrieval_deviation": "official distractor contexts used instead of missing pinned Contriever index",
        },
    }


def _iter_rows(paths: Sequence[Path], columns: Sequence[str] | None = None) -> Iterator[tuple[int, dict[str, Any]]]:
    source_index = 0
    for path in sorted(paths):
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        row = json.loads(line)
                        if columns is not None:
                            row = {key: row.get(key) for key in columns}
                        yield source_index, row
                        source_index += 1
            continue
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"unsupported source file: {path}")
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet input requires pyarrow in eraser-main") from exc
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024, columns=columns):
            for row in batch.to_pylist():
                yield source_index, row
                source_index += 1


def select_unique_indices(
    rows: Iterable[tuple[int, dict[str, Any]]],
    *,
    size: int,
    seed: int,
    excluded_questions: set[str] | None = None,
    is_eligible: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[list[int], list[str]]:
    if size <= 0:
        raise ValueError("selection size must be positive")
    excluded = excluded_questions or set()
    candidates: dict[str, tuple[str, int]] = {}
    for source_index, row in rows:
        if is_eligible is not None and not is_eligible(row):
            continue
        question = _require_text(row.get("question"), "question")
        key = _question_key(question)
        if key in excluded:
            continue
        candidate = (_priority(seed, key, source_index), source_index)
        if key not in candidates or candidate < candidates[key]:
            candidates[key] = candidate
    if len(candidates) < size:
        raise ValueError(f"only {len(candidates)} unique questions available for {size} requested")
    ranked = sorted((priority, index, key) for key, (priority, index) in candidates.items())[:size]
    return [index for _, index, _ in ranked], [key for _, _, key in ranked]


def hotpot_has_top_k_contexts(row: dict[str, Any], *, top_k: int) -> bool:
    """Return whether an official HotpotQA record can supply the requested ranks."""
    context = row.get("context")
    if isinstance(context, dict):
        titles = context.get("title")
        sentences = context.get("sentences")
        return (
            isinstance(titles, list)
            and isinstance(sentences, list)
            and len(titles) == len(sentences)
            and len(titles) >= top_k
        )
    if isinstance(context, list):
        return len(context) >= top_k and all(isinstance(item, dict) for item in context)
    return False


def _write_selected(
    *,
    paths: Sequence[Path],
    selected_indices: Sequence[int],
    output_path: Path,
    builder,
) -> dict[str, Any]:
    selected = set(selected_indices)
    order = {source_index: rank for rank, source_index in enumerate(selected_indices)}
    records: dict[int, dict[str, Any]] = {}
    document_count = 0
    for source_index, row in _iter_rows(paths):
        if source_index not in selected:
            continue
        record = builder(row, source_index)
        records[source_index] = record
        document_count += len(record["ctxs"])
    missing = selected.difference(records)
    if missing:
        raise RuntimeError(f"selected source indices missing during second pass: {sorted(missing)[:10]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for source_index in sorted(records, key=order.__getitem__):
                json.dump(records[source_index], output, ensure_ascii=False, separators=(",", ":"))
                output.write("\n")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(output_path.resolve()),
        "sha256": _sha256(output_path),
        "records": len(records),
        "documents": document_count,
        "source_indices": list(selected_indices),
    }


def _paths(values: Sequence[str], label: str) -> list[Path]:
    paths = [Path(value) for value in values]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {label} files: {missing}")
    return paths


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.top_k <= 0:
        raise ValueError("top_k must be positive")
    popqa_paths = _paths(args.popqa_input, "PopQA")
    hotpot_train_paths = _paths(args.hotpot_train_input, "HotpotQA train")
    hotpot_eval_paths = _paths(args.hotpot_eval_input, "HotpotQA eval")

    pop_indices, pop_questions = select_unique_indices(
        _iter_rows(popqa_paths, columns=["question"]),
        size=args.train_size + args.eval_size,
        seed=args.seed,
    )
    pop_train_indices = pop_indices[: args.train_size]
    pop_eval_indices = pop_indices[args.train_size :]
    if set(pop_train_indices) & set(pop_eval_indices):
        raise AssertionError("PopQA train/eval source indices overlap")

    hot_train_indices, hot_train_questions = select_unique_indices(
        _iter_rows(hotpot_train_paths, columns=["question", "context"]),
        size=args.train_size,
        seed=args.seed,
        is_eligible=lambda row: hotpot_has_top_k_contexts(row, top_k=args.top_k),
    )
    hot_eval_indices, hot_eval_questions = select_unique_indices(
        _iter_rows(hotpot_eval_paths, columns=["question", "context"]),
        size=args.eval_size,
        seed=args.seed + 1,
        excluded_questions=set(hot_train_questions),
        is_eligible=lambda row: hotpot_has_top_k_contexts(row, top_k=args.top_k),
    )

    output_root = Path(args.output_root)
    outputs = {
        "popqa_train": _write_selected(
            paths=popqa_paths,
            selected_indices=pop_train_indices,
            output_path=output_root / "popqa_train_retrieved.jsonl",
            builder=lambda row, index: build_popqa_record(row, source_index=index, top_k=args.top_k),
        ),
        "popqa_eval": _write_selected(
            paths=popqa_paths,
            selected_indices=pop_eval_indices,
            output_path=output_root / "popqa_eval_retrieved.jsonl",
            builder=lambda row, index: build_popqa_record(row, source_index=index, top_k=args.top_k),
        ),
        "hotpotqa_train": _write_selected(
            paths=hotpot_train_paths,
            selected_indices=hot_train_indices,
            output_path=output_root / "hotpotqa_train_retrieved.jsonl",
            builder=lambda row, index: build_hotpot_record(
                row, source_index=index, top_k=args.top_k, source_split="train"
            ),
        ),
        "hotpotqa_eval": _write_selected(
            paths=hotpot_eval_paths,
            selected_indices=hot_eval_indices,
            output_path=output_root / "hotpotqa_eval_retrieved.jsonl",
            builder=lambda row, index: build_hotpot_record(
                row, source_index=index, top_k=args.top_k, source_split="validation"
            ),
        ),
    }
    input_files = [*popqa_paths, *hotpot_train_paths, *hotpot_eval_paths]
    manifest = {
        "schema": SCHEMA,
        "seed": args.seed,
        "train_size_per_dataset": args.train_size,
        "eval_size_per_dataset": args.eval_size,
        "top_k": args.top_k,
        "source_revisions": {
            "popqa": POPQA_REVISION,
            "hotpotqa": HOTPOTQA_REVISION,
        },
        "input_files": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in input_files
        ],
        "outputs": outputs,
        "question_overlap": {
            "popqa_train_eval": 0,
            "hotpotqa_train_eval": len(set(hot_train_questions) & set(hot_eval_questions)),
        },
        "deviations": [
            "The released repository lacks the pinned Wikipedia/Contriever index.",
            "PopQA uses a pinned third-party pre-retrieved top-20 source truncated to top-k.",
            "HotpotQA uses official distractor contexts instead of reconstructed Contriever retrieval.",
            "HotpotQA rows with fewer than top-k official contexts are excluded before deterministic selection.",
            "The reduced run uses deterministic 5k train / 1k eval subsets per dataset by default.",
        ],
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--popqa-input", nargs="+", required=True)
    parser.add_argument("--hotpot-train-input", nargs="+", required=True)
    parser.add_argument("--hotpot-eval-input", nargs="+", required=True)
    parser.add_argument(
        "--output-root", default="dataset/retrieved_data/two_dataset_v1"
    )
    parser.add_argument(
        "--manifest", default="outputs/manifests/two_dataset_retrieval_manifest.json"
    )
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    result = prepare(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
