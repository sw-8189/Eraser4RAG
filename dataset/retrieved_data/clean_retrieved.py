"""Resolve coreferences in retrieved documents.

This script intentionally belongs to the lightweight ``eraser-coref``
environment.  In particular, it must not import Torch/ReLiK/TRL.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import platform
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import coreferee  # noqa: F401 - importing registers the spaCy component
import spacy


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CorefereeContextTimeout(TimeoutError):
    """Raised when a single Coreferee context exceeds its configured budget."""


@contextmanager
def _context_timeout(seconds: float | None) -> Iterator[None]:
    """Bound one Coreferee call on Unix without changing the default behavior."""

    if seconds is None:
        yield
        return
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        raise RuntimeError("Per-context timeouts require Unix signal.setitimer support")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Per-context timeouts must run on the main thread")

    expired = False

    def _raise_timeout(signum: int, frame: Any) -> None:
        nonlocal expired
        del signum, frame
        expired = True
        raise CorefereeContextTimeout(
            f"Coreferee exceeded the {seconds:g}-second per-context limit"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    # Coreferee catches pipeline exceptions internally. Re-raise after it
    # returns so the caller still preserves the original text and provenance.
    if expired:
        raise CorefereeContextTimeout(
            f"Coreferee exceeded the {seconds:g}-second per-context limit"
        )


def _safe_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def iter_data(
    data_path: str,
    max_samples: int | None = None,
    start_samples: int = 0,
) -> Iterator[dict[str, Any]]:
    """Yield JSON/JSONL records without loading a JSONL file into memory."""

    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative or None")
    if start_samples < 0:
        raise ValueError("start_samples must be non-negative")

    suffix = Path(data_path).suffix.lower()
    if suffix == ".json":
        with open(data_path, "r", encoding="utf-8") as fin:
            data = json.load(fin)
        if not isinstance(data, list):
            raise TypeError(f"Expected a JSON array in {data_path}")
        for index, record in enumerate(data):
            if index < start_samples:
                continue
            if max_samples is not None and index >= start_samples + max_samples:
                break
            if not isinstance(record, dict):
                raise TypeError(f"Record {index} in {data_path} is not an object")
            yield record
        return

    if suffix == ".jsonl":
        with open(data_path, "r", encoding="utf-8") as fin:
            emitted = 0
            skipped = 0
            for line_number, line in enumerate(fin, start=1):
                if not line.strip():
                    continue
                if skipped < start_samples:
                    skipped += 1
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


def load_data(
    data_path: str, max_samples: int | None = None, start_samples: int = 0
) -> list[dict[str, Any]]:
    """Backward-compatible materializing wrapper used by older callers/tests."""

    return list(
        iter_data(
            data_path,
            max_samples=max_samples,
            start_samples=start_samples,
        )
    )


def _entity_text(coref_doc: Any, token: Any) -> str:
    """Return an entity span for a representative token when one is available."""

    if token.ent_type_ == "":
        return token.text
    for entity in coref_doc.ents:
        if token.i >= entity.start and token.i < entity.end:
            return entity.text
    # [ENGINEERING] The author code indexed the first match and could crash.
    return token.text


def coref_text(coref_nlp: Any, text: str) -> str:
    """Apply the author's token-level Coreferee resolution strategy."""

    coref_doc = coref_nlp(text)
    resolved_text = ""

    # [AUTHOR-CODE] Preserve token-by-token replacement and spacing semantics so
    # rebuilt triples remain comparable with the released author data.
    for token in coref_doc:
        representatives = coref_doc._.coref_chains.resolve(token)
        if representatives:
            resolved_text += " " + " and ".join(
                _entity_text(coref_doc, representative)
                for representative in representatives
            )
        else:
            resolved_text += " " + token.text

    return resolved_text


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input_path
    output_path = args.output_path
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("Input and output paths must be different")

    logger.info("Loading spaCy model %s with Coreferee", args.spacy_model)
    coref_nlp = spacy.load(args.spacy_model)
    coref_nlp.add_pipe("coreferee")

    _safe_parent(output_path)
    processed_records = 0
    processed_contexts = 0
    timed_out_contexts: list[dict[str, Any]] = []
    with open(output_path, "w", encoding="utf-8") as fout:
        for record_index, data_each in enumerate(
            iter_data(
                input_path,
                max_samples=args.max_samples,
                start_samples=args.start_samples,
            ),
            start=1,
        ):
            contexts = data_each.get("ctxs")
            if not isinstance(contexts, list):
                raise TypeError(f"Record {record_index} has no list-valued 'ctxs'")

            for context_index, example in enumerate(contexts):
                if not isinstance(example, dict):
                    raise TypeError(
                        f"Record {record_index} ctx {context_index} is not an object"
                    )
                text = example.get("text")
                if not isinstance(text, str):
                    raise TypeError(
                        f"Record {record_index} ctx {context_index} has non-string text"
                    )
                try:
                    with _context_timeout(args.per_context_timeout_seconds):
                        example["text"] = coref_text(coref_nlp, text)
                except CorefereeContextTimeout:
                    example["text"] = text
                    source_index = args.start_samples + record_index - 1
                    record_id = data_each.get("id")
                    if record_id is None:
                        record_id = data_each.get("_id", f"record-{source_index}")
                    timeout_record = {
                        "id": str(record_id),
                        "source_index": source_index,
                        "context_index": context_index,
                    }
                    timed_out_contexts.append(timeout_record)
                    logger.warning(
                        "Timed out resolving record %s, context %d; preserving original text",
                        timeout_record["id"],
                        context_index,
                    )
                processed_contexts += 1

            json.dump(data_each, fout, ensure_ascii=False)
            fout.write("\n")
            processed_records += 1
            if processed_records % 100 == 0:
                logger.info("Processed %d records", processed_records)

    metadata = {
        "coreference_applied": True,
        "coreference_backend": "coreferee",
        "python_version": platform.python_version(),
        "spacy_version": spacy.__version__,
        "coreferee_version": _package_version("coreferee"),
        "spacy_model": args.spacy_model,
        "input": os.path.abspath(input_path),
        "output": os.path.abspath(output_path),
        "records": processed_records,
        "contexts": processed_contexts,
        "max_samples": args.max_samples,
        "start_samples": args.start_samples,
        "per_context_timeout_seconds": args.per_context_timeout_seconds,
        "timed_out_contexts": len(timed_out_contexts),
        "timeout_records": timed_out_contexts,
    }
    metadata_path = args.metadata_output or f"{output_path}.meta.json"
    _safe_parent(metadata_path)
    with open(metadata_path, "w", encoding="utf-8") as fout:
        json.dump(metadata, fout, ensure_ascii=False, indent=2)
        fout.write("\n")

    logger.info("Saved %d records to %s", processed_records, output_path)
    logger.info("Saved provenance metadata to %s", metadata_path)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "--data",
        dest="input_path",
        default="./dataset/retrieved_data/triviaqa_retrieved.jsonl",
        help="Input .json or .jsonl retrieval records",
    )
    parser.add_argument(
        "--output",
        "--output_dir",
        dest="output_path",
        default="./dataset/retrieved_data/triviaqa_retrieved_clean.jsonl",
        help="Output .jsonl file",
    )
    parser.add_argument("--spacy_model", default="en_core_web_lg")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_samples", type=int, default=0)
    parser.add_argument(
        "--per-context-timeout-seconds",
        type=float,
        default=None,
        help="Optional Unix timeout for each context; preserve original text on timeout",
    )
    parser.add_argument(
        "--metadata_output",
        default=None,
        help="Optional metadata JSON path; defaults to <output>.meta.json",
    )
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    if (
        parsed_args.per_context_timeout_seconds is not None
        and parsed_args.per_context_timeout_seconds <= 0
    ):
        raise ValueError("per-context-timeout-seconds must be positive when provided")
    main(parsed_args)
