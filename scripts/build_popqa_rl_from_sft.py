"""Regroup the author SFT JSONL into a PPO engineering smoke dataset.

The source is flattened per document and has no query IDs.  Consecutive rows
with identical, order-sensitive global public/private graphs are treated as one
group.  The output is explicitly marked as reconstructed smoke data and must
not be presented as the author's formal four-dataset PPO corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def graph_fingerprint(public: Any, private: Any) -> str:
    payload = json.dumps(
        [public, private], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def iter_sft_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"record at {path}:{line_number} is not an object")
            yield line_number, record


def _ctx(record: dict[str, Any], group_index: int, document_index: int) -> dict[str, Any]:
    for field in ("text", "o_public", "o_private"):
        if field not in record:
            raise ValueError(f"SFT record is missing required field {field!r}")
    return {
        "id": f"regrouped-{group_index:06d}-doc-{document_index:02d}",
        "text": record["text"],
        "public": record["o_public"],
        "private": record["o_private"],
    }


def build_smoke_dataset(
    data_path: str | Path,
    output_path: str | Path,
    *,
    max_groups: int | None = None,
    max_docs_per_group: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    source_path = Path(data_path)
    destination = Path(output_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"SFT dataset not found: {source_path}")
    if destination.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    if max_groups is not None and max_groups <= 0:
        raise ValueError("max_groups must be positive when provided")
    if max_docs_per_group is not None and max_docs_per_group <= 0:
        raise ValueError("max_docs_per_group must be positive when provided")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    stats = {
        "source": "regrouped_author_sft",
        "source_records": 0,
        "groups_written": 0,
        "documents_written": 0,
        "documents_skipped_by_group_cap": 0,
        "non_contiguous_repeated_groups": 0,
    }
    seen_fingerprints: set[str] = set()

    current_fingerprint: str | None = None
    current_record: dict[str, Any] | None = None
    current_contexts: list[dict[str, Any]] = []

    def emit(output) -> bool:
        nonlocal current_record, current_contexts, current_fingerprint
        if current_record is None or current_fingerprint is None:
            return True
        if max_groups is not None and stats["groups_written"] >= max_groups:
            return False
        group_index = stats["groups_written"]
        result = {
            "question": "",
            "answers": [],
            "public": current_record["public"],
            "privacy": current_record["private"],
            "ctxs": current_contexts,
            "metadata": {
                "source": "regrouped_author_sft",
                "purpose": "ppo_engineering_smoke_only",
                "group_fingerprint": current_fingerprint,
                "group_index": group_index,
            },
        }
        output.write(json.dumps(result, ensure_ascii=False) + "\n")
        stats["groups_written"] += 1
        stats["documents_written"] += len(current_contexts)
        return True

    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for _, record in iter_sft_records(source_path):
                stats["source_records"] += 1
                for field in ("public", "private"):
                    if field not in record or not isinstance(record[field], list):
                        raise ValueError(f"SFT record has invalid global {field!r}")
                fingerprint = graph_fingerprint(record["public"], record["private"])
                if current_fingerprint is None:
                    current_fingerprint = fingerprint
                    current_record = record
                elif fingerprint != current_fingerprint:
                    if not emit(output):
                        break
                    seen_fingerprints.add(current_fingerprint)
                    if fingerprint in seen_fingerprints:
                        stats["non_contiguous_repeated_groups"] += 1
                    current_fingerprint = fingerprint
                    current_record = record
                    current_contexts = []

                if max_docs_per_group is None or len(current_contexts) < max_docs_per_group:
                    current_contexts.append(
                        _ctx(record, stats["groups_written"], len(current_contexts))
                    )
                else:
                    stats["documents_skipped_by_group_cap"] += 1
            else:
                emit(output)
        if stats["groups_written"] == 0:
            raise ValueError("SFT dataset produced no PPO smoke groups")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    stats["data_path"] = str(source_path.resolve())
    stats["output_path"] = str(destination.resolve())
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="dataset/constructed_dataset/popqa_10_25_filtered_new.jsonl",
    )
    parser.add_argument(
        "--output",
        default="outputs/smoke/popqa_regrouped_author_sft_rl.jsonl",
    )
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--max-docs-per-group", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats = build_smoke_dataset(
        args.data,
        args.output,
        max_groups=args.max_groups,
        max_docs_per_group=args.max_docs_per_group,
        force=args.force,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
