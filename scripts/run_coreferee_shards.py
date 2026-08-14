"""Run deterministic Coreferee JSONL shards and merge them in source order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Shard:
    index: int
    start: int
    count: int


def build_shards(total: int, records_per_shard: int) -> list[Shard]:
    if total <= 0:
        raise ValueError("input must contain at least one JSONL record")
    if records_per_shard <= 0:
        raise ValueError("records_per_shard must be positive")
    return [
        Shard(index=index, start=start, count=min(records_per_shard, total - start))
        for index, start in enumerate(range(0, total, records_per_shard))
    ]


def count_jsonl_records(path: Path) -> int:
    with path.open("r", encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_jsonl_shards(shard_paths: list[Path], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    records = 0
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for shard_path in shard_paths:
                with shard_path.open("r", encoding="utf-8") as source:
                    for line in source:
                        if line.strip():
                            output.write(line if line.endswith("\n") else line + "\n")
                            records += 1
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return records


def run_shard(
    *,
    task: Shard,
    input_path: Path,
    coref_script: Path,
    output_path: Path,
    metadata_path: Path,
    log_path: Path,
    spacy_model: str,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(coref_script),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--spacy_model",
        spacy_model,
        "--start_samples",
        str(task.start),
        "--max_samples",
        str(task.count),
        "--metadata_output",
        str(metadata_path),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("records") != task.count:
        raise RuntimeError(
            f"shard {task.index} processed {metadata.get('records')}; expected {task.count}"
        )
    return {
        "index": task.index,
        "start": task.start,
        "records": task.count,
        "output": str(output_path.resolve()),
        "metadata": str(metadata_path.resolve()),
        "log": str(log_path.resolve()),
    }


def process_input(
    *,
    input_path: Path,
    output_dir: Path,
    shard_dir: Path,
    log_dir: Path,
    coref_script: Path,
    spacy_model: str,
    workers: int,
    records_per_shard: int,
    max_records: int | None,
) -> dict[str, object]:
    available_records = count_jsonl_records(input_path)
    total = min(available_records, max_records) if max_records is not None else available_records
    shards = build_shards(total, records_per_shard)
    prefix = input_path.stem
    current_shard_dir = shard_dir / prefix
    tasks = []
    for shard in shards:
        shard_stem = f"{shard.index:04d}"
        tasks.append(
            (
                shard,
                current_shard_dir / f"{shard_stem}.jsonl",
                current_shard_dir / f"{shard_stem}.meta.json",
                log_dir / prefix / f"{shard_stem}.log",
            )
        )

    completed: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                run_shard,
                task=shard,
                input_path=input_path,
                coref_script=coref_script,
                output_path=output_path,
                metadata_path=metadata_path,
                log_path=log_path,
                spacy_model=spacy_model,
            )
            for shard, output_path, metadata_path, log_path in tasks
        ]
        for future in as_completed(futures):
            completed.append(future.result())

    ordered = sorted(completed, key=lambda item: int(item["index"]))
    output_path = output_dir / f"{prefix}_clean.jsonl"
    merged_records = merge_jsonl_shards(
        [Path(str(item["output"])) for item in ordered], output_path
    )
    if merged_records != total:
        raise RuntimeError(f"merged {merged_records} records; expected {total}")
    return {
        "input": str(input_path.resolve()),
        "input_sha256": sha256(input_path),
        "records": total,
        "shards": ordered,
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "merged_records": merged_records,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument(
        "--coref-script",
        type=Path,
        default=Path("dataset/retrieved_data/clean_retrieved.py"),
    )
    parser.add_argument("--spacy-model", default="en_core_web_lg")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--records-per-shard", type=int, default=250)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("max_records must be positive when provided")
    if not args.coref_script.is_file():
        raise FileNotFoundError(f"Coreferee script not found: {args.coref_script}")
    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"input JSONL not found: {missing}")

    reports = [
        process_input(
            input_path=path,
            output_dir=args.output_dir,
            shard_dir=args.shard_dir,
            log_dir=args.log_dir,
            coref_script=args.coref_script,
            spacy_model=args.spacy_model,
            workers=args.workers,
            records_per_shard=args.records_per_shard,
            max_records=args.max_records,
        )
        for path in args.inputs
    ]
    result = {
        "schema": "eraser4rag-coreferee-shards-v1",
        "spacy_model": args.spacy_model,
        "workers_per_input": args.workers,
        "records_per_shard": args.records_per_shard,
        "max_records_per_input": args.max_records,
        "inputs": reports,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
