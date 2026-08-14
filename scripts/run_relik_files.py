"""Run the released ReLiK extractor over multiple cleaned JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def count_jsonl_records(path: Path) -> int:
    with path.open("r", encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    stem = input_path.stem
    suffix = "_retrieved_clean"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return output_dir / f"{stem}_with_triplets.jsonl"


def process_file(
    *,
    input_path: Path,
    output_path: Path,
    log_path: Path,
    relik_script: Path,
    model: str,
    device: str,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    input_records = count_jsonl_records(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(relik_script),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--relik_model",
        model,
        "--device",
        device,
        "--use_nme",
        "--batch_size",
        str(batch_size),
        "--seed",
        str(seed),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    output_records = count_jsonl_records(output_path)
    if output_records != input_records:
        raise RuntimeError(
            f"{input_path.name}: produced {output_records} records, expected {input_records}"
        )
    return {
        "input": str(input_path.resolve()),
        "input_records": input_records,
        "input_sha256": sha256(input_path),
        "output": str(output_path.resolve()),
        "output_records": output_records,
        "output_sha256": sha256(output_path),
        "log": str(log_path.resolve()),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--relik-script", type=Path, default=Path("dataset/to_triplets/RELIK_re.py")
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not args.relik_script.is_file():
        raise FileNotFoundError(f"ReLiK script not found: {args.relik_script}")
    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"input JSONL not found: {missing}")

    reports = [
        process_file(
            input_path=input_path,
            output_path=output_path_for(input_path, args.output_dir),
            log_path=args.log_dir / f"{input_path.stem}.log",
            relik_script=args.relik_script,
            model=args.model,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        for input_path in args.inputs
    ]
    result = {
        "schema": "eraser4rag-relik-files-v1",
        "model": args.model,
        "device": args.device,
        "use_nme": True,
        "batch_size": args.batch_size,
        "seed": args.seed,
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
