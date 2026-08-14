"""Build validated PPO JSONL files from ReLiK extraction outputs."""

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


def dataset_name_for(triples_path: Path) -> str:
    suffix = "_with_triplets"
    stem = triples_path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def process_file(
    *,
    triples_path: Path,
    sampled_dir: Path,
    final_dir: Path,
    validation_dir: Path,
    log_dir: Path,
    process_script: Path,
    map_script: Path,
    validator_script: Path,
    top_k: int,
    privacy_rate: float,
    seed: int,
    spacy_model: str,
) -> dict[str, object]:
    name = dataset_name_for(triples_path)
    input_records = count_jsonl_records(triples_path)
    sampled_path = sampled_dir / f"{name}_sampled.jsonl"
    final_path = final_dir / f"{name}_rl.jsonl"
    process_command = [
        sys.executable,
        str(process_script),
        "--input",
        str(triples_path),
        "--output",
        str(sampled_path),
        "--dataset_name",
        name,
        "--top_k",
        str(top_k),
        "--privacy_rate",
        str(privacy_rate),
        "--seed",
        str(seed),
        "--spacy_model",
        spacy_model,
    ]
    _run(process_command, log_dir / f"{name}_sample.log")
    map_command = [
        sys.executable,
        str(map_script),
        "--triples_input",
        str(triples_path),
        "--sampled_input",
        str(sampled_path),
        "--output",
        str(final_path),
    ]
    _run(map_command, log_dir / f"{name}_map.log")
    validation_command = [
        sys.executable,
        str(validator_script),
        "--data",
        str(final_path),
        "--output-dir",
        str(validation_dir),
        "--require-qa",
    ]
    _run(validation_command, log_dir / f"{name}_validate.log")
    sampled_records = count_jsonl_records(sampled_path)
    final_records = count_jsonl_records(final_path)
    if input_records != sampled_records or input_records != final_records:
        raise RuntimeError(
            f"{name}: record counts input/sample/final are "
            f"{input_records}/{sampled_records}/{final_records}"
        )
    validation_path = validation_dir / f"{final_path.stem}_rl_dataset_report.json"
    report = json.loads(validation_path.read_text(encoding="utf-8"))
    if not report.get("valid"):
        raise RuntimeError(f"{name}: final RL validation failed")
    return {
        "dataset": name,
        "input": str(triples_path.resolve()),
        "input_records": input_records,
        "input_sha256": sha256(triples_path),
        "sampled": str(sampled_path.resolve()),
        "sampled_records": sampled_records,
        "sampled_sha256": sha256(sampled_path),
        "final": str(final_path.resolve()),
        "final_records": final_records,
        "final_sha256": sha256(final_path),
        "validation_report": str(validation_path.resolve()),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triples-inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--sampled-dir", required=True, type=Path)
    parser.add_argument("--final-dir", required=True, type=Path)
    parser.add_argument("--validation-dir", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--process-script", type=Path, default=Path("utils/process_triplets.py"))
    parser.add_argument("--map-script", type=Path, default=Path("utils/add_sampled_data.py"))
    parser.add_argument("--validator-script", type=Path, default=Path("scripts/validate_rl_dataset.py"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--privacy-rate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0.0 <= args.privacy_rate <= 1.0:
        raise ValueError("privacy_rate must be in [0, 1]")
    for label, path in {
        "process": args.process_script,
        "map": args.map_script,
        "validator": args.validator_script,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} script not found: {path}")
    missing = [str(path) for path in args.triples_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"ReLiK input JSONL not found: {missing}")

    reports = [
        process_file(
            triples_path=triples_path,
            sampled_dir=args.sampled_dir,
            final_dir=args.final_dir,
            validation_dir=args.validation_dir,
            log_dir=args.log_dir,
            process_script=args.process_script,
            map_script=args.map_script,
            validator_script=args.validator_script,
            top_k=args.top_k,
            privacy_rate=args.privacy_rate,
            seed=args.seed,
            spacy_model=args.spacy_model,
        )
        for triples_path in args.triples_inputs
    ]
    result = {
        "schema": "eraser4rag-rl-postprocess-v1",
        "top_k": args.top_k,
        "privacy_rate": args.privacy_rate,
        "seed": args.seed,
        "spacy_model": args.spacy_model,
        "datasets": reports,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
