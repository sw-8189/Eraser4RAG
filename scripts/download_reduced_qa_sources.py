"""Download pinned PopQA and HotpotQA sources for the reduced experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


POPQA_REPO = "MinaGabriel/popqa-with-retrieval-20"
POPQA_REVISION = "dcc3f4f72fab2f7bca386c51e5cf329109727919"
HOTPOTQA_REPO = "hotpotqa/hotpot_qa"
HOTPOTQA_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_plan(output_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "name": "popqa_train",
            "repo_id": POPQA_REPO,
            "config": None,
            "split": "train",
            "revision": POPQA_REVISION,
            "output": str((output_dir / "popqa_train.parquet").resolve()),
        },
        {
            "name": "hotpotqa_train",
            "repo_id": HOTPOTQA_REPO,
            "config": "distractor",
            "split": "train",
            "revision": HOTPOTQA_REVISION,
            "output": str((output_dir / "hotpotqa_train.parquet").resolve()),
        },
        {
            "name": "hotpotqa_validation",
            "repo_id": HOTPOTQA_REPO,
            "config": "distractor",
            "split": "validation",
            "revision": HOTPOTQA_REVISION,
            "output": str((output_dir / "hotpotqa_validation.parquet").resolve()),
        },
    ]


def download_split(item: dict[str, Any], *, cache_dir: Path | None, force: bool) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required; use the pinned eraser-main environment") from exc

    output_path = Path(item["output"])
    if output_path.exists() and not force:
        raise FileExistsError(
            f"source output already exists: {output_path}; pass --force to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "path": item["repo_id"],
        "split": item["split"],
        "revision": item["revision"],
    }
    if item["config"] is not None:
        kwargs["name"] = item["config"]
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    dataset = load_dataset(**kwargs)

    temporary = output_path.with_suffix(".tmp.parquet")
    try:
        dataset.to_parquet(str(temporary))
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        **item,
        "records": len(dataset),
        "size_bytes": output_path.stat().st_size,
        "sha256": sha256(output_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/source_data/two_dataset_v1"),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/manifests/two_dataset_sources.json"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    plan = source_plan(args.output_dir)
    if args.dry_run:
        print(json.dumps({"schema": "eraser4rag-source-plan-v1", "sources": plan}, indent=2))
        return 0

    reports = [
        download_split(item, cache_dir=args.cache_dir, force=args.force)
        for item in plan
    ]
    result = {
        "schema": "eraser4rag-source-manifest-v1",
        "sources": reports,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
