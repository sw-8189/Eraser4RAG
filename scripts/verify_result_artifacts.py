"""Verify compressed JSONL result artifacts against a recorded manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def inspect_gzip(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    records = 0
    with gzip.open(path, "rb") as source:
        for line in source:
            digest.update(line)
            if line.strip():
                records += 1
    return digest.hexdigest(), records


def expected_artifacts(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    schema = manifest.get("schema")
    if schema == "eraser4rag-relik-files-v1":
        return [
            {
                "file": f"{Path(entry['output']).name}.gz",
                "uncompressed_sha256": entry["output_sha256"],
                "records": entry["output_records"],
            }
            for entry in manifest.get("inputs", [])
        ]
    if schema == "eraser4rag-compressed-jsonl-artifacts-v1":
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("compressed artifact manifest must contain a list")
        return artifacts
    raise ValueError(f"unexpected artifact manifest schema: {schema!r}")


def verify_artifacts(artifact_dir: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = []
    valid = True
    for entry in expected_artifacts(manifest):
        artifact_path = artifact_dir / entry["file"]
        if not artifact_path.is_file():
            raise FileNotFoundError(f"missing compressed artifact: {artifact_path}")
        actual_sha256, actual_records = inspect_gzip(artifact_path)
        expected_sha256 = entry["uncompressed_sha256"]
        expected_records = entry["records"]
        compressed_bytes = artifact_path.stat().st_size
        expected_compressed_bytes = entry.get("compressed_bytes")
        artifact_valid = (
            actual_sha256 == expected_sha256
            and actual_records == expected_records
            and (
                expected_compressed_bytes is None
                or compressed_bytes == expected_compressed_bytes
            )
        )
        valid = valid and artifact_valid
        artifacts.append(
            {
                "file": artifact_path.name,
                "compressed_bytes": compressed_bytes,
                "expected_compressed_bytes": expected_compressed_bytes,
                "uncompressed_sha256": actual_sha256,
                "expected_sha256": expected_sha256,
                "records": actual_records,
                "expected_records": expected_records,
                "valid": artifact_valid,
            }
        )

    return {
        "schema": "eraser4rag-result-artifact-verification-v1",
        "valid": valid,
        "artifacts": artifacts,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    report = verify_artifacts(args.artifact_dir, args.manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
