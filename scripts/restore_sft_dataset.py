"""Restore and verify the tracked author SFT archive.

The JSONL is intentionally ignored because it is a generated 108 MB artifact;
the small RAR is the reproducible repository input.  This command is safe to
rerun and refuses to overwrite a mismatching existing JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = "b32d7e45c457d27e398b69d58678fea5bc40271445d798c494c016a119493186"
DATASET_SHA256 = "c215e99f5e70b8cab8fa3631fb3b8708b90506527bc5d350fa6f8d658dcb9f36"
EXPECTED_MEMBER = "popqa_10_25_filtered_new.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=str(REPO_ROOT / "dataset/constructed_dataset/popqa_10_25_filtered_new.rar"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "dataset/constructed_dataset"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--extractor",
        help="RAR-capable executable; auto-detects 7zz, 7z, unrar, bsdtar, or tar",
    )
    return parser


def extraction_command(executable: str, archive: Path, output_dir: Path) -> list[str]:
    name = Path(executable).name.lower()
    if name in {"7z", "7z.exe", "7zz", "7zz.exe"}:
        return [executable, "x", str(archive), f"-o{output_dir}", "-y", EXPECTED_MEMBER]
    if name in {"unrar", "unrar.exe"}:
        return [executable, "x", "-o+", str(archive), EXPECTED_MEMBER, str(output_dir)]
    return [executable, "-xf", str(archive), "-C", str(output_dir), EXPECTED_MEMBER]


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive = Path(args.archive).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"SFT archive not found: {archive}")
    actual_archive_sha = sha256(archive)
    if actual_archive_sha != ARCHIVE_SHA256:
        raise ValueError(
            f"archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, got {actual_archive_sha}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / EXPECTED_MEMBER
    if output.exists() and not args.force:
        actual_dataset_sha = sha256(output)
        if actual_dataset_sha == DATASET_SHA256:
            print(f"SFT dataset already exists: {output}")
            return 0
        raise FileExistsError(
            f"refusing to overwrite existing output with SHA-256 {actual_dataset_sha}: {output}"
        )
    extractor = args.extractor
    if extractor is None:
        extractor = next(
            (candidate for candidate in ("7zz", "7z", "unrar", "bsdtar", "tar") if shutil.which(candidate)),
            None,
        )
    if extractor is None:
        raise RuntimeError("no RAR-capable extractor found; install p7zip-full on AutoDL")
    command = extraction_command(extractor, archive, output_dir)
    subprocess.run(command, check=True)
    if not output.is_file():
        raise RuntimeError(f"archive did not produce expected member: {output}")
    actual_dataset_sha = sha256(output)
    if actual_dataset_sha != DATASET_SHA256:
        raise ValueError(
            f"restored dataset SHA-256 mismatch: expected {DATASET_SHA256}, got {actual_dataset_sha}"
        )
    print(f"Restored {output}")
    print(f"SHA-256: {actual_dataset_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
