"""Download one explicitly selected reproduction model snapshot.

There is deliberately no implicit ``all`` stage.  In particular, Llama 3 is
downloaded only when ``--stage eval`` is explicitly requested.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fnmatch import fnmatch
import json
from pathlib import Path
import re
import sys
from typing import Iterable


MODEL_STAGES = {
    "smoke": "google/flan-t5-base",
    "sft": "google/flan-t5-large",
    "relik": "relik-ie/relik-relation-extraction-small",
    "retrieval": "facebook/contriever-msmarco",
    "eval": "meta-llama/Meta-Llama-3-8B-Instruct",
}

MODEL_CONFIG_KEYS = {
    "smoke": "flan_t5_base",
    "sft": "flan_t5_large",
    "relik": "relik",
    "retrieval": "contriever",
    "eval": "llama3",
}

# Flan repositories publish equivalent weights for several frameworks.  This
# project loads them only through Transformers/PyTorch, so downloading all
# formats wastes roughly three model copies per snapshot.
FLAN_ALLOW_PATTERNS = (
    ".gitattributes",
    "README.md",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
)

MODEL_ALLOW_PATTERNS = {
    "smoke": FLAN_ALLOW_PATTERNS,
    "sft": FLAN_ALLOW_PATTERNS,
}

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _config_revision(config_path: str | Path, stage: str) -> str | None:
    from utils.config_utils import config_get, load_yaml_config

    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        return None
    return config_get(
        load_yaml_config(path),
        f"model_revisions.{MODEL_CONFIG_KEYS[stage]}",
    )


def _is_immutable_revision(value: str | None) -> bool:
    return bool(value and re.fullmatch(r"[0-9a-f]{40}", value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(MODEL_STAGES), required=True)
    parser.add_argument("--model-root", default=str(REPO_ROOT / "models"))
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "reproduction.yaml")
    )
    parser.add_argument("--revision")
    parser.add_argument(
        "--allow-floating-revision",
        action="store_true",
        help="Resolve the current Hub head once and record its immutable commit.",
    )
    parser.add_argument("--token", help="Hugging Face token; prefer HF_TOKEN env instead")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _matches_allow_patterns(
    relative_path: str, allow_patterns: tuple[str, ...]
) -> bool:
    return any(fnmatch(relative_path, pattern) for pattern in allow_patterns)


def validate_partial_snapshot(
    destination: Path,
    resolved_revision: str,
    allow_patterns: tuple[str, ...] | None,
) -> None:
    """Permit only an identifiable Hub local-dir download to resume.

    Hugging Face stores the resolved commit on the first line of each completed
    file's ``.metadata`` entry. Incomplete LFS blobs are content-addressed by
    their ETag, so the requested snapshot can safely reuse a matching blob and
    ignores stale blobs with other names.
    """

    cache_dir = destination / ".cache" / "huggingface" / "download"
    metadata_paths = sorted(cache_dir.glob("*.metadata"))
    if not metadata_paths:
        raise FileExistsError(
            f"refusing to merge into non-empty directory without Hub metadata: {destination}"
        )
    revisions = set()
    for metadata_path in metadata_paths:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise FileExistsError(f"empty Hub metadata file: {metadata_path}")
        revisions.add(lines[0].strip())
    if revisions != {resolved_revision}:
        raise FileExistsError(
            f"partial snapshot revision mismatch in {destination}: {sorted(revisions)}"
        )

    if allow_patterns is None:
        return
    visible_files = [
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(destination).parts
    ]
    unexpected = sorted(
        path
        for path in visible_files
        if not _matches_allow_patterns(path, allow_patterns)
    )
    if unexpected:
        raise FileExistsError(
            f"partial snapshot contains files outside the stage download scope: {unexpected}"
        )


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_id = MODEL_STAGES[args.stage]
    selected_revision = args.revision or _config_revision(args.config, args.stage)
    destination = Path(args.model_root) / args.stage
    allow_patterns = MODEL_ALLOW_PATTERNS.get(args.stage)
    plan = {
        "stage": args.stage,
        "repo_id": repo_id,
        "destination": str(destination.resolve()),
        "revision": selected_revision,
        "allow_patterns": list(allow_patterns) if allow_patterns else None,
        "dry_run": args.dry_run,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    if not selected_revision and not args.allow_floating_revision:
        raise ValueError(
            "--revision is required for a reproducible download. Pass an immutable "
            "commit SHA, or explicitly acknowledge a one-time head resolution with "
            "--allow-floating-revision."
        )
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is unavailable; run this in eraser-main after installing requirements"
        ) from exc
    requested_revision = selected_revision or "main"
    # A 40-character Hub commit is already immutable. Avoid an unnecessary
    # API round trip so interrupted downloads can resume during brief Hub API
    # outages; floating refs still resolve through model_info as before.
    if _is_immutable_revision(requested_revision):
        resolved_revision = requested_revision
    else:
        model_info = HfApi().model_info(
            repo_id=repo_id,
            revision=requested_revision,
            token=args.token,
        )
        resolved_revision = model_info.sha
    if not resolved_revision:
        raise RuntimeError(f"could not resolve an immutable revision for {repo_id}")

    manifest_path = destination / "download_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("repo_id") != repo_id
            or existing.get("resolved_revision") != resolved_revision
        ):
            raise FileExistsError(
                f"{destination} already contains a different recorded model snapshot"
            )
    elif destination.exists() and any(destination.iterdir()):
        validate_partial_snapshot(destination, resolved_revision, allow_patterns)

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=destination,
        revision=resolved_revision,
        token=args.token,
        allow_patterns=allow_patterns,
    )
    manifest = {
        "schema": "eraser4rag-model-download-v1",
        "stage": args.stage,
        "repo_id": repo_id,
        "requested_revision": selected_revision,
        "floating_revision_explicitly_allowed": args.allow_floating_revision,
        "resolved_revision": resolved_revision,
        "allow_patterns": list(allow_patterns) if allow_patterns else None,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "destination": str(destination.resolve()),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Downloaded {repo_id} to {destination}")
    print(f"Resolved revision: {resolved_revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
