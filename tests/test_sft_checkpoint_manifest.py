import hashlib
import json

import pytest

from scripts.verify_models import validate_sft_training_manifest
from utils.triple_utils import SPECIAL_TOKENS


def _write_checkpoint(path):
    path.mkdir()
    files = {}
    for name, content in (
        ("model.safetensors", b"model"),
        ("tokenizer.json", b"tokenizer"),
    ):
        (path / name).write_bytes(content)
        files[name] = {
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    manifest = {
        "schema": "eraser4rag-sft-checkpoint-v1",
        "special_tokens": list(SPECIAL_TOKENS),
        "files": files,
    }
    (path / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_sft_checkpoint_manifest_validates_hashes_and_tokens(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    manifest = validate_sft_training_manifest(checkpoint)
    assert manifest["schema"] == "eraser4rag-sft-checkpoint-v1"


def test_sft_checkpoint_manifest_rejects_tampered_weight(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    (checkpoint / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        validate_sft_training_manifest(checkpoint)


def test_sft_checkpoint_manifest_rejects_token_contract_change(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    path = checkpoint / "training_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["special_tokens"] = []
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="special token"):
        validate_sft_training_manifest(checkpoint)
