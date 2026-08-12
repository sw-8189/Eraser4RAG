import json

import pytest

from scripts.verify_models import validate_local_manifest


def _write_manifest(path, **overrides):
    payload = {
        "stage": "sft",
        "repo_id": "google/flan-t5-large",
        "resolved_revision": "a" * 40,
    }
    payload.update(overrides)
    path.mkdir()
    (path / "download_manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_local_manifest_must_match_stage_repo_and_revision(tmp_path):
    model = tmp_path / "model"
    _write_manifest(model)
    manifest = validate_local_manifest(model, "sft", "a" * 40)
    assert manifest["repo_id"] == "google/flan-t5-large"


def test_local_manifest_rejects_wrong_revision(tmp_path):
    model = tmp_path / "model"
    _write_manifest(model)
    with pytest.raises(ValueError, match="revision mismatch"):
        validate_local_manifest(model, "sft", "b" * 40)


def test_local_model_requires_manifest(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        validate_local_manifest(model, "sft", "a" * 40)
