import json
from pathlib import Path
import sys
import types

import pytest

from scripts.download_models import (
    FLAN_ALLOW_PATTERNS,
    main,
    validate_partial_snapshot,
)


REVISION = "a" * 40


def _partial_snapshot(path: Path, revision: str = REVISION) -> None:
    cache = path / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "config.json.metadata").write_text(
        f"{revision}\netag\ntimestamp\n", encoding="utf-8"
    )
    (path / "config.json").write_text("{}", encoding="utf-8")


def test_flan_scope_uses_only_safetensors_weights():
    assert "model.safetensors" in FLAN_ALLOW_PATTERNS
    assert "pytorch_model.bin" not in FLAN_ALLOW_PATTERNS
    assert "tf_model.h5" not in FLAN_ALLOW_PATTERNS
    assert "flax_model.msgpack" not in FLAN_ALLOW_PATTERNS


def test_matching_hub_partial_snapshot_can_resume(tmp_path):
    destination = tmp_path / "sft"
    _partial_snapshot(destination)
    validate_partial_snapshot(destination, REVISION, FLAN_ALLOW_PATTERNS)


def test_partial_snapshot_rejects_unknown_revision(tmp_path):
    destination = tmp_path / "sft"
    _partial_snapshot(destination, "b" * 40)
    with pytest.raises(FileExistsError, match="revision mismatch"):
        validate_partial_snapshot(destination, REVISION, FLAN_ALLOW_PATTERNS)


def test_partial_snapshot_rejects_file_outside_scope(tmp_path):
    destination = tmp_path / "sft"
    _partial_snapshot(destination)
    (destination / "pytorch_model.bin").write_bytes(b"duplicate")
    with pytest.raises(FileExistsError, match="outside the stage download scope"):
        validate_partial_snapshot(destination, REVISION, FLAN_ALLOW_PATTERNS)


def test_sft_download_passes_filter_and_records_manifest(tmp_path, monkeypatch):
    destination = tmp_path / "models" / "sft"
    _partial_snapshot(destination)
    captured = {}

    class FakeApi:
        def model_info(self, **kwargs):
            return types.SimpleNamespace(sha=REVISION)

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        (destination / "model.safetensors").write_bytes(b"weights")

    fake_hub = types.SimpleNamespace(
        HfApi=FakeApi,
        snapshot_download=fake_snapshot_download,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    assert main(
        [
            "--stage",
            "sft",
            "--model-root",
            str(tmp_path / "models"),
            "--revision",
            REVISION,
        ]
    ) == 0
    assert tuple(captured["allow_patterns"]) == FLAN_ALLOW_PATTERNS
    manifest = json.loads(
        (destination / "download_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["resolved_revision"] == REVISION
    assert manifest["allow_patterns"] == list(FLAN_ALLOW_PATTERNS)


def test_fixed_revision_does_not_require_hub_model_info(tmp_path, monkeypatch):
    destination = tmp_path / "models" / "sft"
    _partial_snapshot(destination)
    captured = {}

    class FailingApi:
        def model_info(self, **kwargs):
            raise AssertionError("fixed commits must not call the Hub API")

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        (destination / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            HfApi=FailingApi, snapshot_download=fake_snapshot_download
        ),
    )
    assert main(
        [
            "--stage",
            "sft",
            "--model-root",
            str(tmp_path / "models"),
            "--revision",
            REVISION,
        ]
    ) == 0
    assert captured["revision"] == REVISION
