from pathlib import Path

from scripts.restore_sft_dataset import (
    ARCHIVE_SHA256,
    DATASET_SHA256,
    REPO_ROOT,
    extraction_command,
    find_extractor,
    main,
    sha256,
)


def test_tracked_sft_archive_has_expected_hash():
    archive = REPO_ROOT / "dataset/constructed_dataset/popqa_10_25_filtered_new.rar"
    assert archive.is_file()
    assert sha256(archive) == ARCHIVE_SHA256


def test_restore_is_idempotent_when_expected_dataset_exists(capsys):
    output = REPO_ROOT / "dataset/constructed_dataset/popqa_10_25_filtered_new.jsonl"
    assert output.is_file()
    assert main([]) == 0
    assert "already exists" in capsys.readouterr().out
    assert sha256(output) == DATASET_SHA256


def test_extraction_command_supports_linux_7zip(tmp_path):
    command = extraction_command("7zz", tmp_path / "data.rar", tmp_path / "out")
    assert command[:2] == ["7zz", "x"]
    assert command[-1] == "popqa_10_25_filtered_new.jsonl"


def test_extractor_discovery_prefers_unrar_for_rar5(monkeypatch):
    available = {"unrar": "/usr/bin/unrar", "7z": "/usr/bin/7z"}
    monkeypatch.setattr(
        "scripts.restore_sft_dataset.shutil.which", available.get
    )
    assert find_extractor() == "unrar"
