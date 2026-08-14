from __future__ import annotations

from pathlib import Path

from scripts.run_rl_postprocess import dataset_name_for


def test_dataset_name_for_removes_relik_suffix():
    assert dataset_name_for(Path("popqa_train_with_triplets.jsonl")) == "popqa_train"


def test_dataset_name_for_keeps_custom_stem():
    assert dataset_name_for(Path("custom.jsonl")) == "custom"
