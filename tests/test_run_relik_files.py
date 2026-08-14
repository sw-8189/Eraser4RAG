from __future__ import annotations

from pathlib import Path

from scripts.run_relik_files import output_path_for


def test_output_path_for_removes_clean_retrieval_suffix():
    assert output_path_for(
        Path("input/popqa_train_retrieved_clean.jsonl"), Path("output")
    ) == Path("output/popqa_train_with_triplets.jsonl")


def test_output_path_for_keeps_other_stems():
    assert output_path_for(Path("input/custom.jsonl"), Path("output")) == Path(
        "output/custom_with_triplets.jsonl"
    )
