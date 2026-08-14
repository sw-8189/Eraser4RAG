from __future__ import annotations

from scripts.run_coreferee_shards import (
    Shard,
    build_shards,
    count_jsonl_records,
    merge_jsonl_shards,
)


def test_build_shards_covers_all_records_in_stable_ranges():
    assert build_shards(7, 3) == [
        Shard(index=0, start=0, count=3),
        Shard(index=1, start=3, count=3),
        Shard(index=2, start=6, count=1),
    ]


def test_merge_jsonl_shards_preserves_source_order_and_ignores_blank_lines(tmp_path):
    first = tmp_path / "0000.jsonl"
    second = tmp_path / "0001.jsonl"
    output = tmp_path / "merged.jsonl"
    first.write_text('{"index":0}\n\n{"index":1}\n', encoding="utf-8")
    second.write_text('{"index":2}\n', encoding="utf-8")

    assert merge_jsonl_shards([first, second], output) == 3
    assert output.read_text(encoding="utf-8") == '{"index":0}\n{"index":1}\n{"index":2}\n'
    assert count_jsonl_records(output) == 3
