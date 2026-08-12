"""Pure-CPU regression tests for the reconstructed data pipeline."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest


# ``utils.process_triplets`` only needs spaCy inside its CLI ``main``.  Keep
# these pure-function tests runnable before the dedicated NLP environments are
# installed.
if importlib.util.find_spec("spacy") is None:
    sys.modules.setdefault("spacy", types.ModuleType("spacy"))

from get_special_data import select_special_ctx
from utils.add_sampled_data import main as add_sampled_main
from utils.add_sampled_data import match_triplets
from utils.process_triplets import sample_privacy
from utils.triple_utils import serialize_triple


class FixedRng:
    """Minimal deterministic stand-in for ``random.Random``."""

    def __init__(self, sampled_indices):
        self.sampled_indices = list(sampled_indices)

    def sample(self, population, sample_size):
        assert sample_size == len(self.sampled_indices)
        assert set(self.sampled_indices).issubset(set(population))
        return list(self.sampled_indices)


def test_qa_harming_initial_sample_returns_to_public_candidates():
    qa_triple = ["Question Entity", "answer relation", "Answer Entity"]
    other_triple = ["Public Head", "public relation", "Public Tail"]

    public, private, flag, stats = sample_privacy(
        [qa_triple, other_triple],
        privacy_rate=0.5,
        query_entity=["Question Entity"],
        answers=["Answer Entity"],
        flag=0,
        rng=FixedRng([0]),
        return_stats=True,
    )

    assert tuple(qa_triple) in public
    assert private == []
    assert flag == 1
    assert stats["qa_rejected"] == 1
    # The rejected sample is not silently lost from both graph partitions.
    assert set(public) | set(private) == {tuple(qa_triple), tuple(other_triple)}


def test_privacy_promotions_rebuild_graph_until_fixed_point():
    triples = [
        ["A", "r", "B"],
        ["B", "q", "D"],
        # This appears before the alias edge and is not connected initially.
        ["A full", "s", "D"],
        # Same relation-tail plus substring-compatible head promotes first.
        ["A full", "r", "B"],
        ["X", "u", "Y"],
    ]

    public, private, _, stats = sample_privacy(
        triples,
        privacy_rate=0.4,
        query_entity=[],
        answers=[],
        flag=0,
        rng=FixedRng([0, 1]),
        return_stats=True,
    )

    assert ("A full", "r", "B") in private
    # Requires a second pass after the preceding promotion updates the graph.
    assert ("A full", "s", "D") in private
    assert stats["private_promoted"] == 2
    assert public == [("X", "u", "Y")]
    assert set(public) | set(private) == {tuple(triple) for triple in triples}


def test_local_mapping_normalizes_doc_id_and_preserves_extraction_order():
    triplets_by_id = {
        "7": [
            ["private head", "r0", "private tail"],
            ["public second", "r2", "tail 2"],
            ["public first", "r1", "tail 1"],
        ]
    }
    contexts = [{"id": 7, "text": "document"}]

    mapped = match_triplets(
        triplets_by_id,
        public_items=[
            ["public first", "r1", "tail 1"],
            ["public second", "r2", "tail 2"],
        ],
        private_items=[["private head", "r0", "private tail"]],
        contexts=contexts,
    )

    assert mapped[0]["private"] == [
        serialize_triple(["private head", "r0", "private tail"])
    ]
    # Global membership order differs; output follows per-document extraction.
    assert mapped[0]["public"] == [
        serialize_triple(["public second", "r2", "tail 2"]),
        serialize_triple(["public first", "r1", "tail 1"]),
    ]


def _write_jsonl(path: Path, records) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_add_sampled_data_rejects_unequal_input_lengths(tmp_path):
    extraction_record = {
        "question": "q1",
        "answers": ["a1"],
        "ctxs": [{"id": 1, "text": "document"}],
        "triplets": [{"1": [["h", "r", "t"]]}],
    }
    sampled_record = {
        "question": "q1",
        "answers": ["a1"],
        "ctxs": [{"id": 1, "text": "document"}],
        "public": [["h", "r", "t"]],
        "privacy": [],
    }
    triples_input = tmp_path / "triples.jsonl"
    sampled_input = tmp_path / "sampled.jsonl"
    output = tmp_path / "output.jsonl"
    _write_jsonl(triples_input, [extraction_record, extraction_record])
    _write_jsonl(sampled_input, [sampled_record])

    with pytest.raises(ValueError, match="different record counts"):
        add_sampled_main(
            Namespace(
                triples_input=str(triples_input),
                sampled_input=str(sampled_input),
                output_path=str(output),
                max_samples=None,
            )
        )


def test_d_special_selects_shared_tail_with_different_head_or_relation():
    shared_tail = "shared tail"
    selected_context = {
        "private": [serialize_triple(["private head", "private r", shared_tail])],
        "public": [serialize_triple(["public head", "public r", shared_tail])],
    }
    rejected_context = {
        "private": [serialize_triple(["same head", "same r", shared_tail])],
        "public": [serialize_triple(["same head", "same r", shared_tail])],
    }
    record = {
        "ctxs": [selected_context, rejected_context],
        "anonymized_text": ["selected rewrite", "rejected rewrite"],
    }

    selected, count = select_special_ctx(record)

    assert selected is not None
    assert count == 1
    assert selected["ctxs"] == [selected_context]
    assert selected["anonymized_text"] == ["selected rewrite"]
