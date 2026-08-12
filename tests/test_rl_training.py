import json
import hashlib
from types import SimpleNamespace

import pytest

import RL_train
from utils.triple_utils import serialize_triple


def _triple(head, relation, tail):
    return SimpleNamespace(
        subject=SimpleNamespace(text=head),
        label=relation,
        object=SimpleNamespace(text=tail),
    )


def test_dynamic_reward_extraction_uses_gate_duplicate_rule():
    output = SimpleNamespace(
        triplets=[
            _triple("Alice", "knows", "Bob"),
            _triple("Bob", "works_with", "Alice"),
            _triple("Alice", "self", "Alice"),
        ]
    )

    predicted = RL_train.extract_generated_triples(lambda _: [output], ["rewrite"])

    assert predicted == [{serialize_triple(["Alice", "knows", "Bob"])}]


def _gate_report(tmp_path, *, status="pass", sample_count=2):
    data_file = tmp_path / "author-sft.jsonl"
    data_file.write_text("{}\n", encoding="utf-8")
    data_path = str(data_file)
    data_sha256 = hashlib.sha256(data_file.read_bytes()).hexdigest()
    manifest = tmp_path / "sample_indices.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "eraser4rag-relik-sample-indices-v1",
                "data_path": data_path,
                "data_sha256": data_sha256,
                "seed": 42,
                "actual_sample_size": sample_count,
                "indices": list(range(sample_count)),
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "eraser4rag-relik-consistency-v1",
                "data_path": data_path,
                "seed": 42,
                "gate_status": status,
                "metrics": {"sample_count": sample_count},
                "sample_indices_path": str(manifest),
            }
        ),
        encoding="utf-8",
    )
    return report


def test_ppo_gate_requires_report_manifest_and_minimum_sample_count(tmp_path):
    with pytest.raises(FileNotFoundError):
        RL_train.validate_relik_gate_report(
            tmp_path / "missing.json", minimum_samples=2, allow_warning=False
        )

    report = _gate_report(tmp_path, sample_count=2)
    summary = RL_train.validate_relik_gate_report(
        report, minimum_samples=2, allow_warning=False
    )
    assert summary["status"] == "pass"
    assert summary["sample_count"] == 2

    with pytest.raises(ValueError, match="at least 3"):
        RL_train.validate_relik_gate_report(
            report, minimum_samples=3, allow_warning=False
        )


def test_warning_gate_needs_an_explicit_documented_override(tmp_path):
    report = _gate_report(tmp_path, status="warning", sample_count=2)
    with pytest.raises(RuntimeError, match="WARNING"):
        RL_train.validate_relik_gate_report(
            report, minimum_samples=2, allow_warning=False
        )

    summary = RL_train.validate_relik_gate_report(
        report, minimum_samples=2, allow_warning=True
    )
    assert summary["warning_override"] is True


def test_missing_explicit_config_is_not_silently_ignored(tmp_path):
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        RL_train.build_parser(["--config", str(missing), "--dry-run"])


def test_static_penalty_cannot_exceed_v2_cap():
    argv = ["--dry-run", "--static-p", "--initial-p", "45"]
    args = RL_train.build_parser(argv).parse_args(argv)
    with pytest.raises(ValueError, match="initial_p"):
        RL_train.run(args)
