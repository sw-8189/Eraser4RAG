import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import test_inferattack
import test_special


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def manual_seed_all(_seed):
            return None

        @staticmethod
        def empty_cache():
            return None

    @staticmethod
    def manual_seed(_seed):
        return None

    @staticmethod
    def no_grad():
        return nullcontext()


def output_with_triplets(*triples):
    return SimpleNamespace(
        triplets=[
            SimpleNamespace(
                subject=SimpleNamespace(text=head),
                label=relation,
                object=SimpleNamespace(text=tail),
            )
            for head, relation, tail in triples
        ]
    )


def install_fake_relik(monkeypatch, module, output):
    class FakeRelik:
        @staticmethod
        def from_pretrained(_model, *, device, use_nme):
            assert device == "cpu"
            assert use_nme is True

            def predict(texts):
                if isinstance(texts, str):
                    return output
                return [output for _ in texts]

            return predict

    monkeypatch.setattr(module, "torch", FakeTorch)
    monkeypatch.setattr(module, "Relik", FakeRelik)


def test_special_writes_machine_readable_metrics(tmp_path, monkeypatch):
    install_fake_relik(
        monkeypatch,
        test_special,
        output_with_triplets(("public-head", "relation", "public-tail")),
    )
    data_path = tmp_path / "special.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "ctxs": [
                    {
                        "public": [["public-head", "relation", "public-tail"]],
                        "private": [["private-head", "relation", "private-tail"]],
                    }
                ],
                "anonymized_ctxs": ["rewritten text"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "metrics" / "special.json"
    args = test_special.build_parser().parse_args(
        [
            "--data",
            str(data_path),
            "--relik_model",
            "fake-relik",
            "--device",
            "cpu",
            "--output-json",
            str(output_path),
        ]
    )

    result = test_special.main(args)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["r_pub"] == pytest.approx(1.0)
    assert result["r_pri"] == pytest.approx(0.0)
    assert payload["schema"] == "eraser4rag-evaluation-v1"
    assert payload["evaluator"] == "test_special.py"
    assert payload["metrics"] == result


def test_inferattack_writes_machine_readable_metrics(tmp_path, monkeypatch):
    install_fake_relik(
        monkeypatch,
        test_inferattack,
        output_with_triplets(
            ("private-head", "edge", "middle"),
            ("middle", "edge", "private-tail"),
        ),
    )
    data_path = tmp_path / "inferattack.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "privacy": [["private-head", "relation", "private-tail"]],
                "anonymized_ctxs": ["rewritten text"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "metrics" / "inferattack.json"
    args = test_inferattack.build_parser().parse_args(
        [
            "--data",
            str(data_path),
            "--relik_model",
            "fake-relik",
            "--device",
            "cpu",
            "--output-json",
            str(output_path),
        ]
    )

    result = test_inferattack.main(args)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["r_connect_macro"] == pytest.approx(1.0)
    assert result["r_connect_micro"] == pytest.approx(1.0)
    assert payload["schema"] == "eraser4rag-evaluation-v1"
    assert payload["evaluator"] == "test_inferattack.py"
    assert payload["metrics"] == result
