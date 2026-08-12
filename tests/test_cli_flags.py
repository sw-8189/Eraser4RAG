import ast
from pathlib import Path

import pytest

import RL_train
import finetune_rewrite_doc
from scripts.download_models import main as download_models_main


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_no_argparse_type_bool_in_project_sources():
    offenders = []
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in {".git", ".idea", ".venv", "venv"} for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not isinstance(function, ast.Attribute) or function.attr != "add_argument":
                continue
            for keyword in node.keywords:
                if (
                    keyword.arg == "type"
                    and isinstance(keyword.value, ast.Name)
                    and keyword.value.id == "bool"
                ):
                    offenders.append((str(path.relative_to(REPO_ROOT)), node.lineno))
    assert offenders == []


def test_rl_parser_has_hotpot_gamma_and_reliable_boolean_flags():
    argv = ["--dry-run", "--no-use-prefix", "--no-use-nme"]
    parser = RL_train.build_parser(argv)
    args = parser.parse_args(argv)

    assert args.data_dir_4.endswith("hotpotqa_10_25_trp.jsonl")
    assert args.gamma == 0.99
    assert args.p_max == 40
    assert args.max_steps == 200000
    assert args.use_prefix is False
    assert args.use_nme is False


def test_sft_parser_consumes_reproduction_config_defaults():
    args = finetune_rewrite_doc.get_args([])
    assert args.model_dir.endswith("models/sft")
    assert args.output_dir.endswith("output_checkpoint/SFT")
    assert args.epochs == 3
    assert args.learning_rate == 5e-5
    assert args.max_concat_length == 1300
    assert args.max_passage_length == 128


def test_model_download_requires_revision_or_explicit_floating_acknowledgement(tmp_path):
    with pytest.raises(ValueError, match="--revision is required"):
        download_models_main(
            [
                "--stage",
                "sft",
                "--model-root",
                str(tmp_path / "models"),
                "--config",
                str(tmp_path / "missing.yaml"),
            ]
        )


def test_model_download_dry_run_uses_frozen_config_revision(tmp_path, capsys):
    download_models_main(
        ["--stage", "sft", "--model-root", str(tmp_path / "models"), "--dry-run"]
    )
    assert "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a" in capsys.readouterr().out
