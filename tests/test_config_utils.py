from pathlib import Path
import builtins

import pytest

from utils.config_utils import (
    ConfigError,
    config_get,
    get_config_value,
    load_yaml_config,
    resolve_path,
    validate_reproduction_config,
)
import scripts.check_environment as environment_check
from scripts.check_environment import version_matches


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "reproduction.yaml"


def test_repository_config_loads_and_validates():
    config = load_yaml_config(CONFIG_PATH)
    validate_reproduction_config(config)
    assert get_config_value(config, "seed") == 42
    assert get_config_value(config, "rl.gamma") == 0.99
    assert get_config_value(config, "rl.p_max") == 40
    assert get_config_value(config, "rl.max_steps") == 200000
    assert set(get_config_value(config, "datasets.rl")) == {
        "popqa",
        "triviaqa",
        "nq_open",
        "hotpotqa",
    }
    assert get_config_value(config, "environment.torch_cuda") == "12.1"
    assert get_config_value(config, "model_revisions.flan_t5_large") == (
        "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
    )
    assert get_config_value(config, "datasets.evaluation.special").endswith(
        "dataset/special_data/popqa_10_25_special.jsonl"
    )
    assert get_config_value(config, "artifacts.rewritten.inference_attack").endswith(
        "outputs/rewritten/popqa_10_25_inferattack.jsonl"
    )


def test_get_config_value_supports_default_and_reports_missing_key():
    config = {"outer": {"inner": 3}}
    assert get_config_value(config, "outer.inner") == 3
    assert get_config_value(config, "outer.missing", default=7) == 7
    with pytest.raises(ConfigError):
        get_config_value(config, "outer.missing")


def test_config_get_is_permissive_and_defaults_to_none():
    config = {"outer": {"inner": 3}}
    assert config_get(config, "outer.inner") == 3
    assert config_get(config, "outer.missing") is None
    assert config_get(config, "outer.missing", default="fallback") == "fallback"


@pytest.mark.parametrize("key", ["", ".leading", "trailing.", "two..dots"])
def test_get_config_value_rejects_invalid_dotted_keys(key):
    with pytest.raises(ConfigError):
        get_config_value({}, key)


def test_resolve_path_uses_explicit_base_directory(tmp_path):
    assert resolve_path("data/file.jsonl", tmp_path) == (
        tmp_path / "data" / "file.jsonl"
    ).resolve()


def test_load_yaml_config_rejects_non_mapping(tmp_path):
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_yaml_config(invalid)


def test_environment_version_comparison_accepts_cuda_local_suffix():
    assert version_matches("2.3.1+cu121", "2.3.1")
    assert version_matches("2.3.1", "2.3.1")
    assert not version_matches("2.3.0+cu121", "2.3.1")
    assert not version_matches(None, "2.3.1")


def test_environment_check_marks_torch_import_failure(monkeypatch):
    monkeypatch.setattr(
        environment_check,
        "package_version",
        lambda name: environment_check.MAIN_PACKAGES.get(name),
    )
    real_import = builtins.__import__

    def failing_torch_import(name, *args, **kwargs):
        if name == "torch":
            raise OSError("simulated DLL load failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_torch_import)
    report = environment_check.check_environment(
        "main", cpu_only=True, load_pipeline=False
    )

    assert report["status"] == "fail"
    assert any(item["name"] == "torch_import" for item in report["checks"])
