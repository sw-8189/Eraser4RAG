"""Small, dependency-light helpers for Eraser4RAG YAML configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Union


class ConfigError(ValueError):
    """Raised when a configuration file is missing or structurally invalid."""


_MISSING = object()


def load_yaml_config(path: Union[str, Path]) -> dict[str, Any]:
    """Load a YAML mapping without mutating it or applying implicit defaults."""

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to load reproduction configuration"
        ) from exc

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, Mapping):
        raise ConfigError("Top-level YAML value must be a mapping")
    return dict(loaded)


def get_config_value(
    config: Mapping[str, Any], dotted_key: str, default: Any = _MISSING
) -> Any:
    """Read a nested setting such as ``rl.gamma`` from a mapping."""

    if not isinstance(config, Mapping):
        raise ConfigError("config must be a mapping")
    if not isinstance(dotted_key, str) or not dotted_key or any(
        not part for part in dotted_key.split(".")
    ):
        raise ConfigError("dotted_key must contain non-empty path components")

    value: Any = config
    for component in dotted_key.split("."):
        if not isinstance(value, Mapping) or component not in value:
            if default is not _MISSING:
                return default
            raise ConfigError(f"Missing required configuration key: {dotted_key}")
        value = value[component]
    return value


def config_get(
    config: Mapping[str, Any], dotted_path: str, default: Any = None
) -> Any:
    """Return a dotted-path value, or ``default`` when it is absent.

    This is the stable permissive accessor for training and data scripts.
    Use :func:`get_config_value` when a missing setting must raise an error.
    """

    return get_config_value(config, dotted_path, default=default)


def resolve_path(
    value: Union[str, Path], base_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Resolve a configured path relative to ``base_dir`` or the current cwd."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        base = Path.cwd() if base_dir is None else Path(base_dir).expanduser()
        path = base / path
    return path.resolve(strict=False)


def validate_reproduction_config(config: Mapping[str, Any]) -> None:
    """Validate settings that are hard requirements of the v2 reproduction."""

    required_keys = (
        "seed",
        "paths.data_root",
        "datasets.sft",
        "models.flan_t5_large",
        "models.relik",
        "model_revisions.flan_t5_large",
        "model_revisions.relik",
        "model_paths.sft",
        "model_paths.relik",
        "retrieval.top_k",
        "privacy.sampling_rate",
        "sft.epochs",
        "sft.learning_rate",
        "rl.batch_size",
        "rl.mini_batch_size",
        "rl.ppo_epochs",
        "rl.learning_rate",
        "rl.gamma",
        "rl.initial_p",
        "rl.p_increment",
        "rl.p_interval",
        "rl.p_max",
        "rl.max_steps",
    )
    for key in required_keys:
        get_config_value(config, key)

    if get_config_value(config, "retrieval.top_k") <= 0:
        raise ConfigError("retrieval.top_k must be positive")
    sampling_rate = get_config_value(config, "privacy.sampling_rate")
    if not 0.0 <= sampling_rate <= 1.0:
        raise ConfigError("privacy.sampling_rate must be within [0, 1]")
    if get_config_value(config, "rl.gamma") != 0.99:
        raise ConfigError("rl.gamma must be 0.99 for the v2 reproduction")
    if get_config_value(config, "rl.p_max") != 40:
        raise ConfigError("rl.p_max must be 40 for the v2 reproduction")

    for stage in ("flan_t5_base", "flan_t5_large", "relik", "contriever", "llama3"):
        revision = get_config_value(config, f"model_revisions.{stage}")
        if not isinstance(revision, str) or len(revision) != 40:
            raise ConfigError(f"model_revisions.{stage} must be a 40-character commit SHA")


__all__ = [
    "ConfigError",
    "config_get",
    "get_config_value",
    "load_yaml_config",
    "resolve_path",
    "validate_reproduction_config",
]
