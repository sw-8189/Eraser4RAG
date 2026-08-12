"""Check the pinned Eraser4RAG main or coreference environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable


MAIN_PACKAGES = {
    "torch": "2.3.1",
    "transformers": "4.41.2",
    "trl": "0.11.4",
    "accelerate": "0.34.2",
    "relik": "1.0.7",
    "spacy": "3.7.5",
    "datasets": "2.14.7",
    "numpy": "1.26.4",
    "scikit-learn": "1.5.2",
}

EXPECTED_TORCH_CUDA = "12.1"

COREF_PACKAGES = {
    "spacy": "3.5.0",
    "coreferee": "1.4.1",
}


def version_matches(actual: str | None, expected: str) -> bool:
    """Compare public versions while accepting wheel-local build suffixes."""

    if actual is None:
        return False
    return actual.split("+", 1)[0] == expected.split("+", 1)[0]


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_environment(
    mode: str,
    *,
    cpu_only: bool = False,
    load_pipeline: bool = True,
) -> dict[str, Any]:
    expected = MAIN_PACKAGES if mode == "main" else COREF_PACKAGES
    checks: list[dict[str, Any]] = []

    python_actual = platform.python_version()
    checks.append(
        {
            "name": "python",
            "expected": "3.10.14",
            "actual": python_actual,
            "status": "pass" if python_actual == "3.10.14" else "fail",
        }
    )
    for package, wanted in expected.items():
        actual = package_version(package)
        checks.append(
            {
                "name": package,
                "expected": wanted,
                "actual": actual,
                "status": "pass" if version_matches(actual, wanted) else "fail",
            }
        )

    runtime: dict[str, Any] = {}
    if mode == "main":
        try:
            import torch

            runtime["torch_import"] = "pass"
            runtime["cuda_available"] = torch.cuda.is_available()
            runtime["cuda_version"] = torch.version.cuda
            runtime["gpu_count"] = torch.cuda.device_count()
            runtime["gpus"] = [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
            if not cpu_only:
                actual_cuda = str(torch.version.cuda or "")
                checks.append(
                    {
                        "name": "torch_cuda",
                        "expected": EXPECTED_TORCH_CUDA,
                        "actual": actual_cuda,
                        "status": "pass" if actual_cuda == EXPECTED_TORCH_CUDA else "fail",
                    }
                )
            if not cpu_only and not torch.cuda.is_available():
                checks.append(
                    {
                        "name": "cuda",
                        "expected": "available",
                        "actual": "unavailable",
                        "status": "fail",
                    }
                )
        except Exception as exc:  # import/runtime diagnostic
            runtime["torch_import"] = f"fail: {type(exc).__name__}: {exc}"
            checks.append(
                {
                    "name": "torch_import",
                    "expected": "importable",
                    "actual": runtime["torch_import"],
                    "status": "fail",
                }
            )

        try:
            from trl import (
                AutoModelForSeq2SeqLMWithValueHead,
                PPOConfig,
                PPOTrainer,
            )

            _ = (AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer)
            runtime["trl_legacy_ppo_api"] = "pass"
        except Exception as exc:
            runtime["trl_legacy_ppo_api"] = f"fail: {type(exc).__name__}: {exc}"
            checks.append(
                {
                    "name": "trl_legacy_ppo_api",
                    "expected": "importable",
                    "actual": runtime["trl_legacy_ppo_api"],
                    "status": "fail",
                }
            )
        try:
            from relik import Relik

            _ = Relik
            runtime["relik_api"] = "pass"
        except Exception as exc:
            runtime["relik_api"] = f"fail: {type(exc).__name__}: {exc}"
            checks.append(
                {
                    "name": "relik_api",
                    "expected": "importable",
                    "actual": runtime["relik_api"],
                    "status": "fail",
                }
            )
        if load_pipeline and package_version("spacy") is not None:
            try:
                import spacy

                spacy.load("en_core_web_sm")
                runtime["en_core_web_sm"] = "pass"
            except Exception as exc:
                runtime["en_core_web_sm"] = f"fail: {type(exc).__name__}: {exc}"
                checks.append(
                    {
                        "name": "en_core_web_sm",
                        "expected": "loadable",
                        "actual": runtime["en_core_web_sm"],
                        "status": "fail",
                    }
                )
    else:
        if load_pipeline and all(package_version(name) is not None for name in COREF_PACKAGES):
            try:
                import coreferee  # noqa: F401 - registers spaCy factory
                import spacy

                nlp = spacy.load("en_core_web_lg")
                nlp.add_pipe("coreferee")
                doc = nlp("John went home because he was tired.")
                runtime["coreferee_pipeline"] = "pass"
                runtime["coreferee_example"] = str(doc._.coref_chains)
            except Exception as exc:
                runtime["coreferee_pipeline"] = f"fail: {type(exc).__name__}: {exc}"
                checks.append(
                    {
                        "name": "coreferee_pipeline",
                        "expected": "loadable",
                        "actual": runtime["coreferee_pipeline"],
                        "status": "fail",
                    }
                )

    result = {
        "schema": "eraser4rag-environment-check-v1",
        "mode": mode,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_only": cpu_only,
        "checks": checks,
        "runtime": runtime,
    }
    result["status"] = (
        "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("main", "coref"), required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--skip-pipeline-load", action="store_true")
    parser.add_argument("--json-output")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = check_environment(
        args.mode,
        cpu_only=args.cpu_only,
        load_pipeline=not args.skip_pipeline_load,
    )
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
