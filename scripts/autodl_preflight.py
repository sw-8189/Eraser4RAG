"""Report whether an AutoDL workspace is ready for each reproduction stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.restore_sft_dataset import DATASET_SHA256, sha256
from scripts.verify_models import validate_local_manifest
from scripts.check_environment import check_environment
from utils.config_utils import config_get, load_yaml_config


STAGES = ("sft", "ppo", "full")


def _check_file(checks: list[dict[str, Any]], name: str, path: Path) -> None:
    checks.append({"name": name, "path": str(path), "status": "pass" if path.is_file() else "fail"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/reproduction.yaml"))
    parser.add_argument("--json-output")
    return parser


def build_report(stage: str, config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = REPO_ROOT / config_file
    config = load_yaml_config(config_file)
    checks: list[dict[str, Any]] = []
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()
    checks.append({"name": "git_branch", "expected": "reproduce-v2", "actual": branch,
                   "status": "pass" if branch == "reproduce-v2" else "fail"})

    environment = check_environment("main", cpu_only=False, load_pipeline=True)
    checks.append({
        "name": "main_environment",
        "status": environment["status"],
        "detail": environment,
    })

    sft_data = REPO_ROOT / str(config_get(config, "datasets.sft")).removeprefix("./")
    _check_file(checks, "sft_dataset", sft_data)
    if sft_data.is_file():
        actual = sha256(sft_data)
        checks.append({"name": "sft_dataset_sha256", "expected": DATASET_SHA256,
                       "actual": actual, "status": "pass" if actual == DATASET_SHA256 else "fail"})

    required_models = ["sft", "relik"]
    if stage == "full":
        required_models.extend(["retrieval", "eval"])
    for model_stage in required_models:
        key = {"sft": "flan_t5_large", "relik": "relik", "retrieval": "contriever", "eval": "llama3"}[model_stage]
        model_path = REPO_ROOT / str(config_get(config, f"model_paths.{model_stage}")).removeprefix("./")
        try:
            if not model_path.is_dir():
                raise FileNotFoundError(f"local model directory not found: {model_path}")
            validate_local_manifest(model_path, model_stage, config_get(config, f"model_revisions.{key}"))
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            checks.append({"name": f"model_{model_stage}", "path": str(model_path), "status": "fail", "detail": str(exc)})
        else:
            checks.append({"name": f"model_{model_stage}", "path": str(model_path), "status": "pass"})

    if stage in {"ppo", "full"}:
        _check_file(checks, "sft_checkpoint", REPO_ROOT / "output_checkpoint/SFT/config.json")
        _check_file(checks, "relik_gate", REPO_ROOT / str(config_get(config, "relik_consistency.report_path")).removeprefix("./"))
        for name, value in config_get(config, "datasets.rl").items():
            _check_file(checks, f"rl_dataset_{name}", REPO_ROOT / str(value).removeprefix("./"))

    if stage == "full":
        checks.append({
            "name": "retrieval_and_llama_evaluation_entrypoints",
            "status": "fail",
            "detail": "public snapshot has no pinned Wikipedia index builder or complete Llama-3 QA evaluator",
        })
    status = "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    return {"schema": "eraser4rag-autodl-preflight-v1", "stage": stage, "status": status, "checks": checks}


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(args.stage, args.config)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
