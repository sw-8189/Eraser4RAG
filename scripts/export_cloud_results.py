"""Export a sanitized, reviewable result bundle from an AutoDL artifact copy."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RUN_NAME = "two_dataset_b8_3000"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
TIMESTAMP = r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
REWARD_RE = re.compile(
    rf"{TIMESTAMP}.*?step=(?P<step>\d+) p=(?P<p>\d+) "
    rf"mean_reward=(?P<mean_reward>{NUMBER}) "
    rf"mean_r_pub=(?P<mean_r_pub>{NUMBER}) "
    rf"mean_r_pri=(?P<mean_r_pri>{NUMBER}) lr=(?P<learning_rate>{NUMBER})"
)
UPDATE_RE = re.compile(
    rf"{TIMESTAMP}.*?step=(?P<step>\d+) "
    rf"update_seconds=(?P<update_seconds>{NUMBER}) kl=(?P<kl>{NUMBER})"
)
FINAL_RE = re.compile(rf"{TIMESTAMP}.*?Saved final checkpoint: (?P<path>\S+)")

METRIC_NAMES = (
    "popqa_eval",
    "hotpotqa_eval",
    "popqa_special",
    "hotpotqa_special",
    "popqa_inferattack",
    "hotpotqa_inferattack",
)
MANIFEST_NAMES = (
    "two_dataset_retrieval_manifest.json",
    "coreferee_two_dataset_v1_timeout120.json",
    "relik_two_dataset_v1_timeout120.json",
    "rl_postprocess_two_dataset_v1_timeout120.json",
)


def _sanitize_string(value: str, project_prefix: str) -> str:
    normalized = project_prefix.rstrip("/")
    if value == normalized:
        return "."
    return value.replace(f"{normalized}/", "")


def sanitize(value: Any, project_prefix: str) -> Any:
    if isinstance(value, dict):
        return {key: sanitize(item, project_prefix) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, project_prefix) for item in value]
    if isinstance(value, str):
        return _sanitize_string(value, project_prefix)
    return value


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _timestamp(value: str) -> str:
    return value.replace(" ", "T").replace(",", ".") + "+08:00"


def parse_startup_configuration(log_text: str) -> dict[str, Any]:
    marker = "PPO startup configuration:"
    marker_index = log_text.find(marker)
    if marker_index < 0:
        raise ValueError("PPO startup configuration was not found in the log")
    object_index = log_text.find("{", marker_index + len(marker))
    if object_index < 0:
        raise ValueError("PPO startup configuration JSON was not found")
    value, _ = json.JSONDecoder().raw_decode(log_text[object_index:])
    if not isinstance(value, dict):
        raise ValueError("PPO startup configuration must be a JSON object")
    return value


def parse_ppo_log(log_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    reward_rows: dict[int, dict[str, Any]] = {}
    update_rows: dict[int, dict[str, Any]] = {}
    for line in text.splitlines():
        reward = REWARD_RE.search(line)
        if reward:
            fields = reward.groupdict()
            step = int(fields["step"])
            if step in reward_rows:
                raise ValueError(f"duplicate PPO reward row for step {step}")
            reward_rows[step] = {
                "step": step,
                "p": int(fields["p"]),
                "reward_timestamp": _timestamp(fields["timestamp"]),
                "mean_reward": float(fields["mean_reward"]),
                "mean_r_pub": float(fields["mean_r_pub"]),
                "mean_r_pri": float(fields["mean_r_pri"]),
                "learning_rate": float(fields["learning_rate"]),
            }
        update = UPDATE_RE.search(line)
        if update:
            fields = update.groupdict()
            step = int(fields["step"])
            if step in update_rows:
                raise ValueError(f"duplicate PPO update row for step {step}")
            update_rows[step] = {
                "update_timestamp": _timestamp(fields["timestamp"]),
                "update_seconds": float(fields["update_seconds"]),
                "kl": float(fields["kl"]),
            }

    if not reward_rows or set(reward_rows) != set(update_rows):
        raise ValueError(
            "PPO log must contain matching reward and update rows for every step"
        )
    expected_steps = set(range(1, max(reward_rows) + 1))
    if set(reward_rows) != expected_steps:
        raise ValueError("PPO log steps are not contiguous from 1")

    rows = []
    for step in sorted(reward_rows):
        rows.append({**reward_rows[step], **update_rows[step]})

    phases: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        phases[row["p"]].append(row)

    final_match = None
    for match in FINAL_RE.finditer(text):
        final_match = match
    if final_match is None:
        raise ValueError("final checkpoint marker was not found in the PPO log")

    def aggregate(items: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "mean_reward": statistics.fmean(item["mean_reward"] for item in items),
            "mean_r_pub": statistics.fmean(item["mean_r_pub"] for item in items),
            "mean_r_pri": statistics.fmean(item["mean_r_pri"] for item in items),
            "mean_kl": statistics.fmean(item["kl"] for item in items),
            "mean_update_seconds": statistics.fmean(
                item["update_seconds"] for item in items
            ),
        }

    summary = {
        "schema": "eraser4rag-ppo-run-summary-v1",
        "run_name": RUN_NAME,
        "timezone": "Asia/Shanghai",
        "first_step_at": rows[0]["reward_timestamp"],
        "last_update_at": rows[-1]["update_timestamp"],
        "final_checkpoint_at": _timestamp(final_match.group("timestamp")),
        "final_checkpoint": final_match.group("path"),
        "steps": len(rows),
        "overall": aggregate(rows),
        "p_phases": [
            {
                "p": p,
                "first_step": items[0]["step"],
                "last_step": items[-1]["step"],
                "steps": len(items),
                **aggregate(items),
            }
            for p, items in sorted(phases.items())
        ],
        "startup_configuration": parse_startup_configuration(text),
    }
    return rows, summary


def write_curve(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("PPO curve is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_results(source_root: Path, output_root: Path, project_prefix: str) -> None:
    metric_source = (
        source_root / "outputs" / "evaluation" / RUN_NAME / "metrics"
    )
    for name in METRIC_NAMES:
        source = metric_source / f"{name}.json"
        write_json(
            output_root / "metrics" / source.name,
            sanitize(load_json(source), project_prefix),
        )

    manifest_source = source_root / "outputs" / "manifests"
    for name in MANIFEST_NAMES:
        source = manifest_source / name
        write_json(
            output_root / "manifests" / name,
            sanitize(load_json(source), project_prefix),
        )

    consistency_source = source_root / "outputs" / "relik_consistency"
    for name in ("report.json", "sample_indices.json"):
        source = consistency_source / name
        write_json(
            output_root / "validation" / f"relik_consistency_{name}",
            sanitize(load_json(source), project_prefix),
        )

    sft_source = source_root / "output_checkpoint" / "SFT" / "training_manifest.json"
    write_json(
        output_root / "training" / "sft_training_manifest.json",
        sanitize(load_json(sft_source), project_prefix),
    )

    log_path = source_root / "logs" / "ppo" / RUN_NAME / "run.log"
    rows, summary = parse_ppo_log(log_path)
    write_curve(output_root / "training" / "ppo_training_curve.csv", rows)
    write_json(
        output_root / "training" / "ppo_run_summary.json",
        sanitize(summary, project_prefix),
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--project-prefix", default="/root/autodl-tmp/Eraser4RAG"
    )
    args = parser.parse_args(argv)
    export_results(args.source_root, args.output_root, args.project_prefix)
    print(f"Exported sanitized results to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
