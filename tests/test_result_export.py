import gzip
import hashlib
import json

from scripts.export_cloud_results import parse_ppo_log, sanitize
from scripts.verify_result_artifacts import verify_artifacts


def test_parse_ppo_log_preserves_complete_step_pairs(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        """2026-08-15 10:09:00,168 - __main__ - INFO - PPO startup configuration:
{"max_steps": 2, "model": "/root/autodl-tmp/Eraser4RAG/output_checkpoint/SFT"}
[2026-08-15 10:16:54,749] [INFO] step=1 p=20 mean_reward=0.1 mean_r_pub=0.2 mean_r_pri=0.3 lr=1e-5
[2026-08-15 10:17:02,907] [INFO] step=1 update_seconds=8.1 kl=0.0
[2026-08-15 10:17:08,863] [INFO] step=2 p=20 mean_reward=0.2 mean_r_pub=0.3 mean_r_pri=0.4 lr=9e-6
[2026-08-15 10:17:16,958] [INFO] step=2 update_seconds=8.2 kl=1.5
[2026-08-15 10:17:18,000] [INFO] Saved final checkpoint: /root/autodl-tmp/Eraser4RAG/output_checkpoint/RL/step_final
""",
        encoding="utf-8",
    )

    rows, summary = parse_ppo_log(log_path)

    assert [row["step"] for row in rows] == [1, 2]
    assert rows[1]["kl"] == 1.5
    assert summary["steps"] == 2
    assert summary["p_phases"][0]["first_step"] == 1
    assert summary["p_phases"][0]["last_step"] == 2


def test_sanitize_rewrites_only_cloud_project_prefix():
    value = {
        "path": "/root/autodl-tmp/Eraser4RAG/outputs/report.json",
        "external": "google/flan-t5-large",
    }

    assert sanitize(value, "/root/autodl-tmp/Eraser4RAG") == {
        "path": "outputs/report.json",
        "external": "google/flan-t5-large",
    }


def test_verify_relik_artifacts_checks_uncompressed_content(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    content = b'{"id": 1}\n{"id": 2}'
    artifact = artifact_dir / "sample_with_triplets.jsonl.gz"
    with gzip.open(artifact, "wb") as target:
        target.write(content)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "eraser4rag-relik-files-v1",
                "inputs": [
                    {
                        "output": "dataset/sample_with_triplets.jsonl",
                        "output_sha256": hashlib.sha256(content).hexdigest(),
                        "output_records": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = verify_artifacts(artifact_dir, manifest_path)

    assert report["valid"] is True
    assert report["artifacts"][0]["records"] == 2


def test_verify_generic_artifact_manifest_checks_compressed_size(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    content = b'{"id": 1}\n'
    artifact = artifact_dir / "output.jsonl.gz"
    with gzip.open(artifact, "wb") as target:
        target.write(content)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "eraser4rag-compressed-jsonl-artifacts-v1",
                "artifacts": [
                    {
                        "file": artifact.name,
                        "records": 1,
                        "compressed_bytes": artifact.stat().st_size,
                        "uncompressed_sha256": hashlib.sha256(content).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert verify_artifacts(artifact_dir, manifest_path)["valid"] is True
