from scripts.autodl_preflight import build_report


def test_current_workspace_is_not_misreported_as_ppo_ready(monkeypatch):
    monkeypatch.setattr(
        "scripts.autodl_preflight.check_environment",
        lambda *args, **kwargs: {"status": "fail"},
    )
    report = build_report("ppo", "configs/reproduction.yaml")
    assert report["status"] == "fail"
    failed = {item["name"] for item in report["checks"] if item["status"] == "fail"}
    assert "model_sft" in failed
    assert "relik_gate" in failed
    assert "rl_dataset_hotpotqa" in failed


def test_full_preflight_records_missing_public_snapshot_entrypoints(monkeypatch):
    monkeypatch.setattr(
        "scripts.autodl_preflight.check_environment",
        lambda *args, **kwargs: {"status": "fail"},
    )
    report = build_report("full", "configs/reproduction.yaml")
    match = next(
        item for item in report["checks"]
        if item["name"] == "retrieval_and_llama_evaluation_entrypoints"
    )
    assert match["status"] == "fail"
