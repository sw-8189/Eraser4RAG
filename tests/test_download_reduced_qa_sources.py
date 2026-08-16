import json

from scripts.download_reduced_qa_sources import (
    HOTPOTQA_REVISION,
    POPQA_REVISION,
    main,
    source_plan,
)
from scripts.prepare_reduced_retrieval_data import (
    HOTPOTQA_REVISION as CONVERTER_HOTPOTQA_REVISION,
    POPQA_REVISION as CONVERTER_POPQA_REVISION,
)


def test_source_plan_pins_revisions_and_expected_files(tmp_path):
    plan = source_plan(tmp_path)
    assert [item["name"] for item in plan] == [
        "popqa_train",
        "hotpotqa_train",
        "hotpotqa_validation",
    ]
    assert plan[0]["revision"] == POPQA_REVISION
    assert plan[1]["revision"] == HOTPOTQA_REVISION
    assert POPQA_REVISION == CONVERTER_POPQA_REVISION
    assert HOTPOTQA_REVISION == CONVERTER_HOTPOTQA_REVISION
    assert plan[2]["split"] == "validation"


def test_source_download_dry_run_does_not_require_network(tmp_path, capsys):
    assert main(["--output-dir", str(tmp_path), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "eraser4rag-source-plan-v1"
    assert len(payload["sources"]) == 3
    assert list(tmp_path.iterdir()) == []
