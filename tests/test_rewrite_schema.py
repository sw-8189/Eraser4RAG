import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_pure_function(filename, function_name):
    path = REPO_ROOT / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == function_name
    )
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function_name]


def _batch():
    return {
        "question": ["question"],
        "answers": [["answer"]],
        "privacy": [["private-ref"]],
        "public": [["public-ref"]],
        "ctxs": [[{"id": "doc-1", "public": ["p"], "private": ["r"]}]],
    }


def test_special_rewrite_output_keeps_document_local_references():
    build = _load_pure_function("rewrite_docs_special.py", "build_output_record")
    output = build(_batch(), 0, [["rewritten document"]])

    assert output["ctxs"][0]["public"] == ["p"]
    assert output["ctxs"][0]["private"] == ["r"]
    assert output["anonymized_ctxs"] == ["rewritten document"]


def test_inference_attack_rewrite_uses_global_privacy_schema():
    build = _load_pure_function(
        "rewrite_docs_inferattack_inference.py", "build_output_record"
    )
    output = build(_batch(), 0, [["rewritten document"]])

    assert output["privacy"] == ["private-ref"]
    assert "private" not in output
