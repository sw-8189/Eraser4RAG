"""Load and minimally exercise a selected Eraser4RAG model stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download_models import MODEL_STAGES, REPO_ROOT, _config_revision


def validate_local_manifest(
    model: str | Path, stage: str, expected_revision: str | None
) -> dict | None:
    path = Path(model)
    if not path.exists():
        return None
    if not path.is_dir():
        raise ValueError(f"local model path is not a directory: {path}")
    manifest_path = path / "download_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"local model manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != stage:
        raise ValueError(f"model manifest stage mismatch: expected {stage}")
    if manifest.get("repo_id") != MODEL_STAGES[stage]:
        raise ValueError(f"model manifest repo mismatch for stage {stage}")
    if expected_revision and manifest.get("resolved_revision") != expected_revision:
        raise ValueError(
            f"model manifest revision mismatch: expected {expected_revision}"
        )
    return manifest


def _device(value: str) -> str:
    if value != "auto":
        return value
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def verify_flan(
    model: str, *, load_model: bool, device: str, revision: str | None, text: str
) -> dict:
    import torch
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    config = AutoConfig.from_pretrained(model, revision=revision)
    result = {
        "tokenizer_size": len(tokenizer),
        "model_type": config.model_type,
        "model_loaded": False,
    }
    if load_model:
        seq2seq = AutoModelForSeq2SeqLM.from_pretrained(
            model, revision=revision
        ).to(device)
        seq2seq.eval()
        encoded = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = seq2seq.generate(**encoded, max_new_tokens=8)
        result["model_loaded"] = True
        result["generation"] = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return result


def verify_relik(model: str, *, device: str, use_nme: bool, text: str) -> dict:
    from relik import Relik

    extractor = Relik.from_pretrained(model, device=device, use_nme=use_nme)
    output = extractor(text)
    if isinstance(output, (list, tuple)):
        output = output[0]
    triples = []
    for triple in getattr(output, "triplets", []):
        triples.append(
            [triple.subject.text, triple.label, triple.object.text]
        )
    return {"pipeline_loaded": True, "triples": triples}


def verify_retrieval(
    model: str, *, device: str, revision: str | None, text: str
) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    encoder = AutoModel.from_pretrained(model, revision=revision).to(device)
    encoded = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        output = encoder(**encoded)
    embedding = output.last_hidden_state[:, 0]
    return {"model_loaded": True, "embedding_shape": list(embedding.shape)}


def verify_llama(
    model: str,
    *,
    load_model: bool,
    device: str,
    revision: str | None,
    text: str,
) -> dict:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    config = AutoConfig.from_pretrained(model, revision=revision)
    result = {
        "tokenizer_size": len(tokenizer),
        "model_type": config.model_type,
        "model_loaded": False,
    }
    if load_model:
        causal = AutoModelForCausalLM.from_pretrained(
            model, revision=revision
        ).to(device)
        causal.eval()
        encoded = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = causal.generate(**encoded, max_new_tokens=8)
        result["model_loaded"] = True
        result["generation"] = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(MODEL_STAGES), required=True)
    parser.add_argument("--model", help="Hub ID or local model path")
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "reproduction.yaml")
    )
    parser.add_argument("--revision", help="Expected Hub revision for non-local models")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument(
        "--use-nme", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--text", default="Marie Curie collaborated with Pierre Curie in Paris."
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = args.model or str(REPO_ROOT / "models" / args.stage)
    revision = args.revision or _config_revision(args.config, args.stage)
    manifest = validate_local_manifest(model, args.stage, revision)
    device = _device(args.device)
    if args.stage in {"smoke", "sft"}:
        result = verify_flan(
            model,
            load_model=args.load_model,
            device=device,
            revision=revision,
            text=args.text,
        )
    elif args.stage == "relik":
        result = verify_relik(
            model, device=device, use_nme=args.use_nme, text=args.text
        )
    elif args.stage == "retrieval":
        result = verify_retrieval(
            model, device=device, revision=revision, text=args.text
        )
    else:
        result = verify_llama(
            model,
            load_model=args.load_model,
            device=device,
            revision=revision,
            text=args.text,
        )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "model": model,
                "requested_revision": revision,
                "device": device,
                "download_manifest": manifest,
                "result": result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
