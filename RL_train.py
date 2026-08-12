"""PPO training entry point for the Eraser4RAG v2 reconstruction.

The fixed reward references are each original document's local public/private
triples.  Only the current policy rewrite is passed through ReLiK.  Full PPO is
allowed only after the ReLiK consistency gate and the staged smoke tests in
``docs/AUTODL_RUNBOOK.md`` pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from utils.config_utils import (
    config_get,
    load_yaml_config,
    validate_reproduction_config,
)
from utils.reward_utils import (
    compute_retention_rates,
    compute_reward,
    get_privacy_penalty,
)
from utils.triple_utils import (
    SPECIAL_TOKENS,
    parse_serialized_triple,
    serialize_private_prompt_triple,
    serialize_public_prompt_triple,
    serialize_triple,
)

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent


SYSTEM_PROMPT = (
    "Rewrite the given text to replace only the private tail entities with the "
    "corresponding relations from the provided private triples while preserving "
    "all head entities, closely mirroring the original text, and retaining "
    "relevant public information and details not mentioned in the triples."
)


def _raw_triple(item: Any) -> tuple[str, str, str]:
    if not isinstance(item, (list, tuple)) or len(item) != 3:
        raise ValueError(f"expected a three-item raw triple, got {item!r}")
    if not all(isinstance(part, str) for part in item):
        raise ValueError(f"triple components must be strings: {item!r}")
    return item[0], item[1], item[2]


def _neutral_reference(items: Any) -> set[str]:
    if not isinstance(items, list):
        raise ValueError("local triple references must be a list")
    result: set[str] = set()
    for item in items:
        if isinstance(item, str):
            parsed = tuple(parse_serialized_triple(item))
        else:
            parsed = _raw_triple(item)
        result.add(serialize_triple(parsed))
    return result


def build_prompt(record: dict[str, Any], text: str, *, use_prefix: bool) -> str:
    public_triples = [_raw_triple(item) for item in record["public"]]
    private_triples = [_raw_triple(item) for item in record["privacy"]]
    public = "".join(serialize_public_prompt_triple(item) for item in public_triples)
    private = "".join(serialize_private_prompt_triple(item) for item in private_triples)
    if use_prefix:
        body = (
            f"\n\ntext: {text}"
            f"\n\nprivate triples: {private}"
            f"\n\npublic triples: {public}"
        )
    else:
        body = f"{text}{private}{public}"
    return f"{SYSTEM_PROMPT}{body}**Anonymized Text**:"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"record at {path}:{line_number} is not an object")
            records.append(record)
    return records


def load_flattened_examples(
    paths: Sequence[str | Path],
    *,
    use_data_percent: float,
    seed: int,
    use_prefix: bool,
    min_local_public: int,
    min_local_private: int,
    max_samples: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 < use_data_percent <= 1.0:
        raise ValueError("use_data_percent must be in (0, 1]")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")

    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    stats = {
        "dataset_records": 0,
        "sampled_records": 0,
        "contexts_seen": 0,
        "contexts_filtered": 0,
        "flattened_examples": 0,
        "files": [],
    }
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"RL dataset not found: {path}")
        records = _load_jsonl(path)
        stats["dataset_records"] += len(records)
        sample_count = int(len(records) * use_data_percent)
        if use_data_percent == 1.0:
            selected = records
        else:
            selected = rng.sample(records, sample_count)
        stats["sampled_records"] += len(selected)
        stats["files"].append(
            {"path": str(path.resolve()), "records": len(records), "selected": len(selected)}
        )

        for record_number, record in enumerate(selected):
            for field in ("public", "privacy", "ctxs"):
                if field not in record:
                    raise ValueError(f"{path} record {record_number} is missing {field!r}")
            if not isinstance(record["ctxs"], list):
                raise ValueError(f"{path} record {record_number} ctxs must be a list")
            for ctx_index, ctx in enumerate(record["ctxs"]):
                stats["contexts_seen"] += 1
                if not isinstance(ctx, dict) or not isinstance(ctx.get("text"), str):
                    raise ValueError(
                        f"{path} record {record_number} ctx {ctx_index} has invalid text"
                    )
                public_reference = _neutral_reference(ctx.get("public"))
                private_reference = _neutral_reference(ctx.get("private"))
                if (
                    len(public_reference) < min_local_public
                    or len(private_reference) < min_local_private
                ):
                    stats["contexts_filtered"] += 1
                    continue
                examples.append(
                    {
                        "prompt": build_prompt(record, ctx["text"], use_prefix=use_prefix),
                        "public": public_reference,
                        "private": private_reference,
                        "text": ctx["text"],
                        "source_path": str(path),
                        "source_record": record_number,
                        "source_context": ctx_index,
                    }
                )
                if max_samples is not None and len(examples) >= max_samples:
                    stats["flattened_examples"] = len(examples)
                    return examples, stats
    stats["flattened_examples"] = len(examples)
    return examples, stats


class Stage2RLDataset:
    """Minimal Dataset protocol implementation for the legacy TRL dataloader."""

    def __init__(self, examples: list[dict[str, Any]], tokenizer, max_length: int):
        self.examples: list[dict[str, Any]] = []
        for example in examples:
            encoding = tokenizer(
                example["prompt"],
                max_length=max_length,
                add_special_tokens=True,
                return_token_type_ids=False,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            self.examples.append(
                {
                    "input_ids": encoding["input_ids"].flatten(),
                    "attention_mask": encoding["attention_mask"].flatten(),
                    "private": example["private"],
                    "public": example["public"],
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, item: int) -> dict[str, Any]:
        return self.examples[item]


def collator(data: list[dict[str, Any]]) -> dict[str, list[Any]]:
    if not data:
        raise ValueError("cannot collate an empty batch")
    return {key: [item[key] for item in data] for key in data[0]}


def _relik_outputs(relik, texts: list[str]) -> list[Any]:
    outputs = relik(texts)
    if hasattr(outputs, "triplets"):
        if len(texts) != 1:
            raise RuntimeError(
                "ReLiK returned one output object for a multi-text batch"
            )
        return [outputs]
    outputs = list(outputs)
    if len(outputs) != len(texts):
        raise RuntimeError(
            f"ReLiK returned {len(outputs)} outputs for {len(texts)} generated texts"
        )
    return outputs


def extract_generated_triples(relik, texts: list[str]) -> list[set[str]]:
    batches: list[set[str]] = []
    for output in _relik_outputs(relik, texts):
        serialized: set[str] = set()
        seen_entity_pairs: set[tuple[str, str]] = set()
        for triple in getattr(output, "triplets", []):
            subject = getattr(getattr(triple, "subject", None), "text", None)
            relation = getattr(triple, "label", None)
            object_ = getattr(getattr(triple, "object", None), "text", None)
            if not all(isinstance(value, str) for value in (subject, relation, object_)):
                continue
            if subject == object_:
                continue
            pair = (subject, object_)
            reverse_pair = (object_, subject)
            # [RECONSTRUCTION] Use the same released extraction rule as the
            # fixed references and consistency gate: retain only the first
            # relation for an unordered entity pair.  Without this, the gate
            # would validate a different measurement function from PPO.
            if pair in seen_entity_pairs or reverse_pair in seen_entity_pairs:
                continue
            seen_entity_pairs.add(pair)
            serialized.add(serialize_triple((subject, relation, object_)))
        batches.append(serialized)
    return batches


def batch_rewards(
    predicted: list[set[str]],
    public_references: list[set[str]],
    private_references: list[set[str]],
    p: int,
) -> tuple[list[float], list[float], list[float]]:
    if not (
        len(predicted) == len(public_references) == len(private_references)
    ):
        raise ValueError("prediction/reference batch sizes differ")
    rewards: list[float] = []
    public_rates: list[float] = []
    private_rates: list[float] = []
    for current, public, private in zip(
        predicted, public_references, private_references
    ):
        r_pub, r_pri = compute_retention_rates(current, public, private)
        public_rates.append(r_pub)
        private_rates.append(r_pri)
        rewards.append(compute_reward(r_pub, r_pri, p))
    return rewards, public_rates, private_rates


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _resolve_device(value: str) -> str:
    if value != "auto":
        return value
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _resize_wrapped_model(model, tokenizer_size: int) -> None:
    base_model = getattr(model, "pretrained_model", model)
    base_model.resize_token_embeddings(tokenizer_size)


def _save_checkpoint(ppo_trainer, tokenizer, output_dir: Path, step: str) -> Path:
    checkpoint = output_dir / f"step_{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    ppo_trainer.save_pretrained(str(checkpoint))
    tokenizer.save_pretrained(str(checkpoint))
    return checkpoint


def _resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path).expanduser()
    if path.is_absolute():
        if not path.is_file():
            raise FileNotFoundError(f"reproduction config not found: {config_path}")
        return path.resolve(strict=False)
    if path.is_file():
        return path.resolve()
    repository_relative = REPO_ROOT / path
    if repository_relative.exists():
        return repository_relative.resolve()
    raise FileNotFoundError(f"reproduction config not found: {config_path}")


def _config_defaults(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = _resolve_config_path(config_path)
    config = load_yaml_config(path)
    validate_reproduction_config(config)
    return path, config


def validate_relik_gate_report(
    report_path: str | Path,
    *,
    minimum_samples: int,
    allow_warning: bool,
) -> dict[str, Any]:
    """Validate the mandatory ReLiK consistency artifact before PPO."""

    if minimum_samples <= 0:
        raise ValueError("minimum gate samples must be positive")
    path = Path(report_path).expanduser()
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f"ReLiK consistency report not found: {path}. "
            "Run scripts/validate_relik_consistency.py before PPO."
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid ReLiK consistency report: {path}") from exc
    if not isinstance(report, dict) or report.get("schema") != "eraser4rag-relik-consistency-v1":
        raise ValueError("unexpected ReLiK consistency report schema")

    status = report.get("gate_status")
    if status not in {"pass", "warning"}:
        raise ValueError(f"invalid ReLiK gate status: {status!r}")
    if status == "warning" and not allow_warning:
        raise RuntimeError(
            "ReLiK consistency gate is WARNING. Review the mismatch report and "
            "pass --allow-relik-gate-warning only for a documented deviation."
        )
    metrics = report.get("metrics")
    sample_count = metrics.get("sample_count") if isinstance(metrics, dict) else None
    if not isinstance(sample_count, int) or sample_count < minimum_samples:
        raise ValueError(
            f"ReLiK gate contains {sample_count!r} samples; "
            f"at least {minimum_samples} are required"
        )

    manifest_value = report.get("sample_indices_path")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise ValueError("ReLiK report has no sample_indices_path")
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = path.parent / manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"ReLiK sample manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid ReLiK sample manifest: {manifest_path}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "eraser4rag-relik-sample-indices-v1"
    ):
        raise ValueError("unexpected ReLiK sample manifest schema")
    indices = manifest.get("indices")
    if not isinstance(indices, list) or not all(isinstance(index, int) for index in indices):
        raise ValueError("ReLiK sample manifest has invalid indices")
    if len(indices) != sample_count or len(set(indices)) != len(indices):
        raise ValueError("ReLiK sample manifest count/uniqueness does not match report")
    if manifest.get("actual_sample_size") != sample_count:
        raise ValueError("ReLiK sample manifest size metadata does not match report")
    if manifest.get("data_path") != report.get("data_path"):
        raise ValueError("ReLiK report and sample manifest refer to different datasets")
    if manifest.get("seed") != report.get("seed"):
        raise ValueError("ReLiK report and sample manifest use different seeds")
    data_sha256 = manifest.get("data_sha256")
    if (
        not isinstance(data_sha256, str)
        or len(data_sha256) != 64
        or any(character not in "0123456789abcdef" for character in data_sha256.lower())
    ):
        raise ValueError("ReLiK sample manifest has no valid dataset SHA-256")
    gate_data_path = Path(report["data_path"])
    if not gate_data_path.is_file():
        raise FileNotFoundError(
            f"dataset used by the ReLiK gate is unavailable: {gate_data_path}"
        )
    digest = hashlib.sha256()
    with gate_data_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != data_sha256:
        raise ValueError("dataset used by the ReLiK gate no longer matches its SHA-256")

    return {
        "report": str(path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "status": status,
        "sample_count": sample_count,
        "warning_override": status == "warning" and allow_warning,
    }


def build_parser(argv: Sequence[str] | None = None) -> argparse.ArgumentParser:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", default="configs/reproduction.yaml")
    known, _ = preliminary.parse_known_args(argv)
    config_path, config = _config_defaults(known.config)

    def default(path: str, fallback: Any) -> Any:
        return config_get(config, path, fallback)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument(
        "--data-dir-1",
        "--data_dir_1",
        default=default(
            "datasets.rl.popqa", "dataset/sample_privacy/popqa_10_25_trp.jsonl"
        ),
    )
    parser.add_argument(
        "--data-dir-2",
        "--data_dir_2",
        default=default(
            "datasets.rl.triviaqa",
            "dataset/sample_privacy/triviaqa_10_25_trp.jsonl",
        ),
    )
    parser.add_argument(
        "--data-dir-3",
        "--data_dir_3",
        default=default(
            "datasets.rl.nq_open", "dataset/sample_privacy/NQ-open_10_25_trp.jsonl"
        ),
    )
    parser.add_argument(
        "--data-dir-4",
        "--data_dir_4",
        default=default(
            "datasets.rl.hotpotqa",
            "dataset/sample_privacy/hotpotqa_10_25_trp.jsonl",
        ),
    )
    parser.add_argument("--model-dir", "--model_dir")
    parser.add_argument("--tokenizer-dir", "--tokenizer_dir")
    parser.add_argument(
        "--relik-model",
        default=default("models.relik", "relik-ie/relik-relation-extraction-small"),
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default=str(Path(default("paths.checkpoint_root", "./output_checkpoint")) / "RL"),
    )
    parser.add_argument(
        "--logging-dir",
        "--logging_dir",
        default=str(Path(default("paths.log_root", "logs")) / "ppo"),
    )
    parser.add_argument("--use-data-percent", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=default("seed", 42))
    parser.add_argument(
        "--use-prefix", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--max-passage-length", type=int, default=default("sft.max_passage_length", 128)
    )
    parser.add_argument(
        "--max-concat-length", type=int, default=default("sft.max_concat_length", 1300)
    )
    parser.add_argument("--batch-size", type=int, default=default("rl.batch_size", 16))
    parser.add_argument(
        "--mini-batch-size", type=int, default=default("rl.mini_batch_size", 8)
    )
    parser.add_argument("--ppo-epochs", type=int, default=default("rl.ppo_epochs", 4))
    parser.add_argument(
        "--learning-rate", type=float, default=default("rl.learning_rate", 1.0e-5)
    )
    parser.add_argument("--gamma", type=float, default=default("rl.gamma", 0.99))
    parser.add_argument("--max-steps", type=int, default=default("rl.max_steps", 200_000))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--save-freq", type=int, default=100)
    parser.add_argument("--show-freq", type=int, default=50)
    parser.add_argument("--initial-p", type=int, default=default("rl.initial_p", 20))
    parser.add_argument("--p-increment", type=int, default=default("rl.p_increment", 5))
    parser.add_argument("--p-interval", type=int, default=default("rl.p_interval", 350))
    parser.add_argument("--p-max", type=int, default=default("rl.p_max", 40))
    parser.add_argument("--static-p", action="store_true")
    parser.add_argument("--min-local-public", type=int, default=5)
    parser.add_argument("--min-local-private", type=int, default=2)
    parser.add_argument("--relik-device", default="auto")
    parser.add_argument(
        "--use-nme", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--log-text-samples", action="store_true")
    parser.add_argument(
        "--relik-consistency-report",
        default=default(
            "relik_consistency.report_path",
            "outputs/relik_consistency/report.json",
        ),
    )
    parser.add_argument(
        "--minimum-gate-samples",
        type=int,
        default=default("relik_consistency.sample_size", 500),
    )
    parser.add_argument(
        "--allow-relik-gate-warning",
        action="store_true",
        help="Continue after explicit review of a WARNING gate report.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--generate-only", action="store_true")
    mode.add_argument("--reward-only", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.p_max != 40:
        raise ValueError("v2 reproduction requires p_max=40")
    if args.gamma != 0.99:
        raise ValueError("v2 reproduction requires PPO gamma=0.99")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.batch_size <= 0 or args.mini_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.batch_size % args.mini_batch_size:
        raise ValueError("batch_size must be divisible by mini_batch_size")
    if args.ppo_epochs <= 0:
        raise ValueError("ppo_epochs must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if args.max_passage_length <= 0 or args.max_concat_length <= 0:
        raise ValueError("maximum sequence lengths must be positive")
    if args.save_freq < 0 or args.show_freq <= 0:
        raise ValueError("save_freq must be non-negative and show_freq must be positive")
    if args.initial_p < 0 or args.initial_p > args.p_max:
        raise ValueError("initial_p must be within [0, p_max]")
    if args.p_interval <= 0 or args.p_increment < 0:
        raise ValueError("p_interval must be positive and p_increment non-negative")
    # Validate the whole schedule even when --static-p bypasses the helper.
    get_privacy_penalty(
        0,
        initial=args.initial_p,
        interval=args.p_interval,
        increment=args.p_increment,
        maximum=args.p_max,
    )

    seed_everything(args.seed)
    data_paths = [
        args.data_dir_1,
        args.data_dir_2,
        args.data_dir_3,
        args.data_dir_4,
    ]
    examples, data_stats = load_flattened_examples(
        data_paths,
        use_data_percent=args.use_data_percent,
        seed=args.seed,
        use_prefix=args.use_prefix,
        min_local_public=args.min_local_public,
        min_local_private=args.min_local_private,
        max_samples=args.max_samples,
    )
    gate_summary: dict[str, Any] | str = "not checked in dry-run mode"
    if not args.dry_run:
        gate_summary = validate_relik_gate_report(
            args.relik_consistency_report,
            minimum_samples=args.minimum_gate_samples,
            allow_warning=args.allow_relik_gate_warning,
        )

    startup = {
        "config": str(Path(args.config).resolve()),
        "data": data_stats,
        "batch_size": args.batch_size,
        "mini_batch_size": args.mini_batch_size,
        "ppo_epochs": args.ppo_epochs,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "initial_p": args.initial_p,
        "p_schedule": {
            "interval": args.p_interval,
            "increment": args.p_increment,
            "maximum": args.p_max,
        },
        "max_steps": args.max_steps,
        "model": args.model_dir,
        "tokenizer": args.tokenizer_dir or args.model_dir,
        "relik": args.relik_model,
        "relik_consistency_gate": gate_summary,
        "dry_run": args.dry_run,
    }
    LOGGER.info("PPO startup configuration:\n%s", json.dumps(startup, indent=2))
    if not examples:
        raise ValueError("no RL examples remain after filtering")
    if args.dry_run:
        LOGGER.info("Dry run complete: dataset and prompts loaded without model initialization")
        return 0
    if not args.model_dir:
        raise ValueError("--model-dir is required outside --dry-run")

    try:
        import torch
        from relik import Relik
        from transformers import AutoTokenizer
        from trl import (
            AutoModelForSeq2SeqLMWithValueHead,
            PPOConfig,
            PPOTrainer,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PPO dependencies are unavailable; use the pinned eraser-main environment"
        ) from exc

    set_seed(args.seed)
    tokenizer_source = args.tokenizer_dir or args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
    policy = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(args.model_dir)
    reference_policy = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(args.model_dir)
    _resize_wrapped_model(policy, len(tokenizer))
    _resize_wrapped_model(reference_policy, len(tokenizer))

    dataset = Stage2RLDataset(examples, tokenizer, args.max_concat_length)
    if len(dataset) < args.batch_size:
        raise ValueError(
            f"TRL drops incomplete batches: {len(dataset)} examples are fewer than "
            f"batch_size={args.batch_size}. Lower --batch-size or raise --max-samples."
        )

    ppo_config = PPOConfig(
        seed=args.seed,
        log_with="tensorboard",
        project_kwargs={"logging_dir": args.logging_dir},
        init_kl_coef=0.01,
        target=6,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        kl_penalty="kl",
        steps=args.max_steps * args.batch_size,
        vf_coef=5,
        gradient_accumulation_steps=1,
        is_encoder_decoder=True,
    )
    optimizer = torch.optim.Adam(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.99
    )
    trainer = PPOTrainer(
        ppo_config,
        policy,
        reference_policy,
        tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )
    if len(trainer.dataloader) == 0:
        raise ValueError("TRL dataloader is empty after drop_last=True")

    relik = None
    if not args.generate_only:
        relik_device = _resolve_device(args.relik_device)
        relik = Relik.from_pretrained(
            args.relik_model, device=relik_device, use_nme=args.use_nme
        )

    generation_kwargs = {
        "max_length": args.max_passage_length,
        "min_length": args.max_passage_length // 2,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir / "tokenizer")

    step = 1
    while step <= args.max_steps:
        for batch in trainer.dataloader:
            if step > args.max_steps:
                break
            prompts = batch["input_ids"]
            responses = trainer.generate(
                prompts,
                return_prompt=False,
                generate_ref_response=False,
                **generation_kwargs,
            )
            decoded = tokenizer.batch_decode(responses, skip_special_tokens=True)
            if args.log_text_samples and step % args.show_freq == 0:
                LOGGER.info("Generated rewrite sample at step %d: %s", step, decoded[0])
            if args.generate_only:
                LOGGER.info("Generate-only smoke completed for %d examples", len(decoded))
                return 0

            assert relik is not None
            predicted = extract_generated_triples(relik, decoded)
            p = (
                args.initial_p
                if args.static_p
                else get_privacy_penalty(
                    step,
                    initial=args.initial_p,
                    interval=args.p_interval,
                    increment=args.p_increment,
                    maximum=args.p_max,
                )
            )
            rewards, public_rates, private_rates = batch_rewards(
                predicted, batch["public"], batch["private"], p
            )
            reward_tensors = [torch.tensor(value, dtype=torch.float32) for value in rewards]
            LOGGER.info(
                "step=%d p=%d mean_reward=%.6f mean_r_pub=%.6f mean_r_pri=%.6f lr=%.8g",
                step,
                p,
                statistics.fmean(rewards),
                statistics.fmean(public_rates),
                statistics.fmean(private_rates),
                optimizer.param_groups[0]["lr"],
            )
            if args.reward_only:
                LOGGER.info("Reward-only smoke completed")
                return 0

            started = time.perf_counter()
            stats = trainer.step(prompts, responses, reward_tensors)
            elapsed = time.perf_counter() - started
            trainer.log_stats(stats, batch, reward_tensors)
            LOGGER.info(
                "step=%d update_seconds=%.3f kl=%s",
                step,
                elapsed,
                stats.get("objective/kl", stats.get("ppo/mean_kl", "unavailable")),
            )
            if args.save_freq and step % args.save_freq == 0:
                checkpoint = _save_checkpoint(trainer, tokenizer, output_dir, str(step))
                LOGGER.info("Saved checkpoint: %s", checkpoint)
            step += 1
    checkpoint = _save_checkpoint(trainer, tokenizer, output_dir, "final")
    LOGGER.info("Saved final checkpoint: %s", checkpoint)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
