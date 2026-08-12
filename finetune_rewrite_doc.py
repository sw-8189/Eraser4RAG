import argparse
import logging
import os
import random
from pathlib import Path

from utils.config_utils import config_get, load_yaml_config, validate_reproduction_config

try:
    from utils.triple_utils import SPECIAL_TOKENS
except (ImportError, ModuleNotFoundError):
    # Compatibility fallback while the shared v2 utility is being introduced.
    SPECIAL_TOKENS = (
        "<csubj>",
        "<crel>",
        "<cobj>",
        "<ce>",
        "<rsubj>",
        "<rrel>",
        "<robj>",
        "<re>",
    )


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent


def _config_defaults(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    config = load_yaml_config(path)
    validate_reproduction_config(config)
    return path.resolve(), config


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def get_args(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", default="configs/reproduction.yaml")
    known, _ = preliminary.parse_known_args(argv)
    config_path, config = _config_defaults(known.config)

    def default(path, fallback):
        return config_get(config, path, fallback)

    parser = argparse.ArgumentParser(description="Supervised fine-tuning for the Eraser4RAG rewriter.")
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument(
        "--data_dir",
        "--data-dir",
        dest="data_dir",
        type=str,
        default=default("datasets.sft", "./dataset/constructed_dataset/popqa_10_25_filtered_new.jsonl"),
    )
    parser.add_argument(
        "--model_dir",
        "--model-dir",
        dest="model_dir",
        type=str,
        default=default("model_paths.sft", "./models/sft"),
    )
    parser.add_argument(
        "--tokenizer_dir",
        "--tokenizer-dir",
        dest="tokenizer_dir",
        type=str,
        default=None,
        help="Tokenizer path/model ID. Defaults to --model_dir.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=str,
        default=default("checkpoints.sft", "./output_checkpoint/SFT"),
    )
    parser.add_argument("--use_data_percent", "--use-data-percent", dest="use_data_percent", type=float, default=1.0)
    parser.add_argument("--max_samples", "--max-samples", dest="max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=default("seed", 42))
    parser.add_argument("--epochs", type=int, default=default("sft.epochs", 3))
    parser.add_argument(
        "--learning-rate", type=float, default=default("sft.learning_rate", 5e-5)
    )
    parser.add_argument(
        "--use_prefix",
        "--use-prefix",
        dest="use_prefix",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--no-use-prefix", dest="use_prefix", action="store_false")
    parser.add_argument("--smoke_test", "--smoke-test", dest="smoke_test", action="store_true")
    parser.add_argument(
        "--resume_from_checkpoint",
        "--resume-from-checkpoint",
        dest="resume_from_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument("--max_passage_length", "--max-passage-length", dest="max_passage_length", type=int, default=default("sft.max_passage_length", 128))
    parser.add_argument("--max_public_length", "--max-public-length", dest="max_public_length", type=int, default=742)
    parser.add_argument("--max_private_length", "--max-private-length", dest="max_private_length", type=int, default=252)
    parser.add_argument("--max_concat_length", "--max-concat-length", dest="max_concat_length", type=int, default=default("sft.max_concat_length", 1300))
    args = parser.parse_args(argv)

    if not 0 < args.use_data_percent <= 1:
        parser.error("--use-data-percent must be in (0, 1]")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.max_concat_length <= 0 or args.max_passage_length <= 0:
        parser.error("maximum sequence lengths must be positive")
    if args.epochs <= 0 or args.learning_rate <= 0:
        parser.error("--epochs and --learning-rate must be positive")

    if args.smoke_test:
        args.max_samples = min(args.max_samples or 100, 100)

    args.tokenizer_dir = args.tokenizer_dir or args.model_dir
    return args


def seed_everything(seed):
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, seed, train_fraction=0.85):
    import torch
    from torch.utils.data import random_split

    if len(dataset) < 2:
        raise ValueError("SFT requires at least two examples so train and evaluation splits are both non-empty")
    train_size = int(len(dataset) * train_fraction)
    train_size = min(max(train_size, 1), len(dataset) - 1)
    eval_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, eval_size], generator=generator)


def add_rewriter_special_tokens(tokenizer, model):
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))


def build_training_args(args):
    from transformers import Seq2SeqTrainingArguments

    interval = 1 if args.smoke_test else 1000
    return Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        fp16=False,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=1 if args.smoke_test else -1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_first_step=True,
        logging_strategy="steps",
        logging_steps=1 if args.smoke_test else 50,
        evaluation_strategy="steps",
        eval_steps=interval,
        save_strategy="steps",
        save_steps=interval,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="tensorboard",
        predict_with_generate=True,
        generation_max_length=args.max_passage_length,
        seed=args.seed,
        data_seed=args.seed,
    )


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = get_args(argv)
    seed_everything(args.seed)

    from data_structure import Anonymize_Doc_popqa
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
    )

    data_path = Path(args.data_dir)
    if not data_path.is_file():
        raise FileNotFoundError(f"SFT dataset not found: {data_path}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading tokenizer=%s and model=%s", args.tokenizer_dir, args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    add_rewriter_special_tokens(tokenizer, model)

    LOGGER.info("Loading SFT dataset=%s", data_path)
    dataset = Anonymize_Doc_popqa(args, tokenizer, str(data_path))
    train_dataset, eval_dataset = split_dataset(dataset, args.seed)

    LOGGER.info(
        "SFT configuration: dataset=%d train=%d eval=%d seed=%d model=%s tokenizer=%s "
        "input_max=%d target_max=%d lr=%g epochs=%d max_samples=%s smoke_test=%s",
        len(dataset),
        len(train_dataset),
        len(eval_dataset),
        args.seed,
        args.model_dir,
        args.tokenizer_dir,
        args.max_concat_length,
        args.max_passage_length,
        args.learning_rate,
        args.epochs,
        args.max_samples,
        args.smoke_test,
    )

    training_args = build_training_args(args)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="pt")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Keep the final checkpoint self-contained even if no periodic save was reached.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    LOGGER.info("Saved self-contained SFT model and tokenizer to %s", args.output_dir)


if __name__ == "__main__":
    main()
