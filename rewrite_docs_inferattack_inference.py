import argparse
import json
import logging
import random
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ModuleNotFoundError:
    torch = None
    DataLoader = None
    Dataset = object
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

try:
    from utils.triple_utils import (
        SPECIAL_TOKENS,
        parse_serialized_triple,
        serialize_private_prompt_triple,
        serialize_public_prompt_triple,
    )
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

    def parse_serialized_triple(text):
        if not isinstance(text, str):
            return text
        try:
            head, remainder = text.removeprefix("<subj>").split("<rel>", 1)
            relation, tail = remainder.split("<obj>", 1)
            return [head, relation, tail.removesuffix("<e>")]
        except ValueError as exc:
            raise ValueError(f"malformed serialized triple: {text!r}") from exc

    def serialize_public_prompt_triple(triple):
        return "<csubj>{}<crel>{}<cobj>{}<ce>".format(triple[0], triple[1], triple[2])

    def serialize_private_prompt_triple(triple):
        return "<rsubj>{}<rrel>{}<robj>{}<re>".format(triple[0], triple[1], triple[2])


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def normalize_triple(triple):
    normalized = parse_serialized_triple(triple) if isinstance(triple, str) else triple
    if not isinstance(normalized, (list, tuple)) or len(normalized) != 3:
        raise ValueError(f"expected a three-field triple, got {triple!r}")
    return normalized


def serialize_prompt_triples(triples, private):
    serializer = serialize_private_prompt_triple if private else serialize_public_prompt_triple
    return "".join(serializer(normalize_triple(triple)) for triple in triples)


def serialize_reference_triple(triple):
    head, relation, tail = normalize_triple(triple)
    return f"<subj>{head}<rel>{relation}<obj>{tail}<e>"


class PromptDataset(Dataset):
    def __init__(self, args, tokenizer, data):
        self.examples = []
        for record in data:
            if "qa" in record and not record["qa"]:
                continue
            prompt_groups, public, privacy, contexts = format_prompt(record, args)
            if not prompt_groups:
                LOGGER.warning("Skipping a record without contexts")
                continue
            prompt_ids = tokenizer(
                prompt_groups,
                return_tensors="pt",
                padding=True,
                max_length=args.max_concat_length,
                truncation=True,
            )
            if "qa" in record:
                question = record["qa"][0]["question"]
                answers = record["qa"][0]["answer"]
            else:
                question = record["question"]
                answers = record["answers"]
            self.examples.append(
                {
                    "question": question,
                    "answers": answers,
                    "public": public,
                    "privacy": privacy,
                    "prompts": prompt_ids,
                    "ctxs": contexts,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn():
        def collate_fn(batch):
            keys = ("question", "answers", "public", "privacy", "ctxs", "prompts")
            return {key: [example[key] for example in batch] for key in keys}

        return collate_fn


def format_prompt(record, args):
    prompts = []
    contexts = []
    system_prompt = (
        "Rewrite the given text to remove or generalize only the private relations and their tail entities "
        "from the provided private triples while preserving all head entities, ensuring the text retains "
        "relevant public information and any details not mentioned in the triples."
    )

    global_public = serialize_prompt_triples(record["public"], private=False)
    global_private = serialize_prompt_triples(record["privacy"], private=True)
    output_public = [serialize_reference_triple(triple) for triple in record["public"]]
    output_private = [serialize_reference_triple(triple) for triple in record["privacy"]]

    for context in record.get("ctxs", []):
        if args.triple_scope == "local":
            public_triples = serialize_prompt_triples(context.get("public", []), private=False)
            private_triples = serialize_prompt_triples(context.get("private", []), private=True)
        else:
            public_triples = global_public
            private_triples = global_private

        if args.use_prefix:
            prompt = (
                f"{system_prompt}\n\ntext: {context['text']}"
                f"\n\nprivacy triples: {private_triples}"
                f"\n\npublic triples: {public_triples}"
                "**Anonymized Text**:"
            )
        else:
            prompt = f"{system_prompt}{context['text']}{private_triples}{public_triples}**Anonymized Text**:"
        prompts.append(prompt)
        contexts.append(context["text"])
    return prompts, output_public, output_private, contexts


def load_data(data_path):
    path = Path(data_path)
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as fin:
            data = json.load(fin)
        return data if isinstance(data, list) else [data]
    if path.suffix == ".jsonl":
        data = []
        with path.open("r", encoding="utf-8") as fin:
            for line_number, line in enumerate(fin, start=1):
                if not line.strip():
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
        return data
    raise ValueError(f"input must use .json or .jsonl: {path}")


def select_records(data, max_samples, seed):
    if max_samples is None or max_samples >= len(data):
        return list(data)
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    indices = sorted(random.Random(seed).sample(range(len(data)), max_samples))
    return [data[index] for index in indices]


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Rewrite the inference-attack evaluation set.")
    parser.add_argument(
        "--data_dir",
        "--data-dir",
        dest="data_dir",
        default="./dataset/inference_attack/popqa_10_25_inferattack.jsonl",
    )
    parser.add_argument("--model_dir", "--model-dir", dest="model_dir", default="./output_checkpoint/RL/step_final")
    parser.add_argument("--tokenizer_dir", "--tokenizer-dir", dest="tokenizer_dir", default=None)
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        "--output",
        dest="output_dir",
        default="./outputs/rewritten/popqa_10_25_inferattack.jsonl",
    )
    parser.add_argument("--max_passage_length", "--max-passage-length", dest="max_passage_length", type=int, default=128)
    parser.add_argument("--max_concat_length", "--max-concat-length", dest="max_concat_length", type=int, default=1300)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=1)
    parser.add_argument("--max_samples", "--max-samples", dest="max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--triple_scope", "--triple-scope", dest="triple_scope", choices=("global", "local"), default="global")
    parser.add_argument(
        "--pri_each",
        "--pri-each",
        dest="pri_each",
        type=str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Backward-compatible alias: true selects --triple-scope local.",
    )
    parser.add_argument("--no-pri-each", dest="pri_each", action="store_false", default=None)
    args = parser.parse_args(argv)
    if torch is None:
        parser.error("Torch/Transformers are required outside --help; use eraser-main")
    if args.pri_each is not None:
        args.triple_scope = "local" if args.pri_each else "global"
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_samples is not None and args.max_samples < 0:
        parser.error("--max-samples must be non-negative")
    if args.max_passage_length <= 0 or args.max_concat_length <= 0:
        parser.error("maximum sequence lengths must be positive")
    args.tokenizer_dir = args.tokenizer_dir or args.model_dir
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return args


def load_rewriter(args):
    if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        raise RuntimeError("Torch/Transformers are unavailable; use eraser-main")
    LOGGER.info("Loading rewrite model=%s tokenizer=%s", args.model_dir, args.tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    embedding_count = model.get_input_embeddings().num_embeddings
    if embedding_count != len(tokenizer):
        raise ValueError(
            "tokenizer/model vocabulary mismatch: "
            f"tokenizer={len(tokenizer)} model_embeddings={embedding_count}; "
            "use the tokenizer saved with the SFT/RL checkpoint"
        )
    model.to(args.device)
    model.eval()
    return tokenizer, model


def build_output_record(batch, item_index, rewritten_groups):
    return {
        "question": batch["question"][item_index],
        "answers": batch["answers"][item_index],
        "privacy": batch["privacy"][item_index],
        "public": batch["public"][item_index],
        "ctxs": batch["ctxs"][item_index],
        "anonymized_ctxs": rewritten_groups[item_index],
    }


def main(argv=None):
    args = get_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    input_path = Path(args.data_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"rewrite input not found: {input_path}")
    output_path = Path(args.output_dir)
    if output_path.suffix != ".jsonl":
        raise ValueError(f"rewrite output must use the .jsonl extension: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_rewriter(args)
    qa_data = select_records(load_data(input_path), args.max_samples, args.seed)
    dataset = PromptDataset(args, tokenizer, qa_data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.get_collate_fn(),
    )
    LOGGER.info(
        "Rewrite configuration: input_records=%d prepared_records=%d max_samples=%s seed=%d "
        "batch_size=%d triple_scope=%s output=%s",
        len(qa_data),
        len(dataset),
        args.max_samples,
        args.seed,
        args.batch_size,
        args.triple_scope,
        output_path,
    )

    with torch.no_grad(), output_path.open("w", encoding="utf-8") as fout:
        for batch_index, batch in enumerate(loader):
            if batch_index % 20 == 0:
                LOGGER.info("Processed %d/%d records", batch_index * args.batch_size, len(dataset))
            rewritten_groups = []
            for prompts in batch["prompts"]:
                prompts = prompts.to(args.device)
                outputs = model.generate(
                    **prompts,
                    max_length=args.max_passage_length,
                    do_sample=False,
                )
                rewritten_groups.append(tokenizer.batch_decode(outputs, skip_special_tokens=True))

            for item_index in range(len(batch["question"])):
                record = build_output_record(batch, item_index, rewritten_groups)
                json.dump(record, fout, ensure_ascii=False)
                fout.write("\n")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    LOGGER.info("Wrote %d rewritten records to %s", len(dataset), output_path)


if __name__ == "__main__":
    main()
