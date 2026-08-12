import json
import random

from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from utils.triple_utils import (
        serialize_private_prompt_triple,
        serialize_public_prompt_triple,
    )
except (ImportError, ModuleNotFoundError):
    # Compatibility fallback while the shared v2 utility is being introduced.
    def serialize_public_prompt_triple(triple):
        return "<csubj>{}<crel>{}<cobj>{}<ce>".format(triple[0], triple[1], triple[2])

    def serialize_private_prompt_triple(triple):
        return "<rsubj>{}<rrel>{}<robj>{}<re>".format(triple[0], triple[1], triple[2])


def load_jsonl(filename):
    records = []
    with open(filename, "r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {filename}") from exc
    return records


def select_records(records, args):
    use_data_percent = getattr(args, "use_data_percent", 1.0)
    if not 0 < use_data_percent <= 1:
        raise ValueError("use_data_percent must be in (0, 1]")

    requested = int(len(records) * use_data_percent)
    max_samples = getattr(args, "max_samples", None)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        requested = min(requested, max_samples)

    if requested >= len(records):
        return list(records)
    rng = random.Random(getattr(args, "seed", 42))
    indices = sorted(rng.sample(range(len(records)), requested))
    return [records[index] for index in indices]


def padding_seq_to_same_length(input_ids, max_pad_length, pad_token=0):
    padding_length = max_pad_length - len(input_ids)
    padding_ids = [pad_token] * padding_length
    attention_mask = []

    if padding_length <= 0:
        attention_mask = [1] * max_pad_length
        input_ids = input_ids[:max_pad_length]
    else:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + padding_ids

    assert len(input_ids) == max_pad_length
    assert len(attention_mask) == max_pad_length

    return input_ids, attention_mask


class Anonymize_Doc_popqa(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        data = select_records(load_jsonl(filename), args)
        sys_prompt = '''Rewrite the given text to replace only the private tail entities with the corresponding relations from the provided private triples
                        while preserving all head entities, closely mirroring the original text, ensuring the text retains relevant public information and any details not mentioned in the triples.'''
        for record in tqdm(data):
            if args.use_prefix:
                public = "\n\npublic triples: "
                private = "\n\nprivacy triples: "
                text = "\n\ntext: "
            else:
                public = ""
                private = ""
                text = ""
            public_trps = record["public"]
            for trp in public_trps:
                public += serialize_public_prompt_triple(trp)

            private_trps = record["private"]
            for trp in private_trps:
                private += serialize_private_prompt_triple(trp)
            text += record["text"]
            anonymized_text = record["anonymized_text"]

            input_text = sys_prompt + text + private + public + "**Anonymized Text**:"
            input_encoding = tokenizer.encode_plus(
                input_text,
                max_length=args.max_concat_length,
                add_special_tokens=True,
                return_token_type_ids=False,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            labels = tokenizer.encode_plus(
                anonymized_text,
                add_special_tokens=True,
                max_length=args.max_passage_length,
                return_token_type_ids=False,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            new_example = {
                "input_ids": input_encoding["input_ids"].flatten(),
                "attention_mask": input_encoding["attention_mask"].flatten(),
                "labels": labels["input_ids"].flatten(),
            }
            self.examples.append(new_example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]



class Test_Rewrite_Doc_popqa(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        data = select_records(load_jsonl(filename), args)
        sys_prompt = '''Rewrite the given text to remove or generalize only the private relations and their tail entities from the provided private triples 
                        while preserving all head entities, ensuring the text retains relevant public information and any details not mentioned in the triples.'''
        for record in tqdm(data):
            if args.use_prefix:
                public = "public triples: "
                private = "privacy triples: "
                text = "text: "
            else:
                public = ""
                private = ""
                text = ""
            public_trps = record["public"]
            for trp in public_trps:
                public += serialize_public_prompt_triple(trp)

            private_trps = record["private"]
            for trp in private_trps:
                private += serialize_private_prompt_triple(trp)
            text += record["text"]
            anonymized_text = record["anonymized_text"]

            input_text = sys_prompt + text + private + public
            input_encoding = tokenizer.encode_plus(
                input_text,
                max_length=args.max_concat_length,
                add_special_tokens=True,
                return_token_type_ids=False,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            labels = tokenizer.encode_plus(
                anonymized_text,
                add_special_tokens=True,
                max_length=args.max_passage_length,
                return_token_type_ids=False,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            new_example = {
                "input_ids": input_encoding["input_ids"].flatten(),
                "attention_mask": input_encoding["attention_mask"].flatten(),
                "labels": labels["input_ids"].flatten(),
            }

            self.examples.append(new_example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]
