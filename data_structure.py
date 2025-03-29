import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
import json
import random
from tqdm import tqdm, trange
from transformers import InputExample


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
        data = []
        with open(filename, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
        n = len(data)
        n = int(args.use_data_percent * n)
        # randomly sample n samples for debugging
        if n < len(data):
            random.seed(args.seed)
            data = random.sample(data, n)
        sys_prompt = '''Rewrite the given text to relace only the private tail entities with the corresponding relations from the provided private triples 
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
            # tik = 0
            for trp in public_trps:  # 105
                # if tik == 85:
                #     break
                # tik += 1
                public = public + "<csubj>{}<crel>{}<cobj>{}<ce>".format(trp[0], trp[1], trp[2])

            private_trps = record["private"]

            for trp in private_trps:  # 35
                private = private + "<rsubj>{}<rrel>{}<robj>{}<re>".format(trp[0], trp[1], trp[2])
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
            new_example = {"input_ids": input_encoding['input_ids'].flatten(), "attention_mask": input_encoding['attention_mask'].flatten(), "labels": labels["input_ids"].flatten()}
            # flat_concat = tokenizer.encode(sys_prompt+text, add_special_tokens=True, max_length=args.max_passage_length)  # 256
            # public_ids = tokenizer.encode(public, add_special_tokens=True, max_length=args.max_public_length)  # 742
            # flat_concat.extend(public_ids)
            # private_ids = tokenizer.encode(private, add_special_tokens=True, max_length=args.max_private_length)  # 252
            # flat_concat.extend(private_ids)
            # target_ids = tokenizer.encode(anonymized_text, add_special_tokens=True, max_length=args.max_passage_length)
            #
            # flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat,
            #                                                            max_pad_length=args.max_concat_length)
            # target_ids, target_mask = padding_seq_to_same_length(target_ids, max_pad_length=args.max_passage_length)

            # new_example = {"input_ids": flat_concat, "attention_mask": flat_concat_mask, "labels": target_ids,
            #                "labels_mask": target_mask}
            self.examples.append(new_example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]



class Test_Rewrite_Doc_popqa(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        data = []
        with open(filename, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
        n = len(data)
        n = int(args.use_data_percent * n)
        # randomly sample n samples for debugging
        if n < len(data):
            random.seed(args.seed)
            data = random.sample(data, n)
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
            for trp in public_trps:  # 105
                public = public + "<subj>{}<rel>{}<obj>{}<e>".format(trp[0], trp[1], trp[2])

            private_trps = record["private"]
            for trp in private_trps:  # 35
                private = private + "<subj>{}<rel>{}<obj>{}<e>".format(trp[0], trp[1], trp[2])
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
            new_example = {"input_ids": input_encoding['input_ids'].flatten(), "attention_mask": input_encoding['attention_mask'].flatten(), "labels": labels["input_ids"].flatten()}

            self.examples.append(new_example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]
