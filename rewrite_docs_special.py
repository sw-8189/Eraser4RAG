import os
import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import logging
import random
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PromptDataset(Dataset):
    def __init__(self, args, tokenizer, data):
        self.examples = []
        for record in tqdm(data):
            prompt_groups, public, privacy = format_prompt(record, args)
            prompt_ids = tokenizer(prompt_groups, return_tensors="pt", padding=True, max_length=1300, truncation=True)
            if "qa" in record.keys():
                if len(record["qa"])==0:
                    continue
                new_example = {"question": record["qa"][0]["question"], "answers": record["qa"][0]["answer"], "public": public, "privacy": privacy, "prompts": prompt_ids}
            else:
                new_example = {"question": record["question"], "answers": record["answers"], "public": public, "privacy": privacy, "prompts": prompt_ids}
            self.examples.append(new_example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        def collate_fn(batch: list):
            collated_dict = {"question": [],
                             "answers": [],
                             "public": [],
                             "privacy": [],
                             "prompts": []}
            for example in batch:
                collated_dict["question"].append(example["question"])
                collated_dict["answers"].append(example["answers"])
                collated_dict["privacy"].append(example["privacy"])
                collated_dict["public"].append(example["public"])
                collated_dict["prompts"].append(example["prompts"])

            return collated_dict

        return collate_fn


def format_prompt(data_each, args):
    input_prompts = []
    input_privacy = []
    input_public = []
    sys_prompt = '''Rewrite the given text to remove or generalize only the private relations and their tail entities from the provided private triples 
                            while preserving all head entities, ensuring the text retains relevant public information and any details not mentioned in the triples.'''
    if args.use_prefix:
        public = "\n\npublic triples: "
        private = "\n\nprivacy triples: "
        text = "\n\ntext: "
    else:
        public = ""
        private = ""
        text = ""
    public_trps = data_each["public"]
    for trp in public_trps:  # 105
        # input_public.append("<subj>{}<rel>{}<obj>{}<e>".format(trp[0], trp[1], trp[2]))
        public = public + "<csubj>{}<crel>{}<cobj>{}<ce>".format(trp[0], trp[1], trp[2])

    private_trps = data_each["privacy"]
    for trp in private_trps:  # 35
        # input_privacy.append("<subj>{}<rel>{}<obj>{}<e>".format(trp[0], trp[1], trp[2]))
        private = private + "<rsubj>{}<rrel>{}<robj>{}<re>".format(trp[0], trp[1], trp[2])
    for ctx in data_each["ctxs"]:
        doc_text = text + ctx["text"]
        input_text = sys_prompt + doc_text + private + public + "**Anonymized Text**:"  #public + 
        input_prompts.append(input_text)
        input_public.append(ctx["public"])
        input_privacy.append(ctx["private"])
    return input_prompts,input_public, input_privacy


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="./dataset/special_data/popqa_10_25_special.jsonl")  
    parser.add_argument("--model_dir", type=str, default="./output_checkpoint/RL/step_3300") 
    parser.add_argument("--output_dir", type=str, default="./output_anonymized_ctxs/popqa_10_25_special.jsonl")
    parser.add_argument("--max_passage_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_prefix", type=bool, default=True)
    args = parser.parse_args()

    # pytorch parallel gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # , args.local_rank)
    args.device = device

    return args


if __name__ == "__main__":
    args = get_args()

    logger.info("Loading Flan-T5 rewrite model...")
    tokenizer = AutoTokenizer.from_pretrained('./flan-t5-large') 
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(args.device)
    model.eval()
    logger.info("Loading QA dataset with docs to rewrite...")
    qa_data = load_data(args.data_dir)
    random.seed(42)
    qa_prompt_dataset = PromptDataset(args, tokenizer, random.sample(qa_data,100))
    qa_prompt_loader = DataLoader(qa_prompt_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=qa_prompt_dataset.get_collate_fn(args))

    output_path = args.output_dir
    logger.info(output_path)
    process_time=[]
    with torch.no_grad():
        with open(output_path, "w") as fout:
            for i, batch in enumerate(qa_prompt_loader):
                if i % 20 == 0:
                    logger.info("Processed {}/{}".format(i*args.batch_size, len(qa_prompt_dataset)))
                bt_prompts = batch['prompts']
                bt_rewritten_docs = []
                for prompts in bt_prompts:
                    prompts = prompts.to(args.device)
                    start_time=time.perf_counter()
                    outputs = model.generate(**prompts, max_length=args.max_passage_length)
                    inference_time = time.perf_counter() - start_time
                    process_time.append(inference_time)
                    rewritten_docs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    bt_rewritten_docs.append(rewritten_docs)
                logger.info(f"inference_time={np.mean(process_time)}")
                for k in range(args.batch_size):
                    new_data = dict(question=batch["question"][k],
                                    answers=batch["answers"][k],
                                    private=batch["privacy"][k],
                                    public=batch["public"][k],
                                    anonymized_ctxs=bt_rewritten_docs[k])
                    json.dump(new_data, fout, ensure_ascii=False)
                    fout.write("\n")
                del prompts
                del outputs
                torch.cuda.empty_cache()



