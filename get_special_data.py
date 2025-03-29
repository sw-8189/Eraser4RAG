import os
import argparse
import numpy as np
from metrics import match, accuracy
import json
import random
import re
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

def trp2list(trp_text):
    pattern = r"<subj>(.*?)<rel>(.*?)<obj>(.*?)<e>"
    matches = re.search(pattern, trp_text)
    if matches:
        result = list(matches.groups())
    else:
        print("no matching content.")
    return result
    

def select_special_ctx(each_data):
    ctxs = each_data["ctxs"]
    new_ctxs = []
    new_anonymized_ctxs = []
    if "anonymized_text" in each_data.keys():
        if len(each_data["anonymized_text"]) != len(ctxs):
            return None,0
    for i in range(len(ctxs)):
        ctx=ctxs[i]
        privacy = ctx["private"]
        privacy = [trp2list(pri) for pri in privacy]
        private_tail = {}
        for pri in privacy:
            if pri[2] in private_tail.keys():
                private_tail[pri[2]].append(pri[:2])
            else:
                private_tail[pri[2]] = [pri[:2]]
        public = ctx["public"]
        for pub in public:
            pub_lis = trp2list(pub)
            if (pub_lis[2] in private_tail.keys()) and (pub_lis[:2] not in private_tail[pub_lis[2]]):
                print("get one!")
                new_ctxs.append(ctx)
                if "anonymized_text" in each_data.keys():
                    new_anonymized_ctxs.append(each_data["anonymized_text"][i])
                break
    each_data["ctxs"] = new_ctxs
    if "anonymized_text" in each_data.keys():
        each_data["anonymized_text"] = new_anonymized_ctxs
    return each_data, len(new_ctxs)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='./dataset/sample_privacy/popqa_trp.jsonl') 
    parser.add_argument("--output_dir", type=str, default="./dataset/special_data/popqa_special.jsonl")
    args = parser.parse_args()

    data = load_data(args.input_file)
    output_path = args.output_dir
    total = 0
    with open(output_path, "w") as fout:
        for each_data in data:
            new_data, num = select_special_ctx(each_data)
            if new_data is None:
                continue
            print(f"num of special ctxs={num}")
            if num>1:
                total += 1
                json.dump(new_data, fout, ensure_ascii=False)
                fout.write("\n")
        print(f"total num = {total}")
if __name__ == '__main__':
    main()
