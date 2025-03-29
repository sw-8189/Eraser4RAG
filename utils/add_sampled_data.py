import json
import random
import os
import argparse
from collections import ChainMap
from utilities import *


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


def organize_triplets(data_each):
    triplets = data_each["triplets"]
    triplets = dict(ChainMap(*triplets))
    return triplets


def match_triplets(triplets_dict, pub, pri, ctxs_lis):
    private = process_private_trps(pri)
    public = process_private_trps(pub)
    for ctx in ctxs_lis:
        ids = ctx["id"]
        trp_each = triplets_dict[ids]
        true_triplets = process_private_trps(trp_each)
        ctx["private"] = [private[key] for key in list(set(true_triplets.keys()) & set(private.keys()))]
        ctx["public"] = [public[key] for key in list(set(true_triplets.keys()) & set(public.keys()))]
    return ctxs_lis


def main(args):
    output_path = args.output_dir
    data_path_0 = args.data_0
    data_path_1 = args.data_1
    data_0 = load_data(data_path_0)
    data_1 = load_data(data_path_1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fout:
        for data_each_0, data_each_1 in zip(data_0, data_1):
            triplets_dict = organize_triplets(data_each_0)
            pub = data_each_1["public"]
            pri = data_each_1["privacy"]
            ctxs_lis = data_each_1["ctxs"]
            data_each_1["ctxs"] = match_triplets(triplets_dict, pub, pri, ctxs_lis)
            json.dump(data_each_1, fout, ensure_ascii=False)
            fout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_0",
        type=str,
        default="./dataset/to_triplets/hotpotqa_with_triplets.jsonl",
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument(
        "--data_1",
        type=str,
        default="./dataset/sample_privacy/hotpotqa_10_25.jsonl",
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./dataset/sample_privacy/hotpotqa_10_25_trp.jsonl'",
        help="Results are written to outputdir with data suffix")

    args = parser.parse_args()
    main(args)
