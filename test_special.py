import os
from typing import List, Optional, Union
from relik import Relik
from relik.inference.data.objects import RelikOutput
from relik.inference.annotator import Relik
import random
import json
import argparse
import torch
# from utils.utilities import *
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
                # if k == 150:
                #     break
                example = json.loads(example)
                data.append(example)
    return data


def main(args):
    relik = Relik.from_pretrained(
        "./relik-relation-extraction",
        device="cuda",
        use_nme=True
    )
    random.seed(42)
    data_paths = args.data
    data = load_data(data_paths)
    data = random.sample(data, 100)

    num = 0
    total = 0
    avg_hit_ratio = 0
    avg_private_hit_ratio = 0
    avg_hit_ratio_public = 0
    with torch.no_grad():
        for data_each in data:
            if "private" in data_each.keys():
                private = data_each["private"]
                public = data_each["public"]
            # num_privacy = len(private)
            # if num_privacy<=1:
            #     continue
            triplets_each = []
            num += 1
            if num % 20 == 0:
                logger.info("Processed {}/{}".format(num, len(data)))
            if "anonymized_ctxs" in data_each.keys() or "anonymized_text" in data_each.keys():
                if "anonymized_text" in data_each.keys():
                    data_each["anonymized_ctxs"] = data_each["anonymized_text"]
                private = [ctx["private"] for ctx in data_each["ctxs"]]
                public = [ctx["public"] for ctx in data_each["ctxs"]]
                len_data_each = len(data_each["anonymized_ctxs"])
                for i in range(len_data_each):
                    text = data_each["anonymized_ctxs"][i]
                    if text == '':
                        triplets_each.append([])
                        continue
                    triplets = relik(text).triplets
                    true_triplets = []
                    # pairs = []
                    for triplet in triplets:
                        t = "<subj>{}<rel>{}<obj>{}<e>".format(triplet.subject.text, triplet.label, triplet.object.text)
                        if (t not in true_triplets) and (triplet.subject.text != triplet.object.text):
                            true_triplets.append(t)
                            # pairs.append((triplet.subject.text, triplet.subject.text))
                    # num_hit = len(list(set(true_triplets) & set(private)))
                    num_hit_private = len(list(set(true_triplets) & set(private[i])))
                    num_hit_public = len(list(set(true_triplets) & set(public[i])))
                    num_pri = len(private[i])
                    num_pub = len(public[i])
                    # hit_ratio = num_hit / num_trps if num_trps > 0 else 0
                    hit_ratio_private = num_hit_private / num_pri if num_pri > 0 else 0.0
                    hit_ratio_public = num_hit_public / num_pub if num_pub > 0 else 1.0
                    # hit_ratio = hit_ratio_public - hit_ratio_private
                    avg_hit_ratio_public += hit_ratio_public
                    avg_private_hit_ratio += hit_ratio_private
                    total+=1
                    logger.info(
                        f"No.{num} data, No.{i} text, public_hit_ratio = {hit_ratio_public}, privacy_hit_ratio = {hit_ratio_private}")  # {"id": doc_id, "trps": true_triplets}

                    torch.cuda.empty_cache()
           
                
        avg_hit_ratio_public = (avg_hit_ratio_public / total) if (
                    "anonymized_ctxs" in data_each.keys()) else (avg_hit_ratio_public / total)
        avg_private_hit_ratio = (avg_private_hit_ratio / total) if (
                "anonymized_ctxs" in data_each.keys()) else (avg_private_hit_ratio / total)
        
        logger.info(f"avg_hit_ratio_public = {avg_hit_ratio_public}, avg_private_hit_ratio = {avg_private_hit_ratio}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default= "./output_anonymized_ctxs/popqa_10_25_special.jsonl"
        help=".json file containing question and answers, similar format to reader data",
    )

    args = parser.parse_args()
    main(args)
