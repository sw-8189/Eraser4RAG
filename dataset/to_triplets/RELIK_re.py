from typing import List, Optional, Union
from relik import Relik
from relik.inference.data.objects import RelikOutput
from relik.inference.annotator import Relik
import json
import glob
import os
import argparse
import torch
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

def main(args):
    relik = Relik.from_pretrained(
    "./relik-relation-extraction",
    device="cuda",
    use_nme=True
    )
    data_paths = args.data
    output_path = args.output_dir  # a .jsonl file
    data = load_data(data_paths)
    # data_already = load_data(output_path)
    # data = data[len(data_already):]
    # del data_already
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    num = 0
    with open(output_path, "w") as fout:
        with torch.no_grad():
            for data_each in data:
                triplets_each = []
                num += 1
                if num % 20 == 0:
                    logger.info("Processed {}/{}".format(num, len(data)))
                for doc in data_each["ctxs"]:
                    if "id" not in doc.keys():
                        continue
                    doc_id = doc["id"]
                    text = doc["text"]
                    if text == '':
                        triplets_each.append({doc_id: []})
                        continue
                    triplets = relik(text).triplets
                    true_triplets = []
                    pairs = []
                    for triplet in triplets:
                        t = (triplet.subject.text, triplet.label, triplet.object.text)
                        if (t not in true_triplets) and (t[0] != t[2]) and ((t[0],t[2]) not in pairs) and ((t[2], t[0]) not in pairs):
                            true_triplets.append(t)
                            pairs.append((t[0],t[2]))
                    triplets_each.append({doc_id: true_triplets})  # {"id": doc_id, "trps": true_triplets}
                data_each["triplets"] = triplets_each
                # write data_each in fout
                json.dump(data_each, fout, ensure_ascii=False)
                fout.write("\n")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default="./dataset/retrieved_data/popqa_retrieved_clean.jsonl",
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./dataset/to_triplets/popqa_with_triplets.jsonl", help="Results are written to outputdir with data suffix"
    )
    args = parser.parse_args()
    main(args)

