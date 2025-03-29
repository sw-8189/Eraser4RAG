import re
import glob
import os
import json
import argparse
import spacy, coreferee
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import torch
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

def coref_text(coref_nlp, text):
    coref_doc = coref_nlp(text)
    resolved_text = ""

    for token in coref_doc:
        repres = coref_doc._.coref_chains.resolve(token)
        if repres:
            resolved_text += " " + " and ".join(
                [
                    t.text
                    if t.ent_type_ == ""
                    else [e.text for e in coref_doc.ents if t in e][0]
                    for t in repres
                ]
            )
        else:
            resolved_text += " " + token.text

    return resolved_text

def main(args):
    coref_nlp = spacy.load('en_core_web_lg')
    coref_nlp.add_pipe('coreferee')
    t = 0
    data = load_data(args.data)
    output_path = args.output_dir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fout:
        for data_each in data:
            t += 1
            if t % 100 == 0:
                logger.info("Processed {}/{}".format(t, len(data)))
            for example in data_each["ctxs"]:
                text = example["text"]
                title = example["title"]

                example["text"] = coref_text(coref_nlp, text)
            json.dump(data_each, fout, ensure_ascii=False)
            fout.write("\n")

    logger.info(f"Saved results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default="./dataset/retrieved_data/triviaqa_retrieved.jsonl",
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./dataset/retrieved_data/triviaqa_retrieved_clean.jsonl", help="Results are written to outputdir with data suffix"
    )
    args = parser.parse_args()
    main(args)
   
