import os
import json
import random
import argparse
import re
from relik import Relik
from relik.inference.data.objects import RelikOutput
from relik.inference.annotator import Relik
from collections import defaultdict, deque
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_adjacency_list(edges):
    # print("edges=", edges)
    adjacency_list = defaultdict(list)
    for edge in edges:
        # print(edge)
        u = edge[0]
        v = edge[1]
        adjacency_list[u].append(v)
        adjacency_list[v].append(u) 
    return adjacency_list


def are_nodes_connected(adjacency_list, start, end):
    # adjacency_list = build_adjacency_list(edges)
    visited = set()
    queue = deque([start])

    while queue:
        current = queue.popleft()
        if current == end or (end in current) or (current in end) :
            return True
        if current not in visited:
            visited.add(current)
            for neighbor in adjacency_list[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
    return False


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                if k == 6051:
                    break
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


def main(args):
    relik = Relik.from_pretrained(
        "./relik-relation-extraction",
        device="cuda",
        use_nme=True
    )
    data_path = args.data_path
    data = load_data(data_path)
    random.seed(42)
    avg_connection_ratio = 0
    num = 0
    for each_data in data:
        num += 1
        if "anonymized_text" in each_data.keys():
            anonymized_text = each_data["anonymized_text"]
        else:
            anonymized_text = each_data["anonymized_ctxs"]
        privacy = each_data["privacy"]
        # print(privacy)
        total_edges = []
        for text in anonymized_text:
            triplets = relik(text).triplets
            # pairs = []
            for triplet in triplets:
                t = [triplet.subject.text, triplet.object.text]
                if (t not in total_edges) and (triplet.subject.text != triplet.object.text):
                    total_edges.append(t)
        rel_lis = build_adjacency_list(total_edges)
        connect = 0
        for pri in privacy:
            if isinstance(pri, str):
                pri = trp2list(pri)
            if are_nodes_connected(rel_lis, pri[0], pri[2]):
                connect += 1
        connection_ratio = connect / len(privacy)
        logger.info(f"Connection ratio: {connection_ratio}")
        avg_connection_ratio += connection_ratio
    avg_connection_ratio = avg_connection_ratio / num
    logger.info(f"Avg connection ratio: {avg_connection_ratio}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default= "./output_anonymized_ctxs/popqa_10_25_inferattack.jsonl",
        help=".json file containing question and answers, similar format to reader data",
    )
    args = parser.parse_args()
    main(args)


