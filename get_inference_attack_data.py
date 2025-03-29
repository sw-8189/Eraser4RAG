import json
import os
import argparse
from collections import ChainMap
import re
from collections import defaultdict, deque
from tqdm import tqdm


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
        print("no matching content")
    return result


def organize_triplets(data_each):
    triplets = data_each["triplets"]
    triplets = dict(ChainMap(*triplets))
    return triplets


def to_edge(trps):
    return [[trp[0], trp[2]] for trp in trps]


def supplement(lis_1, lis_2):
    return [edge for edge in lis_1 if edge not in lis_2]


def build_adjacency_list(edges):
    adjacency_list = defaultdict(list)
    for edge in edges:
        # print(edge)
        u = edge[0]
        v = edge[1]
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
    return adjacency_list


def are_nodes_connected(adjacency_list, start, end):
    visited = set()
    queue = deque([start])

    while queue:
        current = queue.popleft()
        if current == end or (end in current) or (current in end):
            return True
        if current not in visited:
            visited.add(current)
            for neighbor in adjacency_list[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
    return False


def main(args):
    output_path = os.path.join(args.output_dir,
                               'inferattack_file.jsonl')
    data_path_0 = args.data_0
    data_path_1 = args.data_1
    data_0 = load_data(data_path_0)
    data_1 = load_data(data_path_1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_num = 0

    with open(output_path, "w") as fout:
        for each_data_0, each_data_1 in tqdm(zip(data_0, data_1)):
            if ("anonymized_text" in each_data_1.keys()) and len(each_data_1["anonymized_text"]) != len(each_data_1["ctxs"]):
                continue
            total = to_edge(each_data_1["triplets"]["trps"])
            privacy = to_edge(each_data_1["privacy"])
            triplets_dict = organize_triplets(each_data_0)
            unprivacy = [edge for edge in total if edge not in privacy]
            ctxs = each_data_1["ctxs"]
            flag = 0
            for ctx in ctxs:
                ids = ctx["id"]
                privacy_k = to_edge([trp2list(pri) for pri in ctx["private"]])
                trps_k = to_edge(triplets_dict[ids])
                unprivacy_k = [edge for edge in trps_k if edge not in privacy_k]
                unprivacy_k_supp = supplement(unprivacy, unprivacy_k)
                unprivacy_k_supp_graph = build_adjacency_list(unprivacy_k_supp)
                privacy_k_supp = supplement(privacy, privacy_k)
                privacy_k_supp_graph = build_adjacency_list(privacy_k_supp)
                for pri in privacy_k:
                    if are_nodes_connected(privacy_k_supp_graph, pri[0], pri[1]) != True and are_nodes_connected(
                            unprivacy_k_supp_graph, pri[0], pri[1]):
                        total_num += 1
                        flag = 1
                        print(f"pri={pri}\n")
                        print(f"ctx={ctx}\n")
                        json.dump(each_data_1, fout, ensure_ascii=False)
                        fout.write("\n")
                        break
                if flag == 1:
                    break
            if total_num==5:
                break
            # if flag == 0:
                # print("dump one!")
        print(f"total num: {total_num}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_0",
        type=str,
        default="./dataset/to_triplets/popQA_with_triplets.jsonl",
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument(
        "--data_1",
        type=str,
        default='./dataset/sample_privacy/popqa_trp.jsonl',
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument(
        "--output_dir", type=str, default= "./dataset/sample_privacy/"
        help="Results are written to outputdir with data suffix")

    args = parser.parse_args()
    main(args)
