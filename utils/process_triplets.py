import json
import random
import os
import argparse
from collections import ChainMap
from collections import defaultdict, deque
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm


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
                example = json.loads(example)
                data.append(example)
    return data


def merge_triplets(data_each, n_retrived):
    docs = data_each["ctxs"]
    new_docs = []
    triplets = data_each["triplets"]
    triplets = dict(ChainMap(*triplets))
    global_triplets = []
    h_r = []
    global_ids = []
    i = 0
    for doc in docs:
        if i >= n_retrived:
            break
        if "id" not in doc:
            continue
        new_docs.append(doc)
        i += 1
        doc_id = doc["id"]
        global_ids.append(doc_id)
        triplets_o_doc = triplets[doc_id]
        for triplet in triplets_o_doc:
            if triplet[2] in triplet[0]:
                print("no valid triples...")
                continue
            if (triplet not in global_triplets) and (triplet[0], triplet[1]) in h_r:
                print("bad example existing...")
            if (triplet not in global_triplets) and (triplet[0], triplet[1]) not in h_r:
                global_triplets.append(triplet)
                h_r.append((triplet[0], triplet[1]))
    data_each["triplets"] = {"ids": global_ids, "trps": global_triplets}
    data_each["ctxs"] = new_docs
    return data_each

def is_qa_harm(check_lis, pri_edges, new_edge):
    graph = build_adjacency_list(pri_edges+[new_edge])
    for check in check_lis:
        if are_nodes_connected(graph, check[0], check[1]):
            return True
    return False
    
def sample_privacy(global_triplets, privacy_rate, query_entity, answers, flag):
    query_entity = [ent.text for ent in query_entity]
    num = len(global_triplets)
    random.seed(42)
    sample_size = int(num * privacy_rate)
    random_numbers = random.sample(range(num), sample_size)
    privacy = []
    r_t = {}
    pri_edges = []
    qa_check_lis=[]
    for ent in query_entity:
        for ans in answers:
            qa_check_lis.append([ent,ans])
    for i in random_numbers:
        h = global_triplets[i][0]
        t = global_triplets[i][2]
        if is_qa_harm(qa_check_lis,pri_edges,[h,t]):
            print("{}th questioning relation between {} and {} should not be privacy.".format(flag, h, t))
            flag += 1
            continue
        r_t[(global_triplets[i][1], global_triplets[i][2])] = global_triplets[i][0]
        privacy.append(global_triplets[i])
        pri_edges.append([global_triplets[i][0], global_triplets[i][2]])
    # print(pri_edges)
    a_list = build_adjacency_list(pri_edges)
    public = []
    pub_edges = []
    public_numbers = list(set([i for i in range(num)])-set(random_numbers))
    for i in public_numbers:
        # privacy and public should not be interfered
        if (global_triplets[i][1], global_triplets[i][2]) in r_t.keys():
            if global_triplets[i][0] in r_t[(global_triplets[i][1], global_triplets[i][2])] or r_t[(global_triplets[i][1], global_triplets[i][2])] in global_triplets[i][0]:
                h = global_triplets[i][0]
                t = global_triplets[i][2]
                # ensure question is not attacking privacy to easily calculate accuracy afterward.
                # if ((h in query_entity) and (t in answers)) or ((h in answers) and (t in query_entity)):
                if is_qa_harm(qa_check_lis,pri_edges,[h,t]):
                    print("{}th questioning relation between {} and {} should not be privacy.".format(flag, h, t))
                    flag += 1
                else:
                    privacy.append(global_triplets[i])
                    print("privacy and public should not be interfered")
                    continue
        if are_nodes_connected(a_list, global_triplets[i][0], global_triplets[i][2]):
            print("connection!!!")
            h = global_triplets[i][0]
            t = global_triplets[i][2]
            # ensure question is not attacking privacy to easily calculate accuracy afterward.
            # if ((h in query_entity) and (t in answers)) or ((h in answers) and (t in query_entity)):
            if is_qa_harm(qa_check_lis,pri_edges,[h,t]):
                print("{}th questioning relation between {} and {} should not be privacy.".format(flag, h, t))
                flag += 1
            else:
                privacy.append(global_triplets[i])
                continue
        public.append(global_triplets[i])
        pub_edges.append([global_triplets[i][0], global_triplets[i][2]])
    # filter privacy
    filtered_privacy = []
    b_list = build_adjacency_list(pub_edges)
    for trp in privacy:
        if are_nodes_connected(b_list, trp[0], trp[2]):
            print("filtered!!!")
            continue
        filtered_privacy.append(trp)

    return public, filtered_privacy, flag


def main(args):
    nlp = en_core_web_sm.load()
    n_retrieved = args.n_retrieved
    data_path = args.data_path
    privacy_rate = args.privacy_rate
    output_path = os.path.join(args.output_dir,
                               'hotpotqa_{}_{}_new.jsonl'.format(args.n_retrieved, int(args.privacy_rate * 100)))
    data = load_data(data_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fout:
        flag = 0
        for data_each in data:
            data_each = merge_triplets(data_each, n_retrieved)
            global_triplets = data_each["triplets"]["trps"]
            # if "title" not in data_each:
            #     continue
            query_entity = nlp(data_each["question"]).ents
            answers = data_each["answers"]
            data_each["public"], data_each["privacy"], flag = sample_privacy(global_triplets, privacy_rate,
                                                                             query_entity, answers, flag)
            json.dump(data_each, fout, ensure_ascii=False)
            fout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset/to_triplets/hotpotqa_with_triplets.jsonl",
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./dataset/sample_privacy/",
        help="Results are written to outputdir with data suffix"
    )
    parser.add_argument(
        "--n_retrieved", type=int, default=10,
        help="number of retrieved passages"
    )
    parser.add_argument(
        "--privacy_rate", type=float, default=0.25,
        help="ratio between private and the whole relations"
    )
    args = parser.parse_args()
    main(args)

