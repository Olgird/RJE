import json
import os
import time
from collections import deque

from config import cfg
from knowledge_graph.knowledge_graph_cache import KnowledgeGraphCache


def generate_paths(
    qa,
    paths_with_score: dict,
    kg: KnowledgeGraphCache,
    max_hop: int,
    path_max: int = 50,
):
    answers = {answer["kb_id"] for answer in qa["answers"]}

    for entity in qa["topic_entities"]:
        # head_queue [[head, current_hop, current_path], ...]
        head_queue = deque()
        min_hop = max_hop
        head_queue.append([{entity}, 0, []])
        current_entity_time = time.time()
        while head_queue:
            if time.time() - current_entity_time > 150:
                break
            heads, current_hop, current_path = head_queue.popleft()
            if current_hop > max_hop or current_hop > min_hop:
                break
            hit = answers & heads
            score = len(hit) / len(heads)
            if score > paths_with_score.get(tuple(current_path), 0):
                if min_hop > current_hop and current_hop > 1:
                    min_hop = current_hop
                paths_with_score.update({tuple(current_path): score})
            if len(paths_with_score) >= path_max:
                break
            if current_hop == max_hop or current_hop == min_hop:
                continue
            entity_dict_list = [kg.get_rel_tail(head) for head in heads]
            for entity_dict in entity_dict_list:
                for rel in entity_dict:
                    head_queue.append(
                        [
                            {tail for tail in entity_dict[rel]},
                            current_hop + 1,
                            current_path + [rel],
                        ]
                    )
    return


def search_path_score_multiprocess():
    input_path = cfg.processed_dataset + "train.json"
    output_dir = cfg.tmp_dir
    output_path = cfg.tmp_dir + "path.json"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(output_path):
        return output_path, 0, 0, None

    if cfg.dataset == "webqsp":
        num_processes = 24
    elif cfg.dataset == "cwq":
        num_processes = 8
        # epoch = 1

    with open(input_path, "r") as f:
        lines = f.readlines()
    # num_chunks = num_processes * epoch
    total_len = len(lines)
    chunk_size = len(lines) // num_processes
    tasks = [
        (input_path, i * chunk_size, (i + 1) * chunk_size)
        for i in range(num_processes - 1)
    ]

    tasks.append((input_path, (num_processes - 1) * chunk_size, len(lines)))
    # len_list = [chunk_size] * (num_chunks - 1) + [
    #     len(lines) - (num_chunks - 1) * chunk_size
    # ]
    return output_path, num_processes, total_len, tasks


def search_path_score_wrapper(args):
    return search_path_score(*args)


def search_path_score(input_path, start, end, counter, counter_lock):
    dataset = cfg.dataset
    with open(input_path, "r") as f:
        data_list = [json.loads(line) for i, line in enumerate(f) if start <= i < end]

    kg = KnowledgeGraphCache()

    if dataset == "webqsp":
        max_hop = 2
    elif dataset == "cwq":
        max_hop = 4

    total_list = []
    for qa in data_list:
        paths_with_score = dict()
        generate_paths(qa, paths_with_score, kg, max_hop=max_hop)
        # if not paths_with_score:
        #     print("No path:" + qa["question"])

        path_list = sorted(paths_with_score.items(), key=lambda x: x[1], reverse=True)
        data_dict = {
            "id": qa["id"],
            "question": qa["question"],
            "topic_entities": qa["topic_entities"],
            "topic_entity_names": qa["topic_entity_names"],
            "answers": qa["answers"],
            "path_and_score": [
                {"path": list(path[0]), "score": path[1]} for path in path_list
            ],
        }
        total_list.append(data_dict)
        with counter_lock:
            counter.value += 1
    return total_list
