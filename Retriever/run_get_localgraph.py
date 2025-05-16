"""Get_local_subgraph_multiprocess"""

import json
import os
import time
from multiprocessing import Manager, Pool

from tqdm import tqdm

from config import cfg
from knowledge_graph.knowledge_graph_freebase import KnowledgeGraphFreebase
from preprocess.get_seeds import get_seeds

dataset = cfg.dataset
if dataset == "webqsp":
    hop = 2
    num_processes = 24
    LIMIT1 = 500
    LIMIT2 = 800
elif dataset == "cwq":
    hop = 3
    num_processes = 20
    LIMIT1 = 500
    LIMIT2 = 300
else:
    raise NotImplementedError


def process_data_wrapper(args):
    return process_data(*args)


def process_data(
    part,
    kg: KnowledgeGraphFreebase,
    current_hop,
    entity_dict,
    shared_dict,
    lock,
    counter,
    counter_lock,
):
    global LIMIT2
    if current_hop == 1:
        for entity in part:
            get_one_hop(entity, kg, entity_dict)
            with counter_lock:
                counter.value += 1
    else:
        if current_hop == 3:
            LIMIT2 //= 3
        for entity in part:
            get_multi_hop(entity, kg, entity_dict, current_hop, shared_dict, lock)
            with counter_lock:
                counter.value += 1
    return entity_dict


def get_one_hop(entity, kg: KnowledgeGraphFreebase, entity_dict):
    rel_list = kg.get_relation(entity, LIMIT1)
    entity_dict[entity] = dict()
    for rel in rel_list:
        tail = kg.get_tail(entity, rel, LIMIT2)
        if tail and len(tail) < LIMIT2:
            entity_dict[entity].update({rel: tail})


def get_multi_hop(
    entity, kg: KnowledgeGraphFreebase, entity_dict, current_hop, shared_dict, lock
):
    heads = {item for sublist in entity_dict[entity].values() for item in sublist}

    heads_new = {head for head in heads if head not in entity_dict}
    if not heads_new:
        return
    with lock:
        heads_new = {head for head in heads_new if head not in shared_dict}
        for head in heads_new:
            shared_dict[head] = True
    for head in heads_new:
        entity_dict[head] = dict()
        rel_list = kg.get_relation(head, LIMIT1)
        for rel in rel_list:
            tail = kg.get_tail(head, rel, LIMIT2)
            if tail and len(tail) < LIMIT2:
                entity_dict[head].update({rel: tail})


def load_seed_and_cache():
    global hop

    get_seeds()
    output_dir = cfg.localgraph_dir
    output_path = cfg.localgraph["path"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(output_path):
        return None, None, None, None

    if os.path.exists(output_dir + "2-hop.json") and hop == 3:
        input_path = output_dir + "2-hop.json"
        current_hop = 3
        entity_dict = {}
        with open(input_path, "r") as f:
            for line in f:
                entity_dict.update(json.loads(line))
        entities = list(entity_dict.keys())
    else:
        if hop == 3:
            print(
                "No 2-hop data found, generate 2-hop graph cache first...After that, run this script again."
            )
            hop = 2
            output_path = output_dir + "2-hop.json"
        input_path = cfg.localgraph["seed_path"]
        current_hop = 1
        entity_dict = {}
        with open(input_path, "r") as f:
            entities = [line.strip() for line in f]
    return output_path, current_hop, entity_dict, entities


if __name__ == "__main__":
    output_path, current_hop, entity_dict, entities = load_seed_and_cache()

    kg = KnowledgeGraphFreebase()

    total_len = len(entities)
    entities_parts = [entities[i::num_processes] for i in range(num_processes)]
    timeout = 180

    while current_hop <= hop:
        print(f"Processing {current_hop}-hop data...")
        with Manager() as manager:
            shared_dict = manager.dict()
            lock = manager.Lock()
            counter_lock = manager.Lock()
            counter = manager.Value("i", 0)
            with Pool(processes=num_processes) as pool:
                pbar = tqdm(total=total_len)
                results = pool.imap_unordered(
                    process_data_wrapper,
                    [
                        (
                            part,
                            kg,
                            current_hop,
                            entity_dict,
                            shared_dict,
                            lock,
                            counter,
                            counter_lock,
                        )
                        for part in entities_parts
                    ],
                )

                last_update_time = time.time()  # Record the time of the last update
                last_progress = 0  # Record the last progress

                while True:
                    pbar.n = counter.value
                    pbar.refresh()
                    if pbar.n != last_progress:
                        last_update_time = time.time()
                        last_progress = pbar.n
                    elif time.time() - last_update_time > timeout:
                        print("Timeout! Terminating...")
                        break
                    if pbar.n >= total_len:
                        pbar.close()
                        break
                    time.sleep(2)

                entity_dict = {}
                for result in results:
                    for key, value in result.items():
                        if key not in entity_dict:
                            entity_dict[key] = value

        current_hop += 1

    with open(output_path, "w") as outf:
        for key, value in entity_dict.items():
            outf.write(f'{{"{key}": {json.dumps(value, ensure_ascii=False)}}}\n')
