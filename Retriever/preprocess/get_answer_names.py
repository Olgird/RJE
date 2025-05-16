import json
import os

from tqdm import tqdm

from config import cfg
from knowledge_graph.knowledge_graph_freebase import KnowledgeGraphFreebase


def get_answer_names(TOP_K, RESERVED_COUNT=1):
    data_dir = cfg.inference["output_dir"]
    if TOP_K <= RESERVED_COUNT:
        input_path = os.path.join(data_dir, f"K={TOP_K}.json")
    else:
        input_path = os.path.join(data_dir, f"K={TOP_K}_dc.json")
    output_path = os.path.join(data_dir, f"K_{TOP_K}_wnames.json")

    with open(input_path, "r") as f:
        data_list = [json.loads(line) for line in f]

    with open(output_path, "w") as f:
        kg = KnowledgeGraphFreebase()
        for qa_path in tqdm(data_list, desc="fetch-names"):
            path_json_list = qa_path["path_with_score"]

            for path_json in path_json_list:
                candidate_answers = path_json["tails"]
                candidate_answers = candidate_answers[:100]
                answer_names = [
                    name
                    for name in (kg.get_ent_name(ent) for ent in candidate_answers)
                    if name is not None
                ]
                path_json["candidate_answer_names"] = answer_names
            f.write(json.dumps(qa_path) + "\n")
