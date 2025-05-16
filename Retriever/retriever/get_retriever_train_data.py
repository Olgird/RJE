import json
import os
import random

import pandas as pd
from tqdm import tqdm

from config import cfg
from knowledge_graph.knowledge_graph_cache import KnowledgeGraphCache

EORP = "END OF REASONING PATH"
NEG_SAMPLE_NUM = 15

filtered_rels = {
    "common.topic.webpage",
    "common.topic.image",
    "freebase.type_profile.equivalent_topic",
}


def negative_sampling(
    path_and_score,
    qa_path_score,
    pos_rels,
    kg: KnowledgeGraphCache,
):
    train_data_for_one_path = []

    topic_entities = qa_path_score["topic_entities"]
    topic_entity_names = qa_path_score["topic_entity_names"]
    tpe = path_and_score["src"]
    path = path_and_score["path"]

    flag = False
    for entity, entity_name in zip(topic_entities, topic_entity_names):
        if entity == tpe:
            flag = True
            question = " ".join([qa_path_score["question"], "[SEP]", entity_name, "→"])
            current_entities = {tpe}
            break
    if not flag:
        raise ValueError

    path = path + [EORP]

    prefix_list = []

    for rel in path:
        prefix = ",".join(prefix_list)
        prefix_list.append(rel)

        data_for_one_row = []
        data_for_one_row.append(question)
        data_for_one_row.append(rel)

        neg_rels = set()
        for entity in current_entities:
            neg_rels.update(kg.get_relation(entity))

        neg_rels = list(neg_rels)
        neg_rels.append(EORP)
        neg_rels = [r for r in neg_rels if r not in pos_rels[prefix]]

        # neg_rels.add(EORP)
        # neg_rels = {r for r in neg_rels if r not in pos_rels[prefix]}

        if not neg_rels:
            break

        if len(neg_rels) >= NEG_SAMPLE_NUM:
            sample_rels = random.sample(neg_rels, NEG_SAMPLE_NUM)
            if EORP in neg_rels and EORP not in sample_rels:
                sample_rels[random.randint(0, len(sample_rels) - 1)] = EORP
            data_for_one_row.extend(sample_rels)

        else:
            sub = NEG_SAMPLE_NUM - len(neg_rels)
            sample_rels = []
            while len(sample_rels) < sub:
                sample_rels.extend(neg_rels)

            sample_rels = random.sample(sample_rels, sub)
            neg_rels.extend(sample_rels)
            random.shuffle(neg_rels)
            data_for_one_row.extend(neg_rels)
        train_data_for_one_path.append(data_for_one_row)

        # Update for next step
        if rel != EORP:
            question += f" {rel} #"
            tmp_entities = set()
            for entity in current_entities:
                tmp_entities.update(kg.get_tail(entity, rel))
            current_entities = tmp_entities

    return tpe, train_data_for_one_path


def get_retriever_train_data() -> None:
    dataset = cfg.dataset
    input_path = cfg.tmp_dir + "path.json"
    output_dir = cfg.train_dir
    output_path = cfg.retriever["train"]["input_path"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_path):
        return

    with open(input_path, "r") as f:
        data_list = [json.loads(line) for line in f]

    kg = KnowledgeGraphCache()

    if dataset == "webqsp":
        threshold = 0.3
    elif dataset == "cwq":
        threshold = 0.5

    column_names = ["question", "pos"] + [f"neg{i}" for i in range(NEG_SAMPLE_NUM)]
    pd.DataFrame(columns=column_names).to_csv(output_path, header=True, index=False)

    for qa_path_score in tqdm(data_list, desc="negative-sampling"):
        path_and_score_list = qa_path_score["path_and_score"]
        if not path_and_score_list:
            continue
        max_score = path_and_score_list[0]["score"]
        # if max_score < 0.1:
        #     continue

        filtered_list = []

        min_len = float("inf")
        for path_and_score in path_and_score_list:
            path = path_and_score["path"]
            score = path_and_score["score"]
            if (
                path
                and (score >= threshold or score >= max_score)
                and not any(rel in filtered_rels for rel in path)
            ):
                filtered_list.append(path_and_score)
                if len(path) < min_len:
                    min_len = len(path)

        filtered_list = [
            path_and_score
            for path_and_score in filtered_list
            if len(path_and_score["path"]) <= min_len
        ]

        filtered_list = filtered_list[:25]

        if not filtered_list:
            continue

        """
            pos_rels -> {'': {'rel1'}, 'rel1': {'rel2'}, 'rel1,rel2': {'rel3'}, ...}
            prefix_list -> ['rel1', 'rel2', 'rel3', ...]       
        """
        pos_rels = {}
        for path_and_score in filtered_list:
            path = path_and_score["path"]
            path = path + [EORP]
            prefix_list = []
            for rel in path:
                prefix = ",".join(prefix_list)
                if prefix not in pos_rels:
                    pos_rels[prefix] = set()
                pos_rels[prefix].add(rel)
                prefix_list.append(rel)

        entity_hit_dict = dict()
        for entity, entity_name in zip(
            qa_path_score["topic_entities"], qa_path_score["topic_entity_names"]
        ):
            entity_hit_dict[entity] = entity_name

        for path_and_score in filtered_list:
            entity_hit, train_data = negative_sampling(
                path_and_score, qa_path_score, pos_rels, kg
            )
            if train_data is not None:
                df = pd.DataFrame(train_data)
                entity_hit_dict[entity_hit] = True
                df.to_csv(output_path, mode="a", header=False, index=False)

        # Search for topic entities without any path
        for entity, entity_name in entity_hit_dict.items():
            if entity_name is not True:
                neg_rels = list(kg.get_relation(entity))
                if not neg_rels:
                    continue
                data_for_one_row = []
                question = " ".join(
                    [qa_path_score["question"], "[SEP]", entity_name, "→"]
                )
                data_for_one_row.append(question)
                data_for_one_row.append(EORP)

                if len(neg_rels) >= NEG_SAMPLE_NUM:
                    sample_rels = random.sample(neg_rels, NEG_SAMPLE_NUM)
                    data_for_one_row.extend(sample_rels)
                else:
                    sub = NEG_SAMPLE_NUM - len(neg_rels)
                    sample_rels = []
                    while len(sample_rels) < sub:
                        sample_rels.extend(neg_rels)

                    sample_rels = random.sample(sample_rels, sub)
                    neg_rels.extend(sample_rels)
                    random.shuffle(neg_rels)
                    data_for_one_row.extend(neg_rels)
                df = pd.DataFrame([data_for_one_row])
                df.to_csv(output_path, mode="a", header=False, index=False)
