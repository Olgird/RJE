import heapq
import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import cfg
from discriminator.model import Discriminator
from knowledge_graph.knowledge_graph_cache import KnowledgeGraphCache

# END_REL is used to mark the end of the path, and can take any value
END_REL = "END"

dataset = cfg.dataset
device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))


@torch.no_grad()
def get_texts_embeddings(texts, tokenizer, model):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=cfg.MAX_LEN,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    embeddings = model(**inputs, return_dict=True).pooler_output
    return embeddings


@torch.no_grad()
def judge_label(path, word_list, MAX_HOP, tokenizer, retriever, discriminator):
    if not word_list or not path or path == [END_REL]:
        return -1
    if path[-1] == END_REL:
        rel_list = path[:-1]
    else:
        rel_list = path.copy()
    mask_emb = torch.ones(MAX_HOP).to(device)
    if len(rel_list) < MAX_HOP:
        # Complete the path to MAX_HOP and mask the completed parts
        mask_num = MAX_HOP - len(rel_list)
        mask_emb[-mask_num:] = 0
        rel_list += [""] * mask_num
    word_emb = get_texts_embeddings(word_list, tokenizer, retriever).unsqueeze(0)
    rel_emb = get_texts_embeddings(rel_list, tokenizer, retriever).unsqueeze(0)
    mask_emb = mask_emb.unsqueeze(0)

    logit = discriminator(word_emb, rel_emb, mask_emb)
    label = torch.argmax(logit, dim=-1)
    return label.item()


@torch.no_grad()
def search_candidate_path(
    kg,
    candidate_paths,
    nums,
    max_hop,
    notional_word,
    terminate_prob,
    tokenizer,
    retriever,
    discriminator,
):
    rel_score_list = []
    candidate_paths_new = []

    for index in range(len(candidate_paths)):
        path = candidate_paths[index]["path"]
        if path and path[-1] == END_REL or len(path) >= max_hop:
            continue
        current_score = candidate_paths[index]["score"]
        question = candidate_paths[index]["question"]
        rels = candidate_paths[index]["rels"]
        src = candidate_paths[index]["src"]
        heads = candidate_paths[index]["heads"]

        if path and not rels:
            candidate_paths_new.append(
                {
                    "path": path + [END_REL],
                    "src": src,
                    "heads": heads,
                    "score": current_score,
                }
            )
            continue

        r_emb = get_texts_embeddings(rels, tokenizer, retriever)
        q_emb = get_texts_embeddings([question], tokenizer, retriever).expand_as(r_emb)
        # new_scores = F.cosine_similarity(q_emb, r_emb, dim=-1) * current_score
        scores = F.cosine_similarity(q_emb, r_emb, dim=-1)

        if path:
            label = judge_label(
                path,
                notional_word,
                max_hop,
                tokenizer,
                retriever,
                discriminator,
            )
            if not (scores > terminate_prob).any() or (
                not (scores > terminate_prob * 2).any() and label == 2
            ):
                candidate_paths_new.append(
                    {
                        "path": path + [END_REL],
                        "src": src,
                        "heads": heads,
                        "score": current_score,
                    }
                )
                continue

        new_scores = scores * current_score

        rel_score_list.extend(
            [[index, rel, new_score.item()] for rel, new_score in zip(rels, new_scores)]
        )

    topn = heapq.nlargest(nums, rel_score_list, key=lambda x: x[2])

    for index, rel, new_score in topn:
        if rel != END_REL:
            question_new = " ".join([candidate_paths[index]["question"], rel, "#"])
            heads_new = set.union(
                *[kg.get_tail(head, rel) for head in candidate_paths[index]["heads"]]
            )
            rels_new = set.union(*[kg.get_relation(head) for head in heads_new])
            rels_new = list(rels_new)

            path_new = candidate_paths[index]["path"] + [rel]
            candidate_paths_new.append(
                {
                    "question": question_new,
                    "rels": rels_new,
                    "path": path_new,
                    "src": candidate_paths[index]["src"],
                    "heads": heads_new,
                    "score": new_score,
                }
            )
        else:
            candidate_paths_new.append(
                {
                    "path": candidate_paths[index]["path"] + [rel],
                    "src": candidate_paths[index]["src"],
                    "heads": candidate_paths[index]["heads"],
                    "score": new_score,
                }
            )
    candidate_paths_new = sorted(
        candidate_paths_new, key=lambda x: x["score"], reverse=True
    )
    return candidate_paths_new[:nums]


def infer_paths(
    qa, kg, MAX_HOP, terminate_prob, tokenizer, retriever, discriminator, TOP_K
):
    entities = qa["topic_entities"]
    entity_names = qa["topic_entity_names"]
    if not entities:
        raise ValueError("No Topic Entity")

    path_score_list = []
    candidate_paths = []

    for entity, entity_name in zip(entities, entity_names):
        rels = list(kg.get_relation(entity))
        if not rels:
            continue

        question = " ".join([qa["question"], "[SEP]", entity_name, "→"])
        question = question.replace("[SEP]", tokenizer.sep_token)

        candidate_paths.append(
            {
                "question": question,
                "rels": rels,
                "path": [],
                "src": [entity, entity_name],
                "heads": {entity},
                "score": 1,
            }
        )
    nums = TOP_K
    counter = 0

    while counter < MAX_HOP and nums:
        candidate_paths = search_candidate_path(
            kg,
            candidate_paths,
            nums,
            MAX_HOP,
            qa["notional_word"],
            terminate_prob,
            tokenizer,
            retriever,
            discriminator,
        )
        for candidate in candidate_paths:
            if candidate["path"][-1] == END_REL or len(candidate["path"]) >= MAX_HOP:
                nums -= 1

                if candidate["path"][-1] == END_REL:
                    path_final = candidate["path"][:-1]
                else:
                    path_final = candidate["path"]

                path_score_list.append(
                    {
                        "path": path_final,
                        "src": candidate["src"],
                        "tails": list(candidate["heads"]),
                        "score": candidate["score"],
                    }
                )

        counter += 1

    # path_score_list = heapq.nlargest(TOP_K, path_score_list, key=lambda x: x["score"])

    qa.update({"path_with_score": path_score_list})
    qa.pop("notional_word", None)
    return qa


def retrieve_path_with_discriminator(TOP_K=1):
    if dataset == "webqsp":
        MAX_HOP = 2
        terminate_prob = 0.4
    elif dataset == "cwq":
        MAX_HOP = 4
        terminate_prob = 0.35
    else:
        raise NotImplementedError

    input_path = cfg.inference["input_path"]
    output_dir = cfg.inference["output_dir"]
    output_path = os.path.join(output_dir, f"K={TOP_K}.json")

    if os.path.exists(output_path):
        return

    retriever_path = cfg.retriever["final_model"]
    discriminator_path = cfg.discriminator["final_model"]

    print("[load model begin]")

    kg = KnowledgeGraphCache()

    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model["roberta_base"])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    retriever = AutoModel.from_pretrained(retriever_path)
    discriminator = Discriminator(retriever.config.hidden_size)
    state_dict = torch.load(discriminator_path)
    discriminator.load_state_dict(state_dict)
    for param in retriever.parameters():
        param.requires_grad = False
    for param in discriminator.parameters():
        param.requires_grad = False

    retriever.eval()
    discriminator.eval()
    retriever.to(device)
    discriminator.to(device)

    print("[load model end]")

    with open(input_path, "r") as f:
        test = [json.loads(line) for line in f]

    with open(output_path, "w") as f:
        for qa in tqdm(test, desc="retrieve"):
            qa_path = infer_paths(
                qa,
                kg,
                MAX_HOP,
                terminate_prob,
                tokenizer,
                retriever,
                discriminator,
                TOP_K,
            )
            f.write(json.dumps(qa_path) + "\n")
