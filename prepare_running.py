import json

from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from tqdm import trange

from config import cfg
from func import *
import argparse

retrive_path(test=True)

def read_json(file_path):

    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
       
        return data
    except Exception as e:
        print(f" {e}")
        return None
    


qa_paths_list = []


with open("data/results/"+cfg.dataset+"/test/K=10.json","r") as f:

    for line in f:

        qa_paths_list.append(json.loads(line)) 


import json
from typing import Any

def write_json(data: Any, file_path: str, indent: int = 4, ensure_ascii: bool = False, sort_keys: bool = True) -> None:


    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(
            data,
            f,
            indent=indent,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            separators=(',', ': ') 
        )




uncover = 0

id_name_dict = {}
name_id_dict = {}

for qa_paths in tqdm(qa_paths_list):
    
    for topic_entity,topic_entity_name in zip(qa_paths["topic_entities"],qa_paths["topic_entity_names"]):
        id_name_dict[topic_entity] = topic_entity_name
        name_id_dict[topic_entity_name] = topic_entity


    ans_set = set()
    for answer in qa_paths["answers"]:
        ans_set.add(answer["kb_id"])

    entities_rels_list = []
    flag_cover = False
    for path_with_score in qa_paths["path_with_score"][:5]:
        
        entities_rels_list.append([])

        topic_entity_id = path_with_score['src'][0]
        path = path_with_score['path']
        
        try:
            entities_list = path_2_entities_list(topic_entity_id,path)
        except:
            continue

        for entities in entities_list:
            entities_rels_list[-1].append([])
            entities_rels_list[-1][-1].append(path_with_score['src'][1])
            
            
            for i in range(len(entities)):
                entities_rels_list[-1][-1].append(path[i])
                entity = entities[i]
                

                entity_name = id2entity_name_or_type(entity)
                entities_rels_list[-1][-1].append(entity_name)
                
                id_name_dict[entity] = entity_name
                name_id_dict[entity_name] = entity
                
                if entity in ans_set:
                    flag_cover = True
                    break

                
    qa_paths['entities_rels_list'] = entities_rels_list
    if not flag_cover:
        print(qa_paths['question'])
        print(entities_rels_list)
        print(qa_paths["path_with_score"])
        uncover += 1



write_json(qa_paths_list, "qa_paths_list_top5_" + cfg.dataset + "_test.json")

write_json(id_name_dict, "data/id_name_dict_" + cfg.dataset + ".json")
write_json(name_id_dict, "data/name_id_dict_" + cfg.dataset + ".json")


class SingleTowerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(cfg.pretrained_model["roberta_base"])
        self.classifier = torch.nn.Linear(768, 1) 

    def forward(self, input_ids, attention_mask):
        
        cls_embedding =  self.model(input_ids, attention_mask=attention_mask, return_dict=True).pooler_output

        logits = self.classifier(cls_embedding)
        return logits.squeeze()

MAX_LEN = 128
@torch.no_grad()
def get_scores(texts, tokenizer, model):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}


    scores = model(**inputs)
    return scores





device = 'cuda'

path_tokenizer = cfg.pretrained_model["roberta_base"]
ranker_path = "webqsp_ranker/final_model/single_tower_roberta.pt"
qa_paths_list_top5 = read_json("qa_paths_list_top5_" + cfg.dataset + "_test.json")

tokenizer = AutoTokenizer.from_pretrained(path_tokenizer)


ranker = SingleTowerModel()  
ranker.load_state_dict(torch.load(ranker_path))

ranker.to(device)

MAX_LEN = 128


entity_paths_with_score_list = []

for qa_paths_list in tqdm(qa_paths_list_top5):

    entity_paths = []

    for entities_rels in qa_paths_list['entities_rels_list']:

        for entity_rel in entities_rels:

            entity_paths.append(entity_rel)


    entity_paths_string = []

    for entity_path in entity_paths:

        entity_paths_string.append(qa_paths_list['question'] + tokenizer.sep_token +" → ".join(entity_path))

    if len(entity_paths_string) == 0:
        entity_paths_with_score_list.append([])
        continue
        
    batch_size = 1024
    score_list = []
    for i in range(0,len(entity_paths_string),batch_size):

        entity_paths_string_batch = entity_paths_string[i:i+batch_size]
        
        scores = get_scores(entity_paths_string_batch, tokenizer, ranker)
        
        if scores.dim() == 0:
            
            score_list.append(scores.item())
        else:
            for score in scores:
                score_list.append(score.item())


        

    path_score_list = []
    for entity_path,score in zip(entity_paths,score_list):

        path_score_list.append({"path":entity_path,"score":score})


    path_score_list = sorted(
        path_score_list, key=lambda x: x["score"], reverse=True
    )



    entity_paths_with_score_list.append(path_score_list)


import json
from typing import Any



q_entity_path_dict = dict()

for i in range(len(qa_paths_list_top5)):

    qa_paths = qa_paths_list_top5[i]
    path_list = []
    question = qa_paths["question"]
    for p_s in entity_paths_with_score_list[i]:
        path_list.append(p_s["path"])

    q_entity_path_dict.update({question: path_list})


write_json(q_entity_path_dict,"data/q_entity_path_dict_" + cfg.dataset + ".json")