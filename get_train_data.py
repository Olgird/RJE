from func import *
from config import cfg

retrive_path()

qa_paths_list = []


with open("data/results/"+cfg.dataset+"/K=10.json","r") as f:

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
for qa_paths in tqdm(qa_paths_list):
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
                if entity in ans_set:
                    flag_cover = True
                    break

                
    qa_paths['entities_rels_list'] = entities_rels_list
    if not flag_cover:
        print(qa_paths['question'])
        print(entities_rels_list)
        print(qa_paths["path_with_score"])
        uncover += 1



write_json(qa_paths_list, "qa_paths_list_top5_" + cfg.dataset + "_train.json")


