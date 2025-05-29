from SPARQLWrapper import SPARQLWrapper, JSON
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

import heapq
from tqdm import tqdm

from config import cfg
import os
import json
from multiprocessing import Pool
from openai import OpenAI





END_REL = "END"
device = 'cuda'
SPARQLPATH = "http://localhost:8899/sparql"

abc = 'abcdefghijklmnopqrstuvwxyz'
sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""


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

def execurte_sparql(sparql_query):
    
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]

def replace_relation_prefix(relations):


    new_relations = []
    for relation in relations:
        if "http://rdf.freebase.com/ns/" in relation['relation']['value']:
            new_relations.append(relation['relation']['value'])
        

    relations = new_relations

    return [relation.replace("http://rdf.freebase.com/ns/","") for relation in relations]



    
def id2entity_name_or_type(entity_id):
    try:
        sparql_query = sparql_id % (entity_id, entity_id)
        sparql = SPARQLWrapper(SPARQLPATH)
        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        if len(results["results"]["bindings"])==0:
            return entity_id
        else:
            return results["results"]["bindings"][0]['tailEntity']['value']
        
    except:

        return entity_id
    

sparql_triples = '''  %s %s %s .\n'''
sparql_not_equal_filter ='''FILTER (%s != ns:%s)''' 

sparql_h_r = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
WHERE {\n %s \n}
LIMIT 200"""

def head_relations_sparql(path,topic_entity_id):

    Q = ''

    abc_index = 0
    now_entity = "ns:"+topic_entity_id


    for i in range(len(path)):

        Q += sparql_triples %(now_entity,"ns:" + path[i],"?"+abc[abc_index])
        now_entity = "?"+abc[abc_index]
        abc_index += 1

        

    Q += sparql_triples %(now_entity,"?relation","?"+abc[abc_index])


    return sparql_h_r % Q

def get_relations(path,topic_entity_id):
    q = head_relations_sparql(path,topic_entity_id)

    return replace_relation_prefix(execurte_sparql(q))


def retrive_path(TOP_K = 10,beam_nums = 10,test = False):    
    if cfg.dataset == "webqsp":
        MAX_HOP = 2
        terminate_prob = 0.4
    elif cfg.dataset == "cwq":
        MAX_HOP = 4
        terminate_prob = 0.35
    else:
        raise NotImplementedError
    
    if test == False:
        input_path = cfg.get_train_data["input_path"]
        output_dir = cfg.get_train_data["output_dir"]
        output_path = os.path.join(output_dir, f"K={TOP_K}.json")

    else:

        input_path = cfg.prepare_running["input_path"]
        output_dir = cfg.prepare_running["output_dir"]
        output_path = os.path.join(output_dir, f"K={TOP_K}.json")

    if os.path.exists(output_path):
        print(f"[get_train_data] {output_path} exists")
        return
    retriever_path = cfg.retriever["final_model"]


    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model["roberta_base"])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    retriever = AutoModel.from_pretrained(retriever_path)
    for param in retriever.parameters():
        param.requires_grad = False
    retriever.eval()
    retriever.to(device)

    with open(input_path, "r") as f:
        test = [json.loads(line) for line in f]

    with open(output_path, "w") as f:
        for qa in tqdm(test, desc="retrieve"):
            try: 
                qa_path = infer_paths(
                    qa = qa,
                    MAX_HOP = MAX_HOP,
                    beam_nums = beam_nums,
                    terminate_prob = terminate_prob,
                    tokenizer = tokenizer,
                    retriever = retriever,
                    TOP_K = TOP_K)
                
                f.write(json.dumps(qa_path) + "\n")
            except Exception as e:
                print(e)
                print(qa)
                


def infer_paths(qa,
                MAX_HOP,
                beam_nums,
                TOP_K,
                terminate_prob, 
                tokenizer, retriever):
    entities = qa["topic_entities"]
    entity_names = qa["topic_entity_names"]

    if not entities:
       
        raise ValueError("No Topic Entity")
    
    path_score_list = []

    candidate_paths = []
    for entity, entity_name in zip(entities, entity_names):
        if entity_names == None:
            continue

        question = " ".join([qa["question"], "[SEP]", entity_name, "→"])
        question = question.replace("[SEP]", tokenizer.sep_token)

        candidate_paths.append(
            {
                "question": question,
                "path": [],
                "src": [entity, entity_name],
                "score": 1,
            }
        )

    counter = 0
    nums = TOP_K
    while counter < MAX_HOP and nums >=0 :
        candidate_paths = search_candidate_path(candidate_paths,
                                                max_hop = MAX_HOP,
                                                beam_nums = beam_nums,
                                                terminate_prob = terminate_prob,
                                                tokenizer = tokenizer,
                                                retriever = retriever)
        
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
                        "score": candidate["score"]
                    }
                )

        counter += 1

    qa.update({"path_with_score": path_score_list})

    return qa


@torch.no_grad()
def search_candidate_path(candidate_paths,
                          max_hop,
                          beam_nums,
                          terminate_prob,
                          tokenizer,
                          retriever,):
    '''
    beam search,new candidate path
    '''
    rel_score_list = []
    candidate_paths_new = []

    for index in range(len(candidate_paths)):
        path = candidate_paths[index]["path"]
        if path and path[-1] == END_REL or len(path) >= max_hop:
            continue
        
        current_score = candidate_paths[index]["score"]
        question = candidate_paths[index]["question"]
        src = candidate_paths[index]["src"]

        if  len(path) > 0 and path[-1] == END_REL:
            continue


        try:
            rels = get_relations(path,src[0])
        except Exception as e:
            print(e)
            rels = []
        
        if not rels:
            candidate_paths_new.append(
                    {
                        "path": path + [END_REL],
                        "src": src,
                        "score": current_score,
                    }
                )
            continue
            
        
        r_emb = get_texts_embeddings(rels, tokenizer, retriever)

        q_emb = get_texts_embeddings([question], tokenizer, retriever).expand_as(r_emb)
        scores = F.cosine_similarity(q_emb, r_emb, dim=-1)


        if path:
            if not (scores > terminate_prob).any():

                candidate_paths_new.append(
                    {
                        "path": path + [END_REL],
                        "src": src,
                        "score": current_score,
                    }
                )
                continue

        new_scores = scores * current_score
        rel_score_list.extend(
            [[index, rel, new_score.item()] for rel, new_score in zip(rels, new_scores)]
        )

    topn = heapq.nlargest(beam_nums, rel_score_list, key=lambda x: x[2])

    for index, rel, new_score in topn:
        if rel != END_REL:
            question_new = " ".join([candidate_paths[index]["question"], rel, "#"])

            path_new = candidate_paths[index]["path"] + [rel]
            candidate_paths_new.append(
                {
                    "question": question_new,
                    "path": path_new,
                    "src": candidate_paths[index]["src"],
                    "score": new_score,
                }
            )
        
        else:
            candidate_paths_new.append(
                {
                    "path": candidate_paths[index]["path"] + [rel],
                    "src": candidate_paths[index]["src"],
                    "score": new_score,
                }
            )

    candidate_paths_new = sorted(
        candidate_paths_new, key=lambda x: x["score"], reverse=True
    )

    return candidate_paths_new[:beam_nums]


sparql_entity_list='''PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT %s
WHERE {
   %s
   %s
}'''
def path_2_entity_list_sparql(topic_entity_id,path):

    Q = ''
    abc_index = 0
    now_entity = "ns:"+topic_entity_id

    for i in range(len(path)):
        Q += sparql_triples %(now_entity,"ns:" + path[i],"?"+abc[abc_index])

        now_entity = "?"+abc[abc_index]
        abc_index += 1

    s_q = ''
    for i in range(abc_index):
        s_q += "?" + abc[i] + ","

    s_q = s_q[:-1]

    filter_str = ''
    for i in range(abc_index):
        filter_str += sparql_not_equal_filter % ("?" + abc[i],topic_entity_id)

    return sparql_entity_list % (s_q,Q,filter_str)

def path_2_entities_list(topic_entity_id,path):

    query = path_2_entity_list_sparql(topic_entity_id,path)

    an = execurte_sparql(query)

    entities_list = []

    for a in an:
        entities = []
        for abc_i in range(len(path)):
            key = abc[abc_i]

            entity = a[key]
            entity_id = entity['value'].replace("http://rdf.freebase.com/ns/","")
            entities.append(entity_id)

        entities_list.append(entities)


    return entities_list


