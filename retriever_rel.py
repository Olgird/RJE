import torch
from transformers import AutoModel, AutoTokenizer
from SPARQLWrapper import SPARQLWrapper, JSON
from config import cfg
import torch.nn.functional as F

device = 'cuda:0'
SPARQLPATH = "http://localhost:8899/sparql"

# pre-defined sparqls
sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}""" 
sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}"""
sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""

sparql_head_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n FILTER (?x != ns:%s)}"""

sparql_tail_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n FILTER (?x != ns:%s)}"""

def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation:
        return True

def execurte_sparql(sparql_query):
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    # print(results["results"]["bindings"])
    return results["results"]["bindings"]

def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations]

def replace_entities_prefix(entities):
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]

def id2entity_name_or_type(entity_id):
    sparql_query = sparql_id % (entity_id, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"])==0:
        return entity_id
    else:
        return results["results"]["bindings"][0]['tailEntity']['value']

retriever = AutoModel.from_pretrained("retriver_cwq")
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
retriever.eval()
retriever.to(device)

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

def retriever_select_rel(question, rel_list, pre_rel_path, topic_entity, select_num):
    question = " ".join([question, "[SEP]", topic_entity, "→"])
    question = question.replace("[SEP]", tokenizer.sep_token)
    question = " ".join([question] + [f"{rel} #" for rel in pre_rel_path])

    r_emb = get_texts_embeddings(rel_list, tokenizer, retriever)
    q_emb = get_texts_embeddings([question], tokenizer, retriever).expand_as(r_emb)
    scores = F.cosine_similarity(q_emb, r_emb, dim=-1)

    sorted_relations_scores = sorted(zip(rel_list, scores), key=lambda x: x[1], reverse=True)

    sorted_relations, sorted_scores = zip(*sorted_relations_scores)

    sorted_relations = list(sorted_relations)    

    if len(sorted_relations) > select_num:
        final_sorted_relations = sorted_relations[:select_num]
    else: 
        final_sorted_relations = sorted_relations
    return final_sorted_relations



