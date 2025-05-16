from SPARQLWrapper import SPARQLWrapper, JSON
from utils import *
from retriever_rel import *
import random
from freebase_func import *
from prompt_list import *
from gen_path import convert_paths_to_str
import json
import time
import openai
import re
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import ast

def construct_exploration_entity_prompt(question, topic_entity_question, dep_topic_entity_path, total_entity_set, name_entid):
    prompt = exploration_entity_prompt + question
    i = 1
    for topic_name, topic_question in topic_entity_question.items():
        prompt = prompt + '\n\nTopic ' + str(i) + ':\nTopic Question: '
        prompt = prompt +  topic_question + '\nTopic Enitity: ' + topic_name
        if name_entid[topic_name] not in dep_topic_entity_path.keys():
            dep_topic_entity_path[name_entid[topic_name]] = []
        path_str = convert_paths_to_str(dep_topic_entity_path[name_entid[topic_name]])
        prompt = prompt + '\nTriplets: ' + path_str
        i += 1
    prompt = prompt + '\n\nEntity List: ' + str(total_entity_set)
    return prompt


def construct_select_relation_prompt(question, entity_name, total_relations):
    return  select_relation_prompt + question  + '\nConnected entity: ' + entity_name + '\nRelations List: [' + ', '.join(total_relations) + ']'

def construct_split_question_prompt(question, topic_entity_list):
    return split_question_prompt + question + '\nTopic Entities: ' + str(topic_entity_list)


def split_question(question, topic_entity, args):
    topic_entity_list = []
    for e_id, e_name in topic_entity.items():
        topic_entity_list.append(e_name)
    prompt = construct_split_question_prompt(question, topic_entity_list)
    result, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False, False)
    return result, token_num


def select_exploration_entity(question, topic_entity_question, dep_topic_entity_path, total_entity_set, name_entid, args):

    total_entity_set = set(total_entity_set)
    total_entity_set = list(total_entity_set)
    new_entity = sorted(total_entity_set)
    prompt = construct_exploration_entity_prompt(question, topic_entity_question, dep_topic_entity_path, new_entity, name_entid)
    result, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False, False)

    cur_token = {'total': 0, 'input': 0, 'output': 0}
    for kk in token_num.keys():
        cur_token[kk] += token_num[kk]
    if result == None:
        result = "[]"
    last_brace_l = result.rfind('[')
    last_brace_r = result.rfind(']')
    
    if last_brace_l < last_brace_r:
        result = result[last_brace_l:last_brace_r+1]
    else:
        print("No entity found")
        return [], [], cur_token

    try:
        entity_list = ast.literal_eval(result)
    except:
        result = result.strip().strip("[").strip("]").split(', ')
        entity_list = [x.strip("'") for x in result]

    entity_name_list = [str(x) for x in entity_list if str(x) in new_entity]
    entity_id_list = [name_entid[name] for name in entity_name_list]

    return entity_id_list, entity_name_list, cur_token


def select_relations(string, entity_id, head_relations, tail_relations):
    try:
        last_brace_l = string.rfind('[')
        last_brace_r = string.rfind(']')
    except:
        string = "[]"
        last_brace_l = string.rfind('[')
        last_brace_r = string.rfind(']')
    
    if last_brace_l < last_brace_r:
        string = string[last_brace_l:last_brace_r+1]
    else:
        print("No relations found")
        return False, "No relations found"

    result = string
    relations=[]
    try:
        rel_list = ast.literal_eval(result)
    except:
        result = result.strip().strip("[").strip("]").split(', ')
        rel_list = [x.strip("'") for x in result]

    for relation in rel_list:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "head": True})
        elif relation in tail_relations:
            relations.append({"entity": entity_id, "relation": relation, "head": False})
    
    if not relations:
        print("No relations found")
        return False, "No relations found"
    return True, relations

def relation_condition_prune(ori_question, entity_id, entity_name, pre_relations, pre_head, question, pre_rel_list, topic_name, entity_pre_entity, args):
    sparql_relations_extract_head = sparql_head_relations % (entity_id, entity_pre_entity[entity_id])
    head_relations = execurte_sparql(sparql_relations_extract_head)
    head_relations = replace_relation_prefix(head_relations)
    
    sparql_relations_extract_tail= sparql_tail_relations % (entity_id, entity_pre_entity[entity_id])
    tail_relations = execurte_sparql(sparql_relations_extract_tail)
    tail_relations = replace_relation_prefix(tail_relations)

    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    
    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations + tail_relations

    if len(total_relations) == 0:
        return [], {"total": 0, "input": 0, "output": 0}
    
    final_total_rel_list = retriever_select_rel(ori_question, total_relations, pre_rel_list, topic_name, args.select_num)

    final_total_rel_list = sorted(final_total_rel_list)
    prompt = construct_select_relation_prompt(question, entity_name, final_total_rel_list)

    result, token_num = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type, False, False)

    flag, retrieve_relations = select_relations(result, entity_id, head_relations, tail_relations) 

    if flag:
        return retrieve_relations, token_num
    else:
        return [], token_num # format error or too small max_length
    
    
def entity_search(entity, relation, head=True):
    if head:
        tail_entities_extract = sparql_tail_entities_extract% (entity, relation)
        entities = execurte_sparql(tail_entities_extract)
    else:
        head_entities_extract = sparql_head_entities_extract% (relation, entity)
        entities = execurte_sparql(head_entities_extract)


    entity_ids = replace_entities_prefix(entities)
    return entity_ids


def provide_triple(entity_candidates_id):
    entity_candidates = []
    for entity_id in entity_candidates_id:
        if entity_id.startswith("m.") or entity_id.startswith("g."):
            entity_candidates.append(id2entity_name_or_type(entity_id))
        else:
            entity_candidates.append(entity_id)

    if len(entity_candidates) <= 1:
        return entity_candidates, entity_candidates_id


    ent_id_dict = dict(sorted(zip(entity_candidates, entity_candidates_id)))
    entity_candidates, entity_candidates_id = list(ent_id_dict.keys()), list(ent_id_dict.values())
    return entity_candidates, entity_candidates_id

    
def update_history(entity_candidates, ent_rel, entity_candidates_id, total_candidates, total_relations, total_entities_id, total_topic_entities, total_head):
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]

    candidates_relation = [ent_rel['relation']] * len(entity_candidates)
    topic_entities = [ent_rel['entity']] * len(entity_candidates)
    head_num = [ent_rel['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_relations, total_entities_id, total_topic_entities, total_head


def if_topic_non_retrieve(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
def is_all_digits(lst):
    for s in lst:
        if not s.isdigit():
            return False
    return True


def entity_condition_prune(topic_entity_question, ent_rel_ent_dict, entid_name, name_entid, topic_entity_path, entity_pre_topic, total_entity_set, dep_topic_entity_path, visit_entity, args, model):
    cur_call_time = 0
    cur_token = {'total': 0, 'input': 0, 'output': 0}
    new_ent_rel_ent_dict = {}
    no_prune = ['time', 'number', 'date']
    filter_entities_id, filter_tops, filter_relations, filter_candidates, filter_head = [], [], [], [], []
    for topic_e, h_t_dict in sorted(ent_rel_ent_dict.items()):
        for h_t, r_e_dict in sorted(h_t_dict.items()):
            for rela, e_list in sorted(r_e_dict.items()):
                if is_all_digits(e_list) or rela in no_prune or len(e_list) <= 1:
                    sorted_e_list = [entid_name[e_id] for e_id in sorted(e_list)]
                    select_ent = sorted_e_list
                elif all(entid_name[item].startswith('m.') or entid_name[item].startswith('g.')  for item in e_list):
                    if len(e_list) > 10:
                        e_list = random.sample(e_list, 10)
                    sorted_e_list = [entid_name[e_id] for e_id in sorted(e_list)]
                    select_ent = sorted_e_list
                else:
                    question = topic_entity_question[entity_pre_topic[topic_e]]
                    e_list = [e_n for e_n in e_list if '[' not in entid_name[e_n] and ']' not in entid_name[e_n] and '"' not in entid_name[e_n]]
                    if len(e_list) > 70:
                        sorted_e_list = [entid_name[e_id] for e_id in e_list]
                        topn_entities, topn_scores = retrieve_top_docs(question, sorted_e_list, model, 70)
                        e_list = [name_entid[e_n] for e_n in topn_entities]

                    prompt = select_entity_prompt + question
                    prompt += '\nTriples: '
                    sorted_e_list = [entid_name[e_id] for e_id in sorted(e_list)]
                    prompt += entid_name[topic_e] + ' → ' + rela + ' → ' + str(sorted_e_list)

                    cur_call_time += 1
                    result, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False, False)                    
                    for kk in token_num.keys():
                        cur_token[kk] += token_num[kk]
                    if  result == None:
                        result = "[]"
                    last_brace_l = result.rfind('[')
                    last_brace_r = result.rfind(']')
                    
                    if last_brace_l < last_brace_r:
                        result = result[last_brace_l:last_brace_r+1]
                    else:
                        print("No entity found")

                    try:
                        entity_list = ast.literal_eval(result)
                    except:
                        result = result.strip().strip("[").strip("]").split(', ')
                        entity_list = [x.strip("'") for x in result]

                    select_ent = [x for x in entity_list if x in sorted_e_list]

                if len(select_ent) == 0 or all(x == '' for x in select_ent):
                    continue

                if topic_e not in new_ent_rel_ent_dict.keys():
                    new_ent_rel_ent_dict[topic_e] = {}
                if h_t not in new_ent_rel_ent_dict[topic_e].keys():
                    new_ent_rel_ent_dict[topic_e][h_t] = {}
                if rela not in new_ent_rel_ent_dict[topic_e][h_t].keys():
                    new_ent_rel_ent_dict[topic_e][h_t][rela] = []
                
                for ent in select_ent:
                    if name_entid[entity_pre_topic[topic_e]] not in dep_topic_entity_path.keys():
                        dep_topic_entity_path[name_entid[entity_pre_topic[topic_e]]] = []
                    dep_topic_entity_path[name_entid[entity_pre_topic[topic_e]]].append([entid_name[topic_e], rela, ent])
                    if [entid_name[topic_e], rela, ent] not in topic_entity_path[name_entid[entity_pre_topic[topic_e]]]:
                        topic_entity_path[name_entid[entity_pre_topic[topic_e]]].append([entid_name[topic_e], rela, ent])
                    if name_entid[ent].startswith("m.") or name_entid[ent].startswith("g."):
                        if name_entid[ent] not in visit_entity:
                            total_entity_set.append(ent)

                    new_ent_rel_ent_dict[topic_e][h_t][rela].append(name_entid[ent])
                    filter_tops.append(entid_name[topic_e])
                    filter_relations.append(rela)
                    filter_candidates.append(ent)
                    filter_entities_id.append(name_entid[ent])
                    if h_t == 'head':
                        filter_head.append(True)
                    else:
                        filter_head.append(False)

    if len(filter_entities_id) == 0:
        return False, [], [], [], [], new_ent_rel_ent_dict, cur_call_time, cur_token

    cluster_chain_of_entities = [[(filter_tops[i], filter_relations[i], filter_candidates[i]) for i in range(len(filter_candidates))]]
    return True, cluster_chain_of_entities, filter_entities_id, filter_relations, filter_head, new_ent_rel_ent_dict, cur_call_time, cur_token


def reasoning(question, topic_entity_question, topic_entity_path, entid_name, name_entid, args):

    prompt = reason_prompt + question
    i = 1
    for topic_name, topic_question in topic_entity_question.items():
        prompt = prompt + '\n\nTopic ' + str(i) + ':\nTopic Question: '
        prompt = prompt +  topic_question + '\nTopic Enitity: ' + topic_name
        if name_entid[topic_name] not in topic_entity_path.keys():
            topic_entity_path[name_entid[topic_name]] = []        
        path_str = convert_paths_to_str(topic_entity_path[name_entid[topic_name]])
        prompt = prompt + '\nTriplets: ' + path_str
        i += 1

    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False)

    """
    {
        "Sufficient": "Yes",
        "Answer": "[Correct Answer]"
        "Reason": ""
    }
    {
        "Sufficient": "No",
        "Reason": ""
    }
    """
    last_brace_l = response.rfind('{')
    last_brace_r = response.rfind('}')
    response = response[last_brace_l:last_brace_r+1]
    try:
        response_dict = json.loads(response)
    except:
        response_dict = {"Sufficient": "No","Reason": ""}
    answer = ""
    reason = ""
    if 'Reason' in response_dict.keys():
        reason = response_dict['Reason']
    if response_dict['Sufficient'] == "Yes":
        answer = response_dict['Answer']
    return response, response_dict['Sufficient'], answer, reason, token_num

def first_answer(question, path, args):

    path_str = convert_paths_to_str(path)
    prompt = first_reason_prompt + question + '\nPaths: ' + path_str

    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False)

    return response, token_num

def half_stop(question, topic_entity_question, topic_entity_path, entid_name, name_entid, args):

    prompt = half_stop_prompt + question
    i = 1
    for topic_name, topic_question in topic_entity_question.items():
        prompt = prompt + '\n\nTopic ' + str(i) + ':\nTopic Question: '
        prompt = prompt +  topic_question + '\nTopic Enitity: ' + topic_name
        path_str = convert_paths_to_str(topic_entity_path[name_entid[topic_name]])
        prompt = prompt + '\nTriplets: ' + path_str
        i += 1

    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False)

    """
    {
        "Answer": "[Correct Answer]"
        "Reason": ""
    }

    """
    last_brace_l = response.rfind('{')
    last_brace_r = response.rfind('}')
    response = response[last_brace_l:last_brace_r+1]

    return response, token_num



