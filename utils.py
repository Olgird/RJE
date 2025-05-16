from prompt_list import *
import json
import time
from openai import OpenAI
import re
import requests
import random
from prompt_list import *
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import os

color_yellow = "\033[93m"
color_green = "\033[92m"
color_red= "\033[91m"
color_end = "\033[0m"

def retrieve_top_docs(query, docs, model, width=3):
    query_emb = model.encode(query)
    doc_emb = model.encode(docs)
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)
    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]
    return top_docs, top_scores

def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="qwen", print_in=True, print_out=True):
    if print_in:
        print(color_green+prompt+color_end)

    openai_api_key = "EMPTY"
    openai_api_base = "your local url when use vLLM"

    model_path = "None"
    if 'llama_8b' in engine:
        model_path = "Llama-3.1-8B-Instruct"
    elif 'qwen_14b' in engine:
        model_path = "qwen2.5-14B-Instruct"
    elif 'qwen_72b' in engine:
        model_path = "Qwen2.5-72B-Instruct"
    elif 'deepseek' in engine:
        model_path = "deepseek-v3-241226"
        openai_api_key = "your api key"
        openai_api_base = ""

    elif 'gpt35' in engine:
        # model_path = "gpt-3.5-turbo-0125"
        model_path = "gpt-3.5-turbo"
        openai_api_key = "your api key"
        openai_api_base = ""

    elif 'gpt4omini' in engine:
        model_path = "gpt-4o-mini"
        openai_api_key = "your api key"
        openai_api_base = ""
    

    messages = [{"role":"system","content":"You are an intelligent assistant for a Knowledge Graph Question Answering (KGQA) system."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    completion = client.chat.completions.create(
            model=model_path,
            messages = messages,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=0,
            presence_penalty=0,
            )

    result = completion.choices[0].message.content

    token_num = {"total": completion.usage.total_tokens, "input": completion.usage.prompt_tokens, "output": completion.usage.completion_tokens}

    if print_out:
        print(color_yellow + result + color_end)
    return result, token_num


def convert_dict_name(ent_rel_ent_dict, entid_name):
    name_dict = {}
    for topic_e, h_t_dict in ent_rel_ent_dict.items():
        if entid_name[topic_e] not in name_dict.keys():
            name_dict[entid_name[topic_e]] = {}

        for h_t, r_e_dict in h_t_dict.items():
            if h_t not in name_dict[entid_name[topic_e]].keys():
                name_dict[entid_name[topic_e]][h_t] = {}
            
            for rela, e_list in r_e_dict.items():
                if rela not in name_dict[entid_name[topic_e]][h_t].keys():
                    name_dict[entid_name[topic_e]][h_t][rela] = []
                for ent in e_list:
                    if entid_name[ent] not in name_dict[entid_name[topic_e]][h_t][rela]:
                        name_dict[entid_name[topic_e]][h_t][rela].append(entid_name[ent])
    return name_dict

    

def save_2_jsonl(question, question_string, answer, cluster_chain_of_entities, call_num, all_t, start_time, flag, file_name):
    tt = time.time()-start_time
    dict = {question_string:question, "results": answer, "reasoning_chains": cluster_chain_of_entities, "call_num": call_num, "total_token": all_t['total'], "input_token": all_t['input'], "output_token": all_t['output'], "time": tt, "flag": flag}
    with open("RJE_{}.jsonl".format(file_name), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")
    


def generate_without_explored_paths(question, args):
    prompt = cot_prompt + question 

    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False)
    last_brace_l = response.rfind('{')
    last_brace_r = response.rfind('}')
    response = response[last_brace_l:last_brace_r+1]
    return response, token_num

    
def prepare_dataset(dataset_name, path_num):
    if  'cwq' in dataset_name:
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_path_new = {}
        with open("../data/q_entity_path_dict_cwq.json", "r", encoding="utf-8") as f:
            question_path = json.load(f)
        for question, path_list in question_path.items():
            question_path_new[question] = path_list[:path_num]
        question_string = 'question'
        question_retriever = 'question'
    elif 'webqsp' in dataset_name:
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_path_new = {}
        with open("../data/q_entity_path_dict_webqsp.json", "r", encoding="utf-8") as f:
            question_path = json.load(f)
        for question, path_list in question_path.items():
            question_path_new[question] = path_list[:path_num]
        question_string = 'RawQuestion'
        question_retriever = 'ProcessedQuestion'
    else:
        print("dataset not found, you should pick from {cwq, webqsp}.")
        exit(-1)
    return datas, question_string, question_retriever, question_path_new

