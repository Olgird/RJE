from tqdm import tqdm
import argparse
from utils import *
from freebase_func import *
import pprint

def repeat_unanswer(dataset, datas, question_string, model_name):
    answered_set = set()
    new_data = []

    file_path = 'RJE_'+dataset+'_'+model_name+'.jsonl'
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line) 
            answered_set.add(data[question_string])

    for x in datas:
        if x[question_string] not in answered_set:
            new_data.append(x)
    print(len(new_data))

    return new_data

def get_one_data(datas, question_string, question):
    for data in datas:
        if data[question_string] == question:
            return [data]
        
def remove_key_value(input_dict):
    keys_to_remove = []
    for key, value in input_dict.items():
        if isinstance(value, str) and "male" in value.lower():
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del input_dict[key]
    return input_dict       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="webqsp", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=2000, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.0, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0.0, help="the temperature in reasoning stage.")
    parser.add_argument("--depth", type=int,
                        default=2, help="choose the search rounds of RJE.")
    parser.add_argument("--select_num", type=int,
                        default=30, help="Top-N relations")
    parser.add_argument("--path_num", type=int,
                        default=10, help="Top-K paths")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str,
                        default="deepseek", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")

    args = parser.parse_args()
    datas, question_string, question_retriever, question_path = prepare_dataset(args.dataset, args.path_num)
    datas = repeat_unanswer(args.dataset, datas, question_string, args.LLM_type)

    model = SentenceTransformer('model/msmarco-distilbert-base-tas-b')


    print("Start Running RJE on %s dataset." % args.dataset)

    if "cwq" in args.dataset:
        with open("data/id_name_dict_cwq.json", "r", encoding="utf-8") as f:
            entid_name = json.load(f)
        with open("data/name_id_dict_cwq.json", "r", encoding="utf-8") as f:
            name_entid = json.load(f)
    elif "webqsp" in args.dataset:
        with open("data/id_name_dict_webqsp.json", "r", encoding="utf-8") as f:
            entid_name = json.load(f)
        with open("data/name_id_dict_webqsp.json", "r", encoding="utf-8") as f:
            name_entid = json.load(f)
    else:
        print("dataset name error")
        exit()


    for data in tqdm(datas):
        try:
            start_time = time.time()
            call_num = 0
            all_t = {'total': 0, 'input': 0, 'output': 0}

            question = data[question_string]
            question_ret = data[question_retriever]

            path = question_path[question_ret]
            # print('New question start:', question)

            topic_entity = data['topic_entity']
            first_topic_entity = data['topic_entity']
            topic_entity = remove_key_value(topic_entity)
            first_topic_entity = remove_key_value(first_topic_entity)

            entity_pre_rel_list = {} 
            entity_pre_topic = {}  
            entity_pre_entity = {}  
            topic_entity_question = {} 
            topic_entity_path = {} 
            total_entity_set = [] 
            visit_entity = []
            dep_topic_entity_path = {}

            for e_id, e_name in topic_entity.items():
                total_entity_set.append(e_name)
                entity_pre_topic[e_id] = e_name
                entity_pre_entity[e_id] = e_id
                entity_pre_rel_list[e_id] = []
                entid_name[e_id] = e_name
                name_entid[e_name] = e_id

            for p in path:
                topic_id = name_entid[p[0]]
                if topic_id not in topic_entity.keys():
                    topic_entity[topic_id] = str(p[0])
                    first_topic_entity[topic_id] = str(p[0])

            for p in path:
                rel_list = []
                topic = p[0]
                for i, element in enumerate(p):
                    ##entity
                    if i % 2 == 0:
                        total_entity_set.append(element)
                        entity_pre_rel_list[name_entid[element]] = list(rel_list)
                        entity_pre_topic[name_entid[element]] = topic
                    ##rel
                    else:
                        rel_list.append(element)


            response, token_num = first_answer(question, path, args)
            last_brace_l = response.rfind('{')
            last_brace_r = response.rfind('}')
            response = response[last_brace_l:last_brace_r+1]
            try:
                response_dict = json.loads(response)
            except:
                try:
                    response_dict = ast.literal_eval(response)
                except:
                    response_dict = {"Sufficient": "No"}

            if response_dict['Sufficient'] == "Yes":
                # print("========Judgment_Done!======")
                save_2_jsonl(question, question_string, response, [], 1, token_num, start_time, "first", file_name=args.dataset+'_'+args.LLM_type)
                continue

            if len(topic_entity) == 0:
                call_num += 1
                results, token_num = generate_without_explored_paths(question, args)
                for kk in token_num.keys():
                    all_t[kk] += token_num[kk]

                save_2_jsonl(question, question_string, results, [], call_num, all_t, start_time, "second", file_name=args.dataset+'_'+args.LLM_type)
                continue
            elif len(topic_entity) > 1:
                response, token_num = split_question(question, topic_entity, args)
                last_brace_l = response.rfind('{')
                last_brace_r = response.rfind('}')
                response = response[last_brace_l:last_brace_r+1]
                # print(response)
                try:
                    response = json.loads(response)
                except:
                    try:
                        response = ast.literal_eval(response)
                    except:
                        call_num += 1
                        results, token_num = half_stop(question, topic_entity_question, topic_entity_path, entid_name, name_entid, args)
                        for kk in token_num.keys():
                            all_t[kk] += token_num[kk]

                        save_2_jsonl(question, question_string, results, cluster_chain_of_entities, call_num, all_t, start_time, "second", file_name=args.dataset+'_'+args.LLM_type)
                        flag_printed = True
                        continue
                                           
                for e_id, e_name in topic_entity.items():
                    topic_entity_question[e_name] = question
                    topic_entity_path[e_id] = []
                    dep_topic_entity_path[e_id] = []
                    if e_name in response.keys():
                        topic_entity_question[e_name] = response[e_name]

                call_num += 1
                for kk in token_num.keys():
                    all_t[kk] += token_num[kk]
            else:
                for e_id, e_name in topic_entity.items():               
                    topic_entity_question[e_name] = question
                    topic_entity_path[e_id] = []
                    dep_topic_entity_path[e_id] = []
            
            cluster_chain_of_entities = []
            depth_ent_rel_ent_dict = {}

            flag_printed = False

            for p in path:
                topic_id = name_entid[p[0]]
                dep_topic_entity_path[topic_id].append(p)
                i = 0
                entity_pre_entity[topic_id] = topic_id
                while i < len(p)-1:
                    entity_pre_entity[name_entid[p[i+2]]] = name_entid[p[i]]
                    topic_entity_path[topic_id].append([p[i], p[i+1], p[i+2]])
                    i += 2

            topic_entity = {}
            for e_id, e_name in first_topic_entity.items():
                if len(topic_entity_path[e_id]) == 0:
                    topic_entity[e_id] = e_name
                    visit_entity.append(e_id)
            for depth in range(1, args.depth+1):      
                entity_id_list, entity_name_list, token_num = select_exploration_entity(question, topic_entity_question, dep_topic_entity_path, total_entity_set, name_entid, args)
                call_num += 1
                for kk in token_num.keys():
                    all_t[kk] += token_num[kk]
                for entity in entity_id_list:
                    if if_topic_non_retrieve(entity):
                        continue
                    if entity in visit_entity:
                        continue
                    if entity.startswith("m.") or entity.startswith("g."):
                        visit_entity.append(entity)
                        topic_entity[entity] = entid_name[entity]
                pre_relations = []
                pre_heads= [-1] * len(topic_entity)

                current_entity_relations_list = []
                i=0
                for entity in topic_entity:
                    if entity!="[FINISH_ID]":
                        call_num += 1
                        retrieve_relations, token_num = relation_condition_prune(question, entity, topic_entity[entity], pre_relations, pre_heads[i], topic_entity_question[entity_pre_topic[entity]], entity_pre_rel_list[entity], entity_pre_topic[entity], entity_pre_entity, args)
                        for kk in token_num.keys():
                            all_t[kk] += token_num[kk]
                        current_entity_relations_list.extend(retrieve_relations)
                    i+=1
                total_candidates = []
                total_relations = []
                total_entities_id = [] 
                total_topic_entities = [] 
                total_head = []
                ent_rel_ent_dict = {} # e->head/tail->rel->ent
                for ent_rel in current_entity_relations_list:
                    current_entity_id = ent_rel['entity']
                    current_rel_list = entity_pre_rel_list[current_entity_id]
                    now_rel_list = current_rel_list + [ent_rel['relation']]

                    if ent_rel['entity'] not in ent_rel_ent_dict.keys():
                        ent_rel_ent_dict[ent_rel['entity']] = {}

                    if ent_rel['head']:
                        head_or_tail = 'head'
                        entity_candidates_id = entity_search(ent_rel['entity'], ent_rel['relation'], True)
                    else:
                        head_or_tail = 'tail'
                        entity_candidates_id = entity_search(ent_rel['entity'], ent_rel['relation'], False)
                    
                    if len(entity_candidates_id) == 0:
                        print('the relations without tail entity:', ent_rel)
                        continue

                    entity_candidates, entity_candidates_id = provide_triple(entity_candidates_id)

                    for entity_id in entity_candidates_id:
                        entity_pre_entity[entity_id] = str(current_entity_id)
                        entity_pre_rel_list[entity_id] = now_rel_list
                        entity_pre_topic[entity_id] = entity_pre_topic[current_entity_id]

                        
                    name_entid.update(dict(zip(entity_candidates, entity_candidates_id)))
                    entid_name.update(dict(zip(entity_candidates_id, entity_candidates)))
                    for e_id, e_name in first_topic_entity.items():
                        name_entid[e_name] = e_id

                    if head_or_tail not in ent_rel_ent_dict[ent_rel['entity']].keys():
                            ent_rel_ent_dict[ent_rel['entity']][head_or_tail] = {}

                    if ent_rel['relation'] not in ent_rel_ent_dict[ent_rel['entity']][head_or_tail].keys():
                        ent_rel_ent_dict[ent_rel['entity']][head_or_tail][ent_rel['relation']] = []

                    # store current entities into ent_rel_ent_dict
                    for retrive_ent in entity_candidates_id:
                        if retrive_ent not in ent_rel_ent_dict[ent_rel['entity']][head_or_tail][ent_rel['relation']]:
                            ent_rel_ent_dict[ent_rel['entity']][head_or_tail][ent_rel['relation']].append(retrive_ent)
                    
                    total_candidates, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, ent_rel, entity_candidates_id, total_candidates, total_relations, total_entities_id, total_topic_entities, total_head)
                
                depth_ent_rel_ent_dict[depth] = ent_rel_ent_dict
                # pprint.pprint(convert_dict_name(ent_rel_ent_dict, entid_name))

                if len(total_candidates) == 0:

                    call_num += 1
                    results, token_num = half_stop(question, topic_entity_question, topic_entity_path, entid_name, name_entid, args)
                    for kk in token_num.keys():
                        all_t[kk] += token_num[kk]

                    save_2_jsonl(question, question_string, results, cluster_chain_of_entities, call_num, all_t, start_time, "second", file_name=args.dataset+'_'+args.LLM_type)
                    flag_printed = True
                    break
                dep_topic_entity_path = {}
                total_entity_set = []
                flag, chain_of_entities, entities_id, pre_relations, pre_heads, new_ent_rel_ent_dict, cur_call_time, cur_token = entity_condition_prune(topic_entity_question, ent_rel_ent_dict, entid_name, name_entid, topic_entity_path, entity_pre_topic, total_entity_set, dep_topic_entity_path, visit_entity, args, model)

                cluster_chain_of_entities.append(chain_of_entities)

                call_num += cur_call_time
                for kk in cur_token.keys():
                    all_t[kk] += cur_token[kk]

                # pprint.pprint(convert_dict_name(new_ent_rel_ent_dict, entid_name))
                if flag:
                    call_num += 1
                    response, sufficient, answer, reason, token_num = reasoning(question, topic_entity_question, topic_entity_path, entid_name, name_entid, args)
                    for kk in token_num.keys():
                        all_t[kk] += token_num[kk]

                    if sufficient == 'No' or str(answer).startswith('m.') or str(answer).startswith('[\"m.') or str(answer).startswith("['m.") or str(answer).startswith('g.') or str(answer).startswith("['g.") or str(answer).startswith('[\"g.') or 'yes' not in sufficient.lower():
                        stop = False
                    else:
                        stop = True

                    if stop:
                        print("RJE stoped at depth %d." % depth)
                        save_2_jsonl(question, question_string, response, cluster_chain_of_entities, call_num, all_t, start_time, "second", file_name=args.dataset+'_'+args.LLM_type)
                        flag_printed = True
                        break
                    else:
                        print("depth %d still not find the answer." % depth)
                        topic_entity = {}


                else:
                    call_num += 1
                    results, token_num = half_stop(question, topic_entity_question, topic_entity_path, entid_name, name_entid, args)
                    for kk in token_num.keys():
                        all_t[kk] += token_num[kk]

                    save_2_jsonl(question, question_string, results, cluster_chain_of_entities, call_num, all_t, start_time, "second", file_name=args.dataset+'_'+args.LLM_type)
                    flag_printed = True
                    break
            
            if not flag_printed:
                call_num += 1
            
                results, token_num = half_stop(question, topic_entity_question, topic_entity_path, entid_name, name_entid, args)
                for kk in token_num.keys():
                    all_t[kk] += token_num[kk]

                save_2_jsonl(question, question_string, results, cluster_chain_of_entities, call_num, all_t, start_time, "second", file_name=args.dataset+'_'+args.LLM_type)
        except:
            continue