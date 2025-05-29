import argparse
import numpy as np
from utils import *
import re
import ast

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="webqsp", help="choose the dataset.")
    parser.add_argument("--output_file", type=str,
                        default="RJE_webqsp_gpt35", help="the output file name.")

    args = parser.parse_args()

    ground_truth_datas, question_string, output_datas = prepare_dataset_for_eval(args.dataset, args.output_file)

    count_q = {}
    right_q = {}
    re_list = []
    error_list = []
    first_right = 0
    first_error = 0
    second_right = 0
    second_error = 0
    ranker_right = []

    num_right = 0
    num_error = 0
    error_question = []
    q_list = []

    part_q = False
    aname_dict = {}
    alias_dict = {}
    add_ans_alias_dict = {}
    call_num_list = []
    time_list = []
    token_num_list = {
        "input": [],
        "output": [],
        "total": []
    }

    if args.dataset == 'cwq':

        with open('../cope_alias/cwq_aname_dict.json', 'r', encoding='utf-8') as f:
            aname_dict = json.load(f)
        with open('../cope_alias/CWQ_aliase_data31158.json', 'r', encoding='utf-8') as f:
            alias_dict = json.load(f)
        with open('../cope_alias/ComplexWebQuestions_test_wans.json', 'r', encoding='utf-8') as f:
            q_all_list = json.load(f)
            for q_item in q_all_list:
                ans_list = []
                for ans_item in q_item['answers']:
                    if ans_item['answer']:
                        ans_list.append(ans_item['answer'])
                    else:
                        ans_list.append(ans_item['answer_id'])
                    if 'aliases' in ans_item.keys():
                        ans_list += ans_item['aliases']
                
                add_ans_alias_dict[q_item['question']] = ans_list

    elif args.dataset == 'webqsp':
        with open('../cope_alias/WQSP_aliase_data.json', 'r', encoding='utf-8') as f:
            alias_dict = json.load(f)


    for data in output_datas:
        if data['flag'] == 'second':
            q_list.append(data[question_string])
        try:


            answers, ori_data = align(args.dataset, question_string, data, ground_truth_datas, aname_dict, alias_dict, add_ans_alias_dict)
            if 'time' in data.keys():
                call_num_list.append(data['call_num'])
                time_list.append(data['time'])
                token_num_list['input'].append(data['input_token'])
                token_num_list['output'].append(data['output_token'])
                token_num_list['total'].append(data['total_token'])



            
            results = json.loads(data["results"])
            
            if 'Answer' not in results.keys():
                continue
            response = results['Answer']
            if isinstance(response, list):
                flag = 0


                if len(response) == 1:
                    if exact_match(str(response[0]), answers):
                        num_right+=1
                        if data['flag'] == "first":
                            ranker_right.append(data[question_string])
                            first_right += 1
                        else: second_right += 1
                        flag = 1
                        
                else:
                    if exact_match(str(response), answers):
                        num_right+=1
                        if data['flag'] == "first":
                            ranker_right.append(data[question_string])
                            first_right += 1
                        else: second_right += 1
                        flag = 1
                        
                
                if not flag:
                    num_error+=1
                    if data['flag'] == "first":
                        first_error += 1
                    else: second_error += 1                
                    error_question.append(data[question_string])

            else:
                if ',' in response:
                    response = [item.strip() for item in response.split(",")]
                    flag = 0


                    if len(response) == 1:
                        if exact_match(str(response[0]), answers):
                            num_right+=1
                            if data['flag'] == "first":
                                ranker_right.append(data[question_string])
                                first_right += 1
                            else: second_right += 1
                            flag = 1
                            
                    else:
                        if exact_match(str(response), answers):
                            num_right+=1
                            if data['flag'] == "first":
                                ranker_right.append(data[question_string])
                                first_right += 1
                            else: second_right += 1
                            flag = 1
                            
                    
                    if not flag:
                        num_error+=1
                        if data['flag'] == "first":
                            first_error += 1
                        else: second_error += 1                
                        error_question.append(data[question_string])
                else:
                    if exact_match(response, answers):
                        num_right+=1
                        if data['flag'] == "first":
                            ranker_right.append(data[question_string])
                            first_right += 1
                        else: 
                            second_right += 1                    
                    else:
                        if data['flag'] == "first":
                            first_error += 1
                        else: second_error += 1                    
                        num_error+=1
                        error_question.append(data[question_string])
        except:
            num_error += 1
            if data['flag'] == "first":
                first_error += 1
            else:
                second_error += 1


    print("All: ", len(output_datas))
    print("Exact Match: {}".format(float(num_right/len(output_datas)))) 
    print("right: {}, error: {}".format(num_right, num_error))
    print("first===right: {}, error: {}".format(first_right, first_error))
    print("second==right: {}, error: {}".format(second_right, second_error))

    print('call num:',  np.mean(np.array(call_num_list)))
    print('time:',  np.mean(np.array(time_list)))
    for t_type, nu_l in token_num_list.items():
        print(t_type, np.mean(np.array(nu_l)))


