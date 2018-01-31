# -*- coding: utf-8 -*-
import json
from tools import generate_test_data


def dic_add_item(dic,key,value):
    if key in dic:
        dic[key]+=value
    else:
        dic[key]=value

def calculate_acc(total_dic, hit_dic, acc_dic):
    for k in total_dic:
        if k in hit_dic:
            acc_dic[k] = hit_dic[k] / total_dic[k]
        else:
            acc_dic[k] = 0

def write_dic_res(path,total_dic,hit_dic,acc_dic):
    with open(path,'w') as wf:
        wf.write(json.dumps(total_dic)+'\n')
        wf.write(json.dumps(hit_dic)+'\n')
        wf.write(json.dumps(acc_dic)+'\n')

res_path = r'../data/res/revised/400/lstm.txt'

stop_words_tokens=generate_test_data.stop_words
terminal_set_tokens=generate_test_data.terminal_set

with open('../data/testres/id_to_word.txt','r') as f:
    line=f.readline().strip()
    id_to_word=json.loads(line)

# 初始化结果容器
total_dic = {}
hit_dic = {}
acc_dic={}

nt_total_count=0
nt_total_dic={}
nt_hit_dic={}
nt_acc_dic={}

t_total_count=0
t_total_dic={}
t_hit_dic={}
t_acc_dic={}

#todo add udf nt
udfnt_total_count=0
udfnt_total_dic={}
udfnt_hit_dic={}
udfnt_acc_dic={}


topN=1
with open(res_path, 'r') as f:
    while True:
        true_output_line = f.readline().strip()
        if (true_output_line == ''): break
        pred_output_line = f.readline().strip()
        true_output = json.loads(true_output_line)
        pred_output = json.loads(pred_output_line)
        # i = 60
        # j= 100
        # while i<j:
        #     i+=1
        for i in range(len(true_output)):
            target_id = true_output[i]
            token=id_to_word[str(target_id)]
            if(token in stop_words_tokens):
                continue
            if(generate_test_data.is_terminal(token)):
                candidate_t=[item for item in pred_output[i] if(generate_test_data.is_terminal(id_to_word[str(item)]))]
                if len(candidate_t)>=topN:
                    dic_add_item(t_total_dic,target_id,1)
                    if target_id in candidate_t[:topN]:
                        dic_add_item(t_hit_dic,target_id,1)

            elif(generate_test_data.is_nonterminal(token)):
                candidate_nt=[item for item in pred_output[i] if(generate_test_data.is_nonterminal(id_to_word[str(item)]))]
                if len(candidate_nt)>=topN:
                    dic_add_item(nt_total_dic,target_id,1)
                    if target_id in candidate_nt[:topN]:
                        dic_add_item(nt_hit_dic,target_id,1)

            #todo add UDF NT
            if(generate_test_data.is_udfnonterminal(token)):
                candidate_udfnt=[item for item in pred_output[i] if(generate_test_data.is_udfnonterminal(id_to_word[str(item)]))]
                if len(candidate_nt)>=topN:
                    dic_add_item(udfnt_total_dic,target_id,1)
                    if target_id in candidate_nt[:topN]:
                        dic_add_item(udfnt_hit_dic,target_id,1)

            dic_add_item(total_dic,target_id,1)
            if(target_id in [item for item in pred_output[i][:topN]]):
                dic_add_item(hit_dic,target_id,1)


    calculate_acc(total_dic,hit_dic,acc_dic)
    calculate_acc(t_total_dic,t_hit_dic,t_acc_dic)
    calculate_acc(nt_total_dic,nt_hit_dic,nt_acc_dic)
    calculate_acc(udfnt_total_dic,udfnt_hit_dic,udfnt_acc_dic)


    print(total_dic)
    print(hit_dic)
    print(acc_dic)
    print("")

    print(t_total_dic)
    print(t_hit_dic)
    print(t_acc_dic)
    print("")

    print(nt_total_dic)
    print(nt_hit_dic)
    print(nt_acc_dic)

    print(udfnt_total_dic)
    print(udfnt_hit_dic)
    print(udfnt_acc_dic)


    write_dic_res(r'../data/testres/total_res_dic.txt',total_dic,hit_dic,acc_dic)
    write_dic_res(r'../data/testres/terminal_res_dic.txt',t_total_dic,t_hit_dic,t_acc_dic)
    write_dic_res(r'../data/testres/nonterminal_res_dic.txt',nt_total_dic,nt_hit_dic,nt_acc_dic)
    write_dic_res(r'../data/testres/udfnonterminal_res_dic.txt',udfnt_total_dic,udfnt_hit_dic,udfnt_acc_dic)
