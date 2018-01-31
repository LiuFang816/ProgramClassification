# -*- coding: utf-8 -*-
import json

stop_words={'4999','0','1'}

with open('../data/testres/id_to_word.txt','r') as f:
    line=f.readline().strip()
    id_to_word=json.loads(line)

def display_detail(filepath):

    influence_dic={}
    with open(filepath,'r') as f:
        total_dic=json.loads(f.readline().strip())
        hit_dic=json.loads(f.readline().strip())
        acc_dic=json.loads(f.readline().strip())

        acc_dic=sorted(acc_dic.items(), key=lambda x:x[1])
        total_tokens=0
        for (k,v) in total_dic.items():
            if(k in stop_words):
                continue
            total_tokens+=v
        total_sum=0
        hit_sum=0
        for (k,v) in acc_dic:
            if(k in stop_words):
                continue
            # if(total_dic[k]>=20 and v > 0 and v<1):
            # print('%s  %.3f  %d'%(id_to_word[k],v,total_dic[k]))
            influence_dic[k]=float(float(1.0-v)*1.0*total_dic[k])
                # print(str(id_to_word[k])+' , '+str(v)+" , "+str(total_dic[k]))
            if v >0:
                total_sum+=total_dic[k]
                hit_sum+=hit_dic[k]

                # if v>0:
                #     print(str(k)+' , '+str(v))
        print(total_tokens)
        print(total_sum)
        print(hit_sum)
        print(hit_sum/total_tokens)

        # influence_dic=sorted(influence_dic.items(), key=lambda x:x[1],reverse=True)
        # sum=0
        # for (k,v) in influence_dic:
        #     print('%s  %d'%(id_to_word[k],v))
        #     sum+=v
        #     if sum>int(0.9*(total_tokens-hit_sum)):
        #         break
        # # print(influence_dic)

filePath=r'../data/testres/total_res_dic.txt'
print("============display total===================")
display_detail(filePath)

nt_filepath=r'../data/testres/nonterminal_res_dic.txt'
t_filepath=r'../data/testres/terminal_res_dic.txt'
udfnt_filepath=r'../data/testres/udfnonterminal_res_dic.txt'

print("============display nt===================")
display_detail(nt_filepath)

print("============display udf nt===================")
display_detail(udfnt_filepath)

print("============display t===================")
display_detail(t_filepath)