# -*- coding: utf-8 -*-

stop_words={'{','}','NameConstant','PAD'}
stop_terminal_set={'AsName','ModuleName','ClassName','FuncName','AsyncFuncName'}
terminal_set= {'Num','Arg','Str','Bytes','True','False','UNK',
                      'Name','AttrName','Key'}

udfNTset={'For'}

def is_terminal(token):
    """
    处理终结符，终结符要么是terminal_set第一行中的特殊符号，或者是如Name('a')这种带括号的形式
    :param token:
    :param isTerminalSet:
    :return:
    """
    if token in terminal_set:
        return True
    if(token.endswith(')') and token.find(r'(')>=0):
        return True
    return False

def is_udfnonterminal(token):
    """
    处理自定义终结符
    """
    if token in udfNTset:
        return True


def is_nonterminal(token):
    """
    处理终结符，终结符要么是terminal_set第一行中的特殊符号，或者是如Name('a')这种带括号的形式
    :param token:
    :param isTerminalSet:
    :return:
    """
    if token in terminal_set:
        return False
    if(token.endswith(')') and token.find(r'(')>=0):
        return False
    return True

def handle_line(line, wf, isTerminalSet=True):
    line=line.strip()
    cur_line_data=""
    count=0
    types=line.split(" ")
    for token in types:
        count+=1
        if(count>800):
            break;
        cur_line_data+=" "+token
        if count%100==0:
            if token in stop_words:
                continue
            if(is_terminal(token) and isTerminalSet and count>10):
                wf.write(cur_line_data+"\n")
            elif(is_nonterminal(token) and not isTerminalSet and count>10):
                wf.write(cur_line_data+"\n")

#
# original_data_path=r'D:\MyProjectsRepertory\Python_project\s-lstm\data\test.txt'
#
# is_terminal_set=True
# save_path=r'test_terminal.txt' if is_terminal_set else r'test_nonterminal.txt'
#
# count=0
# with open(save_path,'w') as wf:
#     with open(original_data_path) as f:
#         for line in f:
#             handle_line(line, wf, isTerminalSet=is_terminal_set)
#             count+=1
#             # if count>10:
#             #     break



