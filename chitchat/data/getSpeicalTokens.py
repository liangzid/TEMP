import json
import re
from collections import OrderedDict


# def getSpecialTokens():
#     slot_list=[]
#     src_train_p="./multiwoz-2.1/train.json"
#     with open(src_train_p,'r',encoding='utf8') as f:
#         data=json.load(f, object_pairs_hook=OrderedDict)
#     dialogues = data['dialogues']

#     ## 2. make inference, and add templates into dicts
#     for i,dialog in enumerate(dialogues):
#         sentlist=dialog['items']
#         for j,sent in enumerate(sentlist):
#             if sent["speaker"]=="system":

#                 dialogactions=sent["dialogue_act"]
#                 for action in dialogactions:
#                     intent,_,slot,_= action
#                     if "[{}]".format(slot.lower()) not in slot_list:
#                         slot_list.append("[{}]".format(slot.lower()))
#     return slot_list

def _getSquareBracketFromSentence(sent):
    result_ls=[]
    pattern=re.compile(r"\[value_.*?\]") 
    candidates=pattern.findall(sent)
    # print(candidates)
    # candidates=pattern.findall(sent)
    for candidate in candidates:
        if "[" in candidate:
            result_ls.append(candidate)
    
    # print("----------")
    # print(result_ls)
    return result_ls
    # if "[" in sent and "]" in sent:
    #     word=sent.split("[")

def extractSpecialTokenFromData(act_delex_dict_path):
    token_ls=[]
    with open(act_delex_dict_path,'r') as f:
        data=json.load(f)
    for key in data:
        delexs=data[key]
        token_ls.extend(_getSquareBracketFromSentence(key))
        for delex in delexs:
            token_ls.extend(_getSquareBracketFromSentence(delex))
    token_set=set(token_ls)
    print(token_set)


if __name__=="__main__":
    # sls=getSpecialTokens()
    # print(sls)
    extractSpecialTokenFromData("../DAMDaction-delex-dict.json")





