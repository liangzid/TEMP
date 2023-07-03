import sys

import json
import random
import pickle
import os
import shutil
from collections import OrderedDict
import numpy as np
from copy import deepcopy as dcp

logd=print

def concatActionsWithoutValue(dialog_actions):
    result=""
    for action in dialog_actions:
        intent,domain,slot,value=action
        result+="{}+{}+{}=".format(intent,domain,slot)
    if len(dialog_actions)!=0:
        return result[:-1]
    else:
        return result

    
def parserActions(string):
    actions=[]
    candidates=string.split("=")
    for candidate in candidates:
        newcandidate=candidate.split("-")
        actions.append(list(newcandidate))

def makeDelex(acts,utterance):
    triple_ls=[]
    delex_utterance=utterance

    for act in acts:

        intent,domain,slot,value=act
        if slot is "none":
            continue
        if value != "none" and value !="?":
            delex_utterance=replace_or_return(delex_utterance,value,"["+slot+"]")
    return delex_utterance

def makeDelexWithState(belief_state,utterance):
    triple_ls=[]
    delex_utterance=utterance

    if belief_state=={}:
        return utterance
    belief_state=belief_state[list(belief_state.keys())[0]]

    for key in belief_state:

        value=belief_state[key]
        if key is "none" or key=="type":
            continue
        if value != "none" and value !="?":
            delex_utterance=replace_or_return(delex_utterance,value,"["+key+"]")
    return delex_utterance

def replace_or_return(sent,word,wordreplace2):
    """
    if word in sent, then replace these words into wordreplace2;
    else: give a debug info and return sent originally.
    """
    if word in sent and word !=" ":
        logd("word: {}".format(word))
        sls=sent.split(word)
        for i,part in enumerate(sls):
            if i==0:
                newsent=part
            else:
                newsent+=wordreplace2+part
        return newsent
    else:
        logd("Cannot be replaced! So we just return it originally.")
        return sent

def getActionResponseDelexListTriple(src_dataset_path,dst_pk_path):

    ## 0. initilization
    intentslot_templates_dict={}
    asls=[]
    rls=[]
    dls=[]
    
    for jsonfilename in ("/train.json",):
        ## 1. parser json file to dicts
        src_train_p=src_dataset_path+jsonfilename
        with open(src_train_p,'r',encoding='utf8') as f:
            data=json.load(f, object_pairs_hook=OrderedDict)
        dialogues = data['dialogues']

        newdialogues=dcp(dialogues)
        tem_label=0

        ## 2. make inference, and add templates into dicts
        for i,dialog in enumerate(dialogues):
            sentlist=dialog['items']
            for j,sent in enumerate(sentlist):
                if sent["speaker"]=="system":
                    belief_state=sent["belief"]
                    
                    dialogactions=sent["dialogue_act"]

                    originText=sent["text"]
                    actions=concatActionsWithoutValue(dialogactions)
                    # delexresult=makeDelex(dialogactions,originText)
                    # delexresult=makeDelexWithState(belief_state, delexresult)
                    delexresult=sent['delexicalised_text']

                    asls.append(dialogactions)
                    rls.append(originText)
                    dls.append(delexresult)

                    # if actions in intentslot_templates_dict:
                    #     intentslot_templates_dict[actions].append(delexresult)
                    # else:
                    #     intentslot_templates_dict[actions]=[delexresult]

    with open(dst_pk_path,'wb') as f:
        pickle.dump((asls,rls,dls),f)

    print("every thing DONE.")
    return (asls,rls,dls)

def getIntentSlotDict(src_dataset_path,dst_json_path):
    """this file generation `intent-slot to delexicalised response dict.`"""

    ## 0. initilization
    intentslot_templates_dict={}
    
    for jsonfilename in ("/train.json",):
        ## 1. parser json file to dicts
        src_train_p=src_dataset_path+jsonfilename
        with open(src_train_p,'r',encoding='utf8') as f:
            data=json.load(f, object_pairs_hook=OrderedDict)
        dialogues = data['dialogues']

        newdialogues=dcp(dialogues)
        tem_label=0

        ## 2. make inference, and add templates into dicts
        for i,dialog in enumerate(dialogues):
            sentlist=dialog['items']
            for j,sent in enumerate(sentlist):
                if sent["speaker"]=="system":
                    belief_state=sent["belief"]
                    
                    dialogactions=sent["dialogue_act"]

                    originText=sent["text"]
                    actions=concatActionsWithoutValue(dialogactions)

                    delexresult=sent['delexicalised_text']
                    # delexresult=makeDelex(dialogactions,originText)
                    # delexresult=makeDelexWithState(belief_state, delexresult)

                    if actions in intentslot_templates_dict:
                        intentslot_templates_dict[actions].append(delexresult)
                    else:
                        intentslot_templates_dict[actions]=[delexresult]

    with open(dst_json_path,'w',encoding="utf8") as f:
        json.dump(intentslot_templates_dict,f)

    print("every thing DONE.")
    return intentslot_templates_dict


if __name__=="__main__":
    istd=getIntentSlotDict(src_dataset_path="./multiwoz-2.1",
                           dst_json_path="./action-delex-dict.json")
    # print(len(list(istd.keys())))
    # # print(istd.keys())

    getActionResponseDelexListTriple(src_dataset_path="./multiwoz-2.1",
                           dst_pk_path="./ls-triple.pk")
    pass
