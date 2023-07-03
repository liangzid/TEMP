import random
import json
import pickle
import numpy as np
import os
from collections import Counter
from collections import OrderedDict
from bisect import bisect
from math import exp as exp
from copy import copy
from copy import deepcopy as dcp
import sys

import re
sys.path.append("./data/")
from constructIntentSlotCluster import getIntentSlotDict as gis
from constructIntentSlotCluster import getActionResponseDelexListTriple as gardt

def prepareForGenerateVocabList(is_t_dict_path,triple_ls,dst_path):
    """DONE"""

    with open(is_t_dict_path,'r',encoding='utf8') as f:
        intentslot_delex_dict=json.load(f)
    with open(triple_ls,'rb') as f:
        ls_triple=pickle.load(f)

    ## GOAL of this function is generating a tuple list.
    # so we use triple_ls first and then make random sampling from is_t_dict_path;

    # 1. random select samples from is_t_dict with the same number as triple_ls
    nums=len(ls_triple[0])
    sample_list=[]

    keylist=list(intentslot_delex_dict.keys())

    for i in range(nums):
        randomIndex=random.randint(0,len(keylist)-1)
        candidate_list=intentslot_delex_dict[keylist[randomIndex]]
        if len(candidate_list)==1:
            continue
        else:
            random1=random.randint(0,len(candidate_list)-1)
            random2=random.randint(0,len(candidate_list)-1)
            while random2==random1:
                random2=random.randint(0,len(candidate_list)-1)
            sample_list.append((candidate_list[random1],candidate_list[random2]))

    asls,rls,dls=ls_triple
    newtuple_list=zip(rls,dls)
    sample_list.extend(newtuple_list)

    # # unzip
    # s,t=zip(*sample_list)

    with open(dst_path,'wb') as f:
        pickle.dump(sample_list,f)
    print("everything DONE.")

    return sample_list

def actions2Label(actions):
    """WAITING"""
    return label

def sample_exp(elels):
    lens=len(elels)
    ct=Counter(elels)
    ele_ls=[]
    p_ls=[]

    # print(ct)
    for ele in ct:
        ele_ls.append(ele)
        p_ls.append(ct[ele])

    # print(p_ls)
    new_p_ls=softmax(p_ls)
    accumulate_ls=copy(new_p_ls)
    for i,p in enumerate(new_p_ls):
        if i==0:
            continue
        else:
            accumulate_ls[i]=accumulate_ls[i]+accumulate_ls[i-1]
    random_result=random.random()*accumulate_ls[-1]
    index=bisect(accumulate_ls,random_result)
    return ele_ls[index]
    # return index

def softmax(ls):
    newls=[]
    for l in ls:
        newls.append(exp(l))
    sum_result=sum(newls)+10e-5
    newnewls=[]
    for nnl in newls:
        newnewls.append(nnl/sum_result)
    # print(newnewls)
    return newnewls

def winnerTakeAll(elels):
    lens=len(elels)
    ct=Counter(elels)
    ele_ls=[]
    p_ls=[]

    max_result=""
    maxnum=-1
    for ele in ct:
        if ct[ele]>maxnum:
            maxnum=ct[ele]
            max_result=ele
            
    return max_result




def replaceSlotFormat(delex_text):
    delex_text=delex_text.replace("value_","")
    # print(delex_text)
    return delex_text

def replaceFormats(dets):
    big_ls=[]
    for det in dets:
        det_new=det.replace("value_","")
        big_ls.append(det_new)
    return big_ls


def newgis(src_dataset_path,dst_json_path):
    """this file generation `intent-slot to delexicalised response dict.`"""

    ## 0. initilization
    intentslot_templates_dict={}
    
    for jsonfilename in (src_dataset_path,):
        ## 1. parser json file to dicts
        src_train_p=jsonfilename
        with open(src_train_p,'r',encoding='utf8') as f:
            data=json.load(f, object_pairs_hook=OrderedDict)

        newdialogues=dcp(data)
        tem_label=0

        ## 2. make inference, and add templates into dicts
        for i,dialog in enumerate(data):
            sentlist=data[dialog]['log']
            for j,turn in enumerate(sentlist):
                delex_text=turn["resp"]
                delex_text=replaceSlotFormat(delex_text)
                serial_act=turn["sys_act"]

                ## now decode dialogue acts.
                # serial_acts=serial_act.split(" ")
                # domain_ls=["[hotel]", "[police]", "[train]",
                           # "[attraction]", "[restaurant]", "[hospital]",]

                if serial_act in intentslot_templates_dict:
                    intentslot_templates_dict[serial_act].append(delex_text)
                else:
                    intentslot_templates_dict[serial_act]= [delex_text]
                
                
                # if sent["speaker"]=="system":
                #     belief_state=sent["belief"]
                    
                #     dialogactions=sent["dialogue_act"]

                #     originText=sent["text"]
                #     actions=concatActionsWithoutValue(dialogactions)

                #     delexresult=sent['delexicalised_text']
                #     # delexresult=makeDelex(dialogactions,originText)
                #     # delexresult=makeDelexWithState(belief_state, delexresult)

                #     if actions in intentslot_templates_dict:
                #         intentslot_templates_dict[actions].append(delexresult)
                #     else:
                #         intentslot_templates_dict[actions]=[delexresult]

    with open(dst_json_path,'w',encoding="utf8") as f:
        json.dump(intentslot_templates_dict,f)

    print("every thing DONE.")
    return intentslot_templates_dict

def newgardt(src_dataset_path,dst_pk_path):

    ## 0. initilization
    intentslot_templates_dict={}
    asls=[]
    rls=[]
    dls=[]
    
    for jsonfilename in (src_dataset_path,):
        ## 1. parser json file to dicts
        src_train_p=jsonfilename
        with open(src_train_p,'r',encoding='utf8') as f:
            data=json.load(f, object_pairs_hook=OrderedDict)

        newdialogues=dcp(data)
        tem_label=0

        ## 2. make inference, and add templates into dicts
        for i,dialog in enumerate(data):
            sentlist=data[dialog]['log']
            for j,turn in enumerate(sentlist):
                delex_text=turn["resp"]
                s_acts=turn["sys_act"]
                origin_text=delex_text+"this is wrong."

                asls.append(s_acts)
                rls.append(origin_text)
                dls.append(delex_text)

    with open(dst_pk_path,'wb') as f:
        pickle.dump((asls,rls,dls),f)

    print("every thing DONE.")
    return (asls,rls,dls)

def newSampleTriplesWithFakeLabelsFraction(fraction,prate_multiwoz_path,
                                           action_delex_dict_path,
                                           ls_triple_path,
                                           sample_method="exp", makeSave=None):

    ## 1. get the fake labels with `fraction`
    from data.constructIntentSlotCluster import concatActionsWithoutValue as ccawv

    dataset_triple=[]

    response_ls=[]
    actionlabel_ls=[]
    delexlabel_ls=[]

    path=os.path.dirname(os.path.abspath(__file__))

    
    # with open(path+"/action-delex-dict.json",'r',encoding="utf8") as f:
    #     action_delex_dict=json.load(f)


    # with open(path+"/ls-triple.pk",'rb') as f:
    #     ls_triple=pickle.load(f)

    action_delex_dict=newgis(prate_multiwoz_path,
                          action_delex_dict_path)
    ls_triple=newgardt(prate_multiwoz_path,
                    ls_triple_path)
    
    asls,rls,dls=ls_triple

    nums=len(asls)

    # index_list=np.random.randint(low=0,high=nums,size=int(fraction*nums))
    index_list=np.random.choice(range(nums),int(fraction*nums),replace=False)

    for i,actions in enumerate(asls):
        response=rls[i]
        delex_response=dls[i]
        fakelabel=delex_response

        if i in index_list:
            # all_fakes=action_delex_dict[ccawv(actions)]
            all_fakes=action_delex_dict[actions]
            # print(len(all_fakes))

            if sample_method=="exp":
                fakelabel=sample_exp(all_fakes)
            if sample_method=="wta":
                fakelabel=winnerTakeAll(all_fakes)
            else:
                index=random.randint(0,len(all_fakes)-1)
                fakelabel=all_fakes[index]

        response_ls.append(delex_response)
        actionlabel_ls.append(actions)
        delexlabel_ls.append(fakelabel)

    if makeSave is not None:
        with open(makeSave+str(fraction),'wb') as f:
            pickle.dump((response_ls,actionlabel_ls,delexlabel_ls),f)
        print("save intermidiate Varibles DONE.")
            
    return (response_ls,actionlabel_ls,delexlabel_ls)


def sampleTriplesWithFakeLabelsFraction(fraction,prate=0.01, sample_method="exp", makeSave=None,
                                        dataset_path_prefix="/home/liangzi/datasets/soloist/pollution"):

    ## 1. get the fake labels with `fraction`
    from data.constructIntentSlotCluster import concatActionsWithoutValue as ccawv

    dataset_triple=[]

    response_ls=[]
    actionlabel_ls=[]
    delexlabel_ls=[]

    path=os.path.dirname(os.path.abspath(__file__))

    
    # with open(path+"/action-delex-dict.json",'r',encoding="utf8") as f:
    #     action_delex_dict=json.load(f)


    # with open(path+"/ls-triple.pk",'rb') as f:
    #     ls_triple=pickle.load(f)

    action_delex_dict=gis(dataset_path_prefix+f"{prate}-multiwoz-2.1",
                          "./action-delex-dict.json")
    ls_triple=gardt(dataset_path_prefix+f"{prate}-multiwoz-2.1",
                    "./ls-triple.pk")
    
    asls,rls,dls=ls_triple

    nums=len(asls)

    # index_list=np.random.randint(low=0,high=nums,size=int(fraction*nums))
    index_list=np.random.choice(range(nums),int(fraction*nums),replace=False)

    for i,actions in enumerate(asls):
        response=rls[i]
        delex_response=dls[i]
        fakelabel=delex_response

        if i in index_list:
            all_fakes=action_delex_dict[ccawv(actions)]
            # print(len(all_fakes))

            if sample_method=="exp":
                fakelabel=sample_exp(all_fakes)
            if sample_method=="wta":
                fakelabel=winnerTakeAll(all_fakes)
            else:
                index=random.randint(0,len(all_fakes)-1)
                fakelabel=all_fakes[index]

        response_ls.append(delex_response)
        actionlabel_ls.append(actions)
        delexlabel_ls.append(fakelabel)

    if makeSave is not None:
        with open(makeSave+str(fraction),'wb') as f:
            pickle.dump((response_ls,actionlabel_ls,delexlabel_ls),f)
        print("save intermidiate Varibles DONE.")
            
    return (response_ls,actionlabel_ls,delexlabel_ls)

if __name__=="__main__":
    # prepareForGenerateVocabList("action-delex-dict.json","ls-triple.pk","triple_list_for_vocab.pk")
    sampleTriplesWithFakeLabelsFraction(fraction=0.5,makeSave="./prepareResult-")
    pass


