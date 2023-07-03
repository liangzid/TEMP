"""
Finetuning the pretrained dialgue models.
---
input: the dialogue history (for one turn dialogue, it is the use utterance)
output: the likelihood of system responses.
"""
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
import json
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import transformers
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM

from transformers import BlenderbotForConditionalGeneration,BlenderbotTokenizer

from torch.utils.data import Dataset, DataLoader
from train import train as model_training

from train import eval_quality
from safe_cls import safeInference as safe_test
from safe_cls import make_dataset_with_text_list
from consistency_cls import consInference as cons_test
from consistency_cls import consmake_dataset_with_text_list

import sys
sys.path.append("/home/liangzi/adc/NLG_eval")
from NLG_eval.eval import evaluate_predictions as nlgeval

from safe_cls import convert_text_to_ids_segment

def getTestDataset(dataset_path="./DiaSafety/DiaSafety_dataset/test.json"):
    """
    dataset_path: the path of dataset
    ---
    the utterance list reading from dataset
    """
    with open(dataset_path,'r') as f:
        data=json.load(f)
    context_ls=[]
    for per_data in data:
        context_ls.append(per_data['context'])
    return context_ls

# def evaluateAllDialougeModels():
#     model_paths=['./pretrained_dialogue_model/blenderbot-400M-distill',
#                  "./pretrained_dialogue_model/DialoGPT-medium",
#                  "./pretrained_dialogue_model/blenderbot_small-90M"]

def raw_dialogue_eval(pretrained_path,
                      dialogue_path="./pretrained_dialogue_model/blenderbot-400M-distill",
                      save_path="./temp_res.json"):
    """
    do not use training dialogue model for safety data, but just running
    inference evaluation.
    ---
    """
    if "blender" in dialogue_path:
        # tokenizer = BlenderbotTokenizer.from_pretrained(dialogue_path)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        model=AutoModelForSeq2SeqLM.from_pretrained(dialogue_path)
        # model=BlenderbotForConditionalGeneration.from_pretrained(dialogue_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        model = AutoModelForCausalLM.from_pretrained(dialogue_path)
    print("Model Load Done.")
    context_ls=getTestDataset()
    # context_ls=[context_ls[0]]
    result_ls=[]

    ## run inference
    progress=tqdm(total=len(context_ls),desc="inference_progress")
    i=0
    for context in context_ls:
        i+=1
        # if i>3:
            # break
        progress.update(1)
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        try:
            new_user_input_ids = tokenizer.encode(context +tokenizer.eos_token,
                                                return_tensors='pt')
            # generated a response while limiting the total chat history to 1000 tokens, 

            # chat_history_ids = model.generate(new_user_input_ids,
            #                                   max_length=1000,
            #                                   pad_token_id=tokenizer.eos_token_id)

            chat_history_ids = model.generate(new_user_input_ids,
                                            max_length=64,)

            # pretty print last ouput tokens from bot
            resp=tokenizer.decode(chat_history_ids[:,
                                                new_user_input_ids.shape[-1]:][0],
                                skip_special_tokens=True)
        except:
            print(f"error context: {context}")
            print(f"error input index: {new_user_input_ids}")
            print(f"error model generation:{chat_history_ids}")
            print(f"error resp: {resp}")

        result_ls.append(resp)


    with open(save_path, 'w',encoding='utf8') as f:
        json.dump(result_ls,f,ensure_ascii=False)
    print(f"Save res to {save_path} done.")

    from collections import OrderedDict
    with open(save_path, 'r',encoding='utf8') as f:
        result_ls=json.load(f,object_pairs_hook=OrderedDict)
    context_ls=context_ls[:len(result_ls)]
        
    # assert len(context_ls)==len(result_ls)

    ## unsupervised automatic metrics evaluation
    nlgscores=nlgeval(result_ls,result_ls)
    print(f"the unsupervised scores of NLG for response: {nlgscores}")

    dialogue_ls=[]
    for i,c in enumerate(context_ls):
        r=result_ls[i]
        dialogue="User: "+c+"System: "+r+' end'
        dialogue_ls.append(dialogue)
    
    ## safety cls evaluation
    # 0. load the safety related model
    safePATH = f'./safe-cls-epoch40-lr3e-06-bs32-with_attentionmask'
    safetokenizer = RobertaTokenizer.from_pretrained(safePATH)
    safemodel = RobertaForSequenceClassification.from_pretrained(safePATH)
    cls_device="cuda:3"
    safemodel=safemodel.to(cls_device)
    print("load safe cls test tokenizer done.")
    print(f"total test samples number: {len(dialogue_ls)}")

    test_set=make_dataset_with_text_list(context_ls,result_ls,
                                         safetokenizer,device=cls_device)
    test_loader = DataLoader(test_set,
                              batch_size=32,
                              shuffle=False,
                              drop_last=False)
    safety_rate=safe_test(test_loader,safemodel,device=cls_device)
    print(f"the safety classification result: {safety_rate}")
        
    return result_ls

def fine_tuning_dialouge_models(pretrained_p,save_p,
                                training_data_p,save_log_path):
    step_print=20
    epoch=2
    max_step=2000
    # max_step=2
    batch_size=1
    lr=3e-5
    epoch=2
    DEVICE="cuda:2"

    print(f"device:{DEVICE}\tstep:{max_step}\tpretrnPth:{pretrained_p}")
    print("---"*10)

    is_encoderdecoder_style=True
    if "DialoGPT" in pretrained_p:
        is_encoderdecoder_style=False

    if "blender" in pretrained_p:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_p)
        model=AutoModelForSeq2SeqLM.from_pretrained(pretrained_p)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_p)
        model = AutoModelForCausalLM.from_pretrained(pretrained_p)

    # tokenizer=AutoTokenizer.from_pretrained(pretrained_p)
    print("tokenizer-eos token:",tokenizer.eos_token)
    eos_token=tokenizer.eos_token
    # if "400" in pretrained_p:
    #     eos_token="</s>"
    # elif "90" in pretrained_p:
    #     eos_token="__end__"
    # elif "DialoGPT" in pretrained_p:
    #     eos_token="</s>"

    model.to(DEVICE)

    ## 1. construct train set.
    with open(training_data_p,'r',encoding="utf8") as f:
        data=json.load(f)["data"]

    target_num=len(data[0][2])

    num_samples=len(data)
    msl=128
    if is_encoderdecoder_style:
        contexts=torch.zeros((num_samples,msl),dtype=torch.long)
        attentions=torch.zeros((num_samples,msl),dtype=torch.long)
        resps=torch.zeros((num_samples,target_num,msl),dtype=torch.long)
        for i,(u,origin_r, newrs) in enumerate(data):
            index_tokens, _, attention_mask = convert_text_to_ids_segment(u,
                                                    max_sentence_length=msl,
                                                    tokenizer=tokenizer)
            contexts[i]=index_tokens
            # print("shape of attentionmask",attention_mask.shape)
            attentions[i,:]=attention_mask
            for j, newr in enumerate(newrs):
                resps[i,j],_,_ = convert_text_to_ids_segment(newr+eos_token,
                                                        max_sentence_length=msl,
                                                        tokenizer=tokenizer)
        dataset=TensorDataset(contexts,attentions,resps)

        train_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=False)

        optimizer2 = transformers.AdamW(
            model.parameters(), lr=lr, correct_bias=True)

        # 2. training
        total_step=0
        bad_attention_time=0
        for e in range(epoch):
            for i, (inputs,attentions,outputs) in enumerate(train_loader):
                inputs,attention_mask,labels=inputs.to(DEVICE),\
                    attentions.to(DEVICE),outputs.to(DEVICE)
                print("three shapes:",inputs.shape,
                    attention_mask.shape,
                    labels.shape)
                loss=0.
                for j in range(target_num):
                    outputs = model(inputs,attention_mask,labels=labels[:,j,:])
                    loss += outputs.loss
                loss.backward()
                optimizer2.step()
                optimizer2.zero_grad()

                total_step+=1
                if total_step%step_print==0:
                    print(f"train loss: {loss}")
                if total_step>max_step:
                    break
    else:
        contexts=torch.zeros((num_samples,target_num,msl),dtype=torch.long)
        outputs=torch.zeros((num_samples,target_num,msl),dtype=torch.long)
        for i,(u,origin_r, newrs) in enumerate(data):
            for j,newr in enumerate(newrs):
                index_tokens, _, _ = convert_text_to_ids_segment(u+newr+eos_token,
                                                        max_sentence_length=msl,
                                                        tokenizer=tokenizer)
                # contexts[i,j]=index_tokens[:-1]
                # outputs[i,j]=index_tokens[1:]
                contexts[i,j]=index_tokens
                outputs[i,j]=contexts[i,j]
        dataset=TensorDataset(contexts,outputs)

        train_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=False)

        optimizer2 = transformers.AdamW(
            model.parameters(), lr=lr, correct_bias=True)

        # 2. training
        total_step=0
        bad_attention_time=0
        for e in range(epoch):
            for i, (inputs,outputs) in enumerate(train_loader):
                inputs=inputs.to(DEVICE)
                # print("three shapes:",inputs.shape,
                #     # attention_mask.shape,
                #     outputs.shape)

                loss=0.
                for j in range(target_num):
                    outputs = model(inputs[:,j,:],labels=inputs[:,j,:])
                    loss += outputs.loss
                loss.backward()
                optimizer2.step()
                optimizer2.zero_grad()

                total_step+=1
                if total_step%step_print==0:
                    print(f"train loss: {loss}")
                if total_step>max_step:
                    break

    ## save the checkpoint
    # tokenizer.save_pretrained(save_p)
    model.save_pretrained(save_p)

    raw_dialogue_eval(pretrained_p,save_p,
                      save_path=save_log_path)

def dialogue_result_export(
        dialogue_path="./pretrained_dialogue_model/blenderbot-400M-distill"
        ):
    """
    do not use training dialogue model for safety data, but just running
    inference evaluation.
    ---
    """
    device="cuda:6"
    if "blender" in dialogue_path:
        # tokenizer = BlenderbotTokenizer.from_pretrained(dialogue_path)
        tokenizer = AutoTokenizer.from_pretrained(dialogue_path)
        model=AutoModelForSeq2SeqLM.from_pretrained(dialogue_path)
        model.to(device)
        # model=BlenderbotForConditionalGeneration.from_pretrained(dialogue_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(dialogue_path)
        model = AutoModelForCausalLM.from_pretrained(dialogue_path)
    print("Model Load Done.")
    # context_ls=getTestDataset()
    context_ls=getTestDataset("./DiaSafety/DiaSafety_dataset/only_unsafe_test.json")
    # context_ls=[context_ls[0]]
    result_ls=[]

    ## run inference
    for context in context_ls:
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        try:
            new_user_input_ids = tokenizer.encode(context +tokenizer.eos_token,
                                                  truncation=True,
                                                  max_length=64,
                                                return_tensors='pt')
            new_user_input_ids=new_user_input_ids.to(device)

            # generated a response while limiting the total chat history to 1000 tokens, 

            # chat_history_ids = model.generate(new_user_input_ids,
            #                                   max_length=1000,
            #                                   pad_token_id=tokenizer.eos_token_id)

            chat_history_ids = model.generate(new_user_input_ids,
                                            max_length=1000,)
            # pretty print last ouput tokens from bot
            resp=tokenizer.decode(chat_history_ids[:,
                                                new_user_input_ids.shape[-1]:][0],
                                skip_special_tokens=True)
        except:
            print(f"error context: {context}")
            print(f"error input index: {new_user_input_ids}")
            print(f"error model generation:{chat_history_ids}")
            print(f"error resp: {resp}")

        result_ls.append(resp)

    assert len(context_ls)==len(result_ls)

    ## now reshape to the formation into a standard test formation
    ls_items=[]
    for i,c in enumerate(context_ls):
        adict={"context":c,"response":result_ls[i],
               "category":"none","label":"none"}
        ls_items.append(adict)
    model_name=dialogue_path.split("/")[-1]
    savepath=f"./DiaSafety/DiaSafety_dataset/test_{model_name}onlyunsafe.json"
    with open(savepath, 'w',encoding='utf8') as f:
        json.dump(ls_items,f,ensure_ascii=False)
    print(f"save to {savepath} done.")

if __name__=="__main__":
    model_paths=['./pretrained_dialogue_model/blenderbot-400M-distill',
                 "./pretrained_dialogue_model/DialoGPT-medium",
                 "./pretrained_dialogue_model/blenderbot_small-90M"]

    # tokenizer=AutoTokenizer.from_pretrained(model_paths[2])
    # tokenizer.save_pretrained(model_paths[2]+"_finetuned")
    # tokenizer=AutoTokenizer.from_pretrained(model_paths[2]+"_finetuned")
    # print("-\n-\n--\n---\n--")

    # raw_dialogue_eval(model_paths[1])
    # raw_dialogue_eval(model_paths[0])
    # raw_dialogue_eval(model_paths[2])

    # dialogue_result_export(model_paths[1])
    # dialogue_result_export(model_paths[2])

    # dialogue_result_export(model_paths[0])

    # train_set_path="./data/0.1special_triples_sampled_targetnum5.json"
    train_set_path="./data/0.1special_triples_sampled_targetnum1.json"

    # fine_tuning_dialouge_models(model_paths[0],
    #                             model_paths[0]+"__finetuned",
    #                             training_data_p=train_set_path)


    fine_tuning_dialouge_models(model_paths[1],
                                model_paths[1]+"_finetuned",
                                training_data_p=train_set_path,
                                save_log_path=f"{model_paths[1]}.res.json")
    


    # fine_tuning_dialouge_models(model_paths[2],
    #                             model_paths[2]+"_finetuned",
    #                             training_data_p=train_set_path,
    #                             save_log_path=f"{model_paths[2]}.res.json")
