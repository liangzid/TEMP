'''
dataloader construction.

zi liang
2021.10.08

'''

from torch.utils.data import DataLoader,Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from tqdm import tqdm
import time
import pickle
import os
import json
import random
from pathlib import Path
import re

from data.prepareDataset import sampleTriplesWithFakeLabelsFraction as pfgv
from data.prepareDataset import newSampleTriplesWithFakeLabelsFraction as newpfgv
#--------------------------------------------------------------------------------

# =================================
def nluGenerateQueryDict():
    path=os.path.dirname(os.path.abspath(__file__))
    with open(path+"/nludict.json",'r',encoding='utf8') as f:
        data=json.load(f)

    id2act_dict=data["id2act_dict"]
    act2id_dict=data["act2id_dict"]
    return id2act_dict,act2id_dict

# =================================
def newnluGenerateQueryDict(dict_name):
    path=os.path.dirname(os.path.abspath(__file__))
    with open(dict_name,'r',encoding='utf8') as f:
        data=json.load(f)

    id2act_dict=data["id2act_dict"]
    act2id_dict=data["act2id_dict"]
    return id2act_dict,act2id_dict
# def nlu_id2string(id,querydict):
#     return querydict[string(int(id))]
    

# def nlu_str2id(string,querydict):
#     return querydict[stirng]

def concatActionsWithoutValue(dialog_actions):
    result=""
    for action in dialog_actions:
        intent,domain,slot,value=action
        result+="{}+{}+{}=".format(intent,domain,slot)
    if len(dialog_actions)!=0:
        return result[:-1]
    else:
        return result
    

def concatSlots(dialog_actions):
    result=""
    for action in dialog_actions:
        intent,domain,slot,value=action
        result+="{}[SEP]=".format(slot)
    if len(dialog_actions)!=0:
        return result[:-1]
    else:
        return result


def extractSlotFromDelexilisedSentence(delexilised_res):
    slot_ls=[]
    if "[" in delexilised_res:
        left_ls=delexilised_res.split("[")
        for left in left_ls:
            if "]" in left:
                candidate_ls=left.split("]")
                slot=candidate_ls[0]
                slot_ls.append(slot)
    strr=""
    for ele in slot_ls:
        strr+=ele
    return strr
        
# # =================================

class SampleGenerationForTrainingDataset(Dataset):
    '''for sampling style translations.'''
    def __init__(self, args, tokenizer, prate=0.0,
                 target_num=-1, sample_method="random",
                 fraction=0.5, back_prediction=0,
                 mode="train",use_damd_style_data=0,dataset_path_prefix="/home/liangzi/datasets/soloist/pollution"):

        self.path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.max_sentence_length=64
        # tokenizer.padding_side="left"
        # tokenizer.pad_token=tokenizer.eos_token
        self.tokenizer=tokenizer
        # self.data_path=data_path

        self.act2id_dict=nluGenerateQueryDict()[1]
        
        # if Path("prepareresult-"+str(fraction)).exists():
        #     # load iterator and begin to make transfering.
        #     with open(self.data_path, 'rb')as f:
        #         response_ls,actions_ls,delex_ls=pickle.load(f)
        # else:
        #     response_ls,actions_ls,delex_ls=pfgv(fraction,"./prepareResult-")

        if use_damd_style_data==0:
            response_ls,actions_ls,delex_ls=pfgv(fraction=fraction,makeSave="./prepareResult-",
                                                 sample_method=sample_method, prate=prate,
                                                 dataset_path_prefix=dataset_path_prefix)
        else:
            pathprefix="/home/liangzi/yxp/GPU/damd/damd-multiwoz/data/train_data/polluted_rate_"
            # path_houzhui=f""
            response_ls,actions_ls,delex_ls=newpfgv(fraction=fraction,
                                             makeSave="./DAMDprepareResult-",
                                             sample_method=sample_method,
                                        prate_multiwoz_path=f"{pathprefix}{prate}/data_for_damd_{prate}.json",
                                             action_delex_dict_path="./DAMDaction-delex-dict.json",
                                             ls_triple_path="./DAMDls-triple.pk")

        print("ROUGH INFORMATION LOADED DONE. NOW GENERATING PROCESS ON.")

        # if mode=="train":
        #     ss=tra
        # elif mode=="val":
        #     ss=val
        # elif mode=="test":
        #     ss=tes
        #     # print(ss)

        # get tagging label for each dataset.
        self.dataset=[]

        ## set input text
        max_source_length=args.max_seq_length
        max_target_length=args.max_seq_length

        encoding = self.tokenizer(response_ls,
                     padding='longest',
                     max_length=max_source_length,
                     truncation=True,
                     return_tensors="pt")
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        
        # print(actions_ls[0])
        ## set output text
        if target_num==-1:
            target_num=1
        target_list=[]

        for _ in range(target_num):
            if args.back_prediction==1:
                target_encoding = tokenizer([delex_ls[i] +"[BP]"+extractSlotFromDelexilisedSentence(response_ls[i])+"</s>"  for i in range(len(actions_ls))],
                                padding='longest',
                                max_length=max_target_length,
                                truncation=True)
            else:
                target_encoding = tokenizer([delex_ls[i]+"</s>"  for i in range(len(actions_ls))],
                                padding='longest',
                                max_length=max_target_length,
                                truncation=True)
            target_list.append(target_encoding.input_ids)

        target_tensor=torch.tensor(target_list) # shape: target_num*sample_num*msl
        target_tensor=torch.transpose(target_tensor,0,1)

        # print(target_tensor)
        # print(target_tensor.shape)


        # if args.back_prediction==1:
        #     target_encoding = tokenizer([delex_ls[i] +"[BP]"+extractSlotFromDelexilisedSentence(response_ls[i])+"</s>"  for i in range(len(actions_ls))],
        #                     padding='longest',
        #                     max_length=max_target_length,
        #                     truncation=True)
        # else:
        #     target_encoding = tokenizer([delex_ls[i]+"</s>"  for i in range(len(actions_ls))],
        #                     padding='longest',
        #                     max_length=max_target_length,
        #                     truncation=True)
        # target_tensor=torch.tensor(target_encoding.input_ids)

        # replace padding token id's of the labels by -100
        # labels = [
        #         [(label if label != tokenizer.pad_token_id else -100) for label in labels_example]   for labels_example in labels
        # ]
        # labels = torch.tensor(labels)
        
        # print(input_ids[0].shape)
        # print(attention_mask[0].shape)
        # print(target_tensor[0,0].shape)
        for i in range(len(actions_ls)):
            self.dataset.append((input_ids[i],attention_mask[i],target_tensor[i]))

    def __getitem__(self,i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)

    
# # test doublelasertaggerdataset
# if __name__ == "__main__":
#     adataset=doubleLaserTaggerDataset(MyTokenizer(),mode="train1")
#     i =0
#     for a,b in adataset:
#         print(len(a))
#         print(len(a[0]))
#         print(len(b))
#         i+=1
#         if i >4:
#             break
    
