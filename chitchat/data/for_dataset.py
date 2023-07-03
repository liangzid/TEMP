'''
dataloader construction.

zi liang
2022.06.20
'''

from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence
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
import time
from collections import OrderedDict


from data.parser_safe_dataset import parseSafeJson2TripleLs, SampleURLTriple
from data.parser_safe_dataset import UnsupervisedSampleURLTriple

# # =================================

class SampleRephraingForTrainingDataset(Dataset):
    '''for sampling style translations.'''
    def __init__(self, args, tokenizer, 
                 target_num=1, 
                 mode="train",
                 dataset_path_prefix="/home/liangzi/adc/DiaSafety/DiaSafety_dataset/train.json"):
        root_dir="/home/liangzi/adc/data/"

        sample_type="unsupervised"
        # sample_type="supervised"

        self.path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.max_sentence_length=64
        # tokenizer.padding_side="left"
        # tokenizer.pad_token=tokenizer.eos_token
        self.tokenizer=tokenizer
        # self.data_path=data_path

        sets=parseSafeJson2TripleLs(dataset_path_prefix)
        t1=time.time()
        
        if sample_type!="unsupervised":
            newsets=SampleURLTriple(sets,target_num=target_num)
        else:
            ## if exist, then read
            fraction=args.fraction
            print(f"fraction:{fraction}, tgt num:{target_num}")
            sampled_file_reading=root_dir+f"/{fraction}special_triples_sampled_targetnum{target_num}.json"
            print(f"loading unsupervised_path: {sampled_file_reading}")
            is_exist_x=os.path.exists(sampled_file_reading)
            is_exist_x=False

            if is_exist_x:
                with open(sampled_file_reading,
                            'r',encoding="utf8") as f:
                    newsets=json.load(f,object_pairs_hook=OrderedDict)['data']
            else:
                newsets,_=UnsupervisedSampleURLTriple(sets,
                        target_num=target_num,
                                                      anns_num=150,
                                                      eps=0.40,
                                                      tau=1,
                                                      )
                with open(sampled_file_reading, 'w',encoding='utf8') as f:
                    json.dump({'data':newsets},f,ensure_ascii=False)

            t2=time.time()
            print("dataset loading and preparation time cost: {}".format(t2-t1))
            del sets
        
        # utterancels, resp_ls, sim_resp_ls=zip*(newsets)
        utterancels=[]
        resp_ls=[]
        sim_resp_ls=[]
        for a,b,c in newsets:
            utterancels.append(a)
            resp_ls.append(b)
            sim_resp_ls.append(c)

        
        dialogue_ls=[]
        for i,u in enumerate(utterancels):
            # # utterance-response mode
            dialogue="User: "+u+" System: "+resp_ls[i]
            dialogue_ls.append(dialogue)

            # only response mode
            # dialogue_ls.append(resp_ls[i])
        
        # get tagging label for each dataset.
        self.dataset=[]

        ## set input text
        max_source_length=args.max_seq_length
        max_target_length=args.max_seq_length

        # self.tokenizer.truncation_side="left"
        encoding = self.tokenizer(
            text=dialogue_ls,
            # text=utterancels,
                                  # text_pair=resp_ls,
                     padding='longest',
                     max_length=max_source_length,
                     truncation=True,
                     return_tensors="pt")
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        # token_type_ids=encoding.token.token_type_ids
        
        # print(actions_ls[0])
        ## set output text
        if target_num==-1:
            target_num=1
        target_list=[]

        # self.tokenizer.truncation_side="right"
        target_num=len(sim_resp_ls[0])
        
        for x in range(target_num):
            target_encoding = tokenizer([sim_resp_ls[i][x]+"</s>" \
                                    for i in range(len(sim_resp_ls))],
                                padding='max_length',
                                max_length=max_target_length,
                                truncation=True)
            target_list.append(target_encoding.input_ids)

        
        target_tensor=torch.tensor(target_list) # shape: target_num*sample_num*msl
        target_tensor=torch.transpose(target_tensor,0,1)

        for i in range(len(sim_resp_ls)):
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
