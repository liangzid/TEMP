import random
import argparse
from tqdm import tqdm
import logging
import os
from os.path import join, exists
from itertools import zip_longest, chain
from datetime import datetime
import pickle

import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import BertTokenizer
from transformers import pipeline

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn as nn

# from data.for_dataset import SampleGenerationForTrainingDataset
# from huggingface_transformers_self.src.transformers.models.t5 import T5ForConditionalGeneration as MyNewT5
MyNewT5=T5ForConditionalGeneration

class Inference:
    def __init__(self, model_path="data/rettig_model", cuda_num=6,
                 gbias=0, bp=0,
                 seed=3933, cuda=True,is_for_damd=0):
        #------------------------基础设备的定义和使用----------------------------
        # 日志同时输出到文件和console
        device = 'cuda:{}'.format(cuda_num) if cuda else 'cpu'
        self.device = device
        print('using device:{}'.format(device))
        # 设置使用哪些显卡进行训练
        # os.environ["CUDA_VISIBLE_DEVICES"] ="6"
        self.bp=bp

        # -----------------------------加载数据集，token，和模型---------------------------
        # 加载tokenizer
        master_path = "/home/liangzi/TEMP/soloist/paraphrase/"

        if is_for_damd==0:
            bla_tokenizer = T5Tokenizer.from_pretrained(master_path+"t5-small")
            new_token_list=["[BP]"]
            bla_tokenizer.add_tokens(new_token_list,special_tokens=True)
        else:
            bla_tokenizer = T5Tokenizer.from_pretrained(master_path+f"{model_path}")
        
        if "/home"  not in model_path:
            bla_tokenizer = T5Tokenizer.from_pretrained(master_path+f"{model_path}")
        else:
            bla_tokenizer = T5Tokenizer.from_pretrained(f"{model_path}")
        # print("=======------------")
        # print(bla_tokenizer.encode("[value_name]"))
        # print(bla_tokenizer.decode([32102,1]))
        print("tokenizer loading done...")
        self.tokenizer = bla_tokenizer

        print("INFERENCE-MODEL-PATH: {}".format(model_path))

        if gbias==0:
            if "/home"  in model_path:
                apath=model_path
            else:
                apath=master_path+model_path
            self.decoder = T5ForConditionalGeneration.from_pretrained(apath)
        else:
            ## using guass relative position bias.
            self.decoder=MyNewT5.from_pretrained(master_path+model_path)
        
        self.decoder.resize_token_embeddings(len(bla_tokenizer))
        print("model loading done...")
        # decoder.resize_token_embeddings(len(bla_tokenizer))

        self.decoder.to(device)
        self.decoder.eval()

        # # 尝试将模型设置成多GPU模式
        multi_gpu = False
        # if args.cuda and torch.cuda.device_count() > 1:
        #     LOGGER.info("Let's use GPUs to train")    
        #     decoder = DataParallel(decoder, device_ids=[int(i) for i in args.device.split(',')])
        #     multi_gpu = True

        # 记录模型参数数量
        num_parameters = 0
        parameters = self.decoder.parameters()
        for parameter in parameters:
            num_parameters += parameter.numel()
        print('number of all parameters: {}'.format(num_parameters))

        #--------------------------------测试模型效果----------------------------------------
        # 读取测试数据
        self.max_source_length=512
        self.max_target_length=256

        # for seq in input_sequences:
        #     input_ids.append(tokenizer(seq),return_tensor="pt").input_ids
        
    def inference(self, sequence, generate_mode_test="greedy"):
        new_sent = []
        progress=tqdm(total=len(sequence),desc="Inference Progress")
        print('==========starting testing==========')
        for seq in sequence:
            # print("input: {}".format(seq))
            input_ids = self.tokenizer.encode(seq, return_tensors="pt")
            # print(input_ids)
            input_ids = input_ids.to(self.device)
            # print("input id: {}".format(input_ids))

            try:
                if generate_mode_test == "greedy":
                    # 修改之后的做法，基于beam search的文本生成
                    outputs=self.decoder.generate(input_ids=input_ids,
                                                  max_length=self.max_target_length,
                                                  repetition_penalty=2.5,
                                                  no_repeat_ngram_size=3,
                                                  )
                    # if "Fuck" in seq:
                    #     print("------"*5)
                    #     print("generate {} done.".format(outputs))
                    #     print("string: {}.".format(self.tokenizer.decode(outputs[0],
                    #                                                     skip_special_tokens=True)))

                elif generate_mode_test=="beam":
                    # 基于贪婪搜索的文本生成
                    outputs=self.decoder.generate(input_ids=input_ids,
                                                    num_beams=5,
                                                    num_return_sequences=5,
                                                  max_length=self.max_target_length,
                                                  early_stopping=True)
                    # print("------"*5)
                    # print("generate {} done.".format(outputs))
                    # print("string: {}.".format(self.tokenizer.decode(outputs[0])))
                    # print("shape of output is: {}".format(outputs.shape))
                # print("下面是生成的句子：")

                sentence=self.tokenizer.decode(outputs[0],skip_special_tokens=True)
                # print("sentence: {}".format(sentence))
                if "</s>" in sentence:
                    sentence=sentence.split("</s>")[0]
                if "<pad>" in sentence:
                    sentence=sentence.split("<pad>")[-1]
                # print("sentence: {}".format(sentence))

                if self.bp==1 and "[BP]" in sentence:
                    sentence=sentence.split("[BP]")[0]
                new_sent.append(sentence)

                # print("Predict: {}".format(self.tokenizer.decode(outputs[0])))
                # LOGGER.info("Label: {}".format(tokenizer.convert_prediction_result2_sentence(response.unsqueeze(0))))
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    oom_time += 1
                    print("WARNING: ran out of memory,times: {}".format(oom_time))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    print(str(exception))
                    raise exception
            progress.update(1)
        print('=============testing finished====================')
        return new_sent

def main():
    inputt="<pad>You're welcome. Is there anything else I can help you with?</s>"
    inferenceModel=Inference(model_path="./data/save_without_bp_lr_3e-4_rate-0.8/",cuda_num=0)
    xxx=inferenceModel.inference([inputt])
    print(xxx)

if __name__=="__main__":
    main()

    
    
