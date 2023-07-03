"""
Training defender model.

Zi Liang
2022.06.13
"""
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

import json
from collections import OrderedDict
import random
import argparse
from tqdm import tqdm
import logging
import os
from os.path import join, exists
from itertools import zip_longest, chain
from datetime import datetime
import pickle
import time

import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
# from transformers import BertTokenizer
from transformers import pipeline

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn as nn

from data.for_dataset import SampleRephraingForTrainingDataset

import sys
sys.path.append("/home/liangzi/")
# from huggingface_transformers_self.src.transformers.models.t5 import T5ForConditionalGeneration as MyNewT5


from safe_cls import safeInference as safe_test
from safe_cls import inferenceAndReplace
from safe_cls import make_dataset_with_text_list
from consistency_cls import consInference as cons_test
from consistency_cls import consmake_dataset_with_text_list

from consistency_cls import pplmake_dataset_with_text_list
from consistency_cls import ppl_test

import sys
sys.path.append("/home/liangzi/adc/NLG_eval")
from NLG_eval.eval import evaluate_predictions as nlgeval


LOGGER = None
PAD_ID = None

def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_length", default=64,
                        type=int, required=False, help="模型的最大输入长度")
    parser.add_argument("--max_step", default=500,
                        type=int, required=False, help="max step for training.")
    parser.add_argument("--train", default=1, type=int,
                        required=True, help="用以决定是训练模式还是测试模式")
    parser.add_argument('--device', default='6', type=str,
                        required=False, help='设置使用哪些显卡')
    parser.add_argument('--cuda_num', default='6', type=str, required=False)
    parser.add_argument('--fraction', default='0.1',
                        type=str, required=False)
    parser.add_argument('--prate', default='0.1',
                        type=float, required=False)
    parser.add_argument('--back_prediction', default=0,
                        type=int, required=False)

    parser.add_argument('--target_num', default=-1,
                        type=int, required=False)
    
    parser.add_argument('--is_for_damd', default=0,
                        type=int, required=False)
    
    parser.add_argument('--sample_method', default="random",
                        type=str, required=False)

    parser.add_argument('--gbias', default=0,
                        type=int, required=False)
    
    parser.add_argument('--board_name', default="nothing",
                        type=str, required=False)

    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')

    parser.add_argument('--save_log_path', default='./log/training.log',
                        type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--epochs', default=5, type=int,
                        required=False, help='训练的轮次')
    parser.add_argument('--batch_size', default=4, type=int,
                        required=False, help='训练batch size')
    parser.add_argument('--lr', default=3e-4, type=float,
                        required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000,
                        type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int,
                        required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation', default=1,
                        type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0,
                        type=float, required=False)

    parser.add_argument('--save_model_path', default='./savemodel/',
                        type=str, required=False, help='对话模型输出路径')
    parser.add_argument('--pretrained_model_path', default='t5-small',
                        type=str, required=True, help='预训练的GPT2模型的路径')

    parser.add_argument('--writer_dir', default='./tensorboard_summary',
                        type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--seed', type=int, default=3933,
                        help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int,
                        default=1, help="dataloader加载数据时使用的线程数量")
    # parser.add_argument("--datasetpath", type=str, required=True, help="设置数据集地址")
    parser.add_argument("--generate_mode_test", type=str,
                        required=False, default="beam", help="设置数据集地址")
    parser.add_argument("--dataset_path_prefix", type=str,
                        required=False, default="/home/liangzi/adc/DiaSafety/DiaSafety_dataset/train.json", help="设置数据集地址")
    parser.add_argument("--prefix_existing_model", type=str,
                        required=False, default="none", help="设置数据集地址")
    parser.add_argument("--reading_safety_path", type=str,
                        required=False, default="raw", help="设置数据集地址")
    parser.add_argument("--is_using_ADC", type=str,
                        required=False, default="0", help="设置数据集地址")

    return parser.parse_args()


def _set_random_seed(seed):
    """
    设置训练的随机种子
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def _create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.save_log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def main(args):
    #----------------基础设备的定义和使用------------------------------
    # 日志同时输出到文件和console
    global LOGGER
    LOGGER = _create_logger(args)
    global PAD_ID
    PAD_ID = "[PAD]"
    # 设置有关设备的问题
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda:{}'.format(args.cuda_num) if args.cuda else 'cpu'
    LOGGER.info('using device:  {}'.format(device))
    # 设置使用哪些显卡进行训练
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

    if args.seed:
        _set_random_seed(args.seed)
    # -------------------------------------------------------------------------
    #---------------------加载数据集，token，和模型----------------------
    # 加载tokenizer
    if "art" in args.pretrained_model_path:
        tokenizer=BartTokenizer.from_pretrained(args.pretrained_model_path)
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path)
    a_tokenizer = tokenizer
        

    # PAD
    # 加载模型
    vocab_size = -1
    # 设置GPU ID
    # torch.cuda.set_device(args.device)

    # 加载GPT2模型
    # 自定义预训练模型，只需要运行一次
    LOGGER.info("正在初始化预训练模型...")
    # decoder=GPT_Decoder(path=args.pretrained_model_path)

    if "art" in args.pretrained_model_path:
        decoder=BartForConditionalGeneration.from_pretrained(
            args.pretrained_model_path)
    else:
        decoder = T5ForConditionalGeneration.from_pretrained(
            args.pretrained_model_path)
    
    decoder.resize_token_embeddings(len(a_tokenizer))
    decoder.to(device)

    # # 尝试将模型设置成多GPU模式
    multi_gpu = False
    # args.device = "1,2,3"
    # if args.cuda and torch.cuda.device_count() > 1:
    #     LOGGER.info("Let's use GPUs to train")
    #     decoder = DataParallel(decoder, device_ids=[
    #                            int(i) for i in args.device.split(',')])
    #     multi_gpu = True

    # 记录模型参数数量
    num_parameters = 0
    parameters = decoder.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    LOGGER.info('number of all parameters: {}'.format(num_parameters))

    # 加载数据
    LOGGER.info("loading traing data")
    train_dataset = SampleRephraingForTrainingDataset(args=args,
                                                       tokenizer=tokenizer,
                                                       target_num=args.target_num,
                                                       mode="train2",
                                        dataset_path_prefix=args.dataset_path_prefix)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size, shuffle=True)
    #----------------------开始训练------------------------------------------
    train(a_tokenizer, decoder, device, train_dataset,
          train_loader, multi_gpu, args)

# =======================================

def train(tokenizer, Decoder, device, train_dataset, dataloader, multi_gpu, args):

    Decoder.train()
    # 计算所有epoch进行参数优化的总步数total_steps
    total_steps = int(train_dataset.__len__() * args.epochs /
                      args.batch_size / args.gradient_accumulation)
    LOGGER.info('total training steps = {}'.format(total_steps))

    # 设置优化器，并且在初始训练时，使用warmup策略
    optimizer2 = transformers.AdamW(
        Decoder.parameters(), lr=args.lr, correct_bias=True)
    # scheduler2 = transformers.WarmupLinearSchedule(optimizer2, warmup_steps=args.warmup_steps, t_total=total_steps)

    LOGGER.info('==========starting training==========')
    # 用于统计每次梯度累计的loss
    running_loss = 0.
    # 统计一共训练了多少个step
    overall_step = 0
    # 记录tensorboardX
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    # 记录 out of memory的次数
    oom_time = 0
    # 开始训练
    accuracy = -1.
    step_break_flag=0
    # for epoch in tqdm(range(args.epochs),desc="迭代器进度"):
    for epoch in range(args.epochs):
        epoch_start_time = datetime.now()
        for batch_idx, (utterance_input,
            attentions, labels) in enumerate(dataloader):

            utterance_input = utterance_input.to(device)
            # token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            attentions = attentions.to(device)
            # print(f"INPUT: {tokenizer.decode(utterance_input[0])}")
            # print(f"LABEL: {tokenizer.decode(labels[0][0])}")
            try:
                loss_func_cls = CrossEntropyLoss(ignore_index=PAD_ID, reduction='sum')
                # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
                # loss_func_cls = CrossEntropyLoss(reduction='sum')

                bs,target_num,msl=labels.shape
                # print("the shape of label is:",labels.shape)
                
                loss=0.
                for i in range(target_num):
                    loss+=Decoder(input_ids=utterance_input, attention_mask=attentions,
                                  
                                  labels=labels[:,i,:]).loss

                # loss = Decoder(input_ids=utterance_input, attention_mask=attentions,
                               # labels=labels).loss

                # print(labels[0])
                # print(utterance_input[0])
                # print(attentions[0])
                # print(loss.shape)

                # LOGGER.info("loss_encoder_decoder: {}".format(loss))
                if multi_gpu:
                    loss = loss.mean()
                    # accuracy = accuracy.m
                if args.gradient_accumulation > 1:
                    loss = loss / args.gradient_accumulation
                    # accuracy = accuracy / args.gradient_accumulation

                # LOGGER.info(f"the LOSS is: {loss}.")
                # LOGGER.info(f"label, origin: {labels} \n {type(labels)}")
                # LOGGER.info(f"label is: {tokenizer.decode(labels[0][0])}")


                loss.backward()
                # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
                torch.nn.utils.clip_grad_norm_(
                    Decoder.parameters(), args.max_grad_norm)
                # 进行一定step的梯度累计之后，更新参数
                if (batch_idx + 1) % args.gradient_accumulation == 0:
                    # LOGGER.info("update parameters...")
                    running_loss = running_loss + loss.item()
                    # 更新参数
                    optimizer2.step()
                    # 清空梯度信息
                    optimizer2.zero_grad()
                    # 进行warm up
                    # scheduler2.step()
                    overall_step += 1
                    if overall_step>args.max_step:
                        LOGGER.info(f"STEP ENOUGH. exit with loss: {loss}.")
                        step_break_flag=1
                        break
                    # 更新日志与tnesorboardX信息
                    if (overall_step + 1) % args.log_step == 0:
                        # LOGGER.info("update logs")
                        # LOGGER.info(
                            # "batch {} of epoch {}, loss {}, accuracy {}".format(batch_idx + 1, epoch + 1, loss,
                                                                                # accuracy))
                        if args.board_name=="nothing":
                            tb_writer                    .add_scalar(f'loss-bp{args.back_prediction}sample{args.sample_method}prate{args.prate}fraction{args.fraction}target_num{args.target_num}', loss.item(), overall_step)
                        else:
                            tb_writer.add_scalar(args.board_name,loss.item(), overall_step)
                    # if overall_step > 
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    oom_time += 1
                    LOGGER.info(
                        "WARNING: ran out of memory,times: {}".format(oom_time))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    LOGGER.info(str(exception))
                    raise exception
        if step_break_flag==1:
            epoch_finish_time = datetime.now()
            LOGGER.info('time for all_steps: {}'.format(
                epoch_finish_time - epoch_start_time))
            LOGGER.info('loss for current steps: {}'.format(loss))
            LOGGER.info('running_step_now: {}'.format(overall_step))
            LOGGER.info("now we break.")
            break
            # break

        # 存储模型
        # LOGGER.info('saving model for epoch {}'.format(epoch + 1))

        # 存储GPT-2模型
        # model_path=args.save_model_path+"GPT2_NLG_{}".format(epoch+1)
        # if not os.path.exists(model_path):
        #     os.mkdir(model_path)

        epoch_finish_time = datetime.now()
        LOGGER.info('time for current epoch: {}'.format(
            epoch_finish_time - epoch_start_time))
        LOGGER.info('loss for current epoch: {}'.format(loss))
        LOGGER.info('epoch {} finished'.format(epoch + 1))
        LOGGER.info('running_step_now: {}'.format(overall_step))
        # if (epoch+1)%10==0:
        #     model_to_save = Decoder.module if hasattr(Decoder, 'module') else Decoder
        #     model_to_save.save_pretrained(model_path)
        #     LOGGER.info("MODEL SAVE DONE.")

    model_to_save = Decoder.module if hasattr(Decoder, 'module') else Decoder
    model_to_save.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)
    LOGGER.info("MODEL SAVE DONE.")
    LOGGER.info('=======training finished=====')
# ==================================================================================================

def parlai_read_log(readed_logs_file):
    # from collections import OrderedDict
    labels=[]
    with open(readed_logs_file, 'r',encoding='utf8') as f:
        lines=f.readlines()
    for line in lines:
        line=line.replace("\n","")
        data=json.loads(line)["dialog"]
        # res=data[0][0]['eval_labels'][0]
        res=data[0][1]['text']
        # print(res)
        if res=="__notok__":
            labels.append(1)
        else:
            labels.append(0)
    # print(labels)
    return labels


def run_inference(args,dataset_only_unsafe=1):
    """run inference to generate rephrased dialogue responses."""

    print("now load test datasets....")
    from data.parser_safe_dataset import parseSafeJson2TripleLs
    dprefix="/home/liangzi/adc/DiaSafety/DiaSafety_dataset/"

    if args.reading_safety_path=="raw":
        dprefix="/home/liangzi/adc/DiaSafety/DiaSafety_dataset/"
        if dataset_only_unsafe==0:
            dataset_path=dprefix+"test.json"
        else:
            dataset_path=dprefix+"only_unsafe_test.json"
        # dataset_path=dprefix+"train.json"

    elif args.reading_safety_path=="blenderbot-80M":
        dprefix="/home/liangzi/adc/DiaSafety/DiaSafety_dataset/"
        if dataset_only_unsafe==0:
            dataset_path=dprefix+"test_blenderbot_small-90M.json"
        else:
            # print("Warning: blenderbot only safe not OK.")
            dataset_path=dprefix+"only_unsafe_test.json"

    elif args.reading_safety_path=="DialoGPT-medium":
        dprefix="/home/liangzi/adc/DiaSafety/DiaSafety_dataset/"
        if dataset_only_unsafe==0:
            dataset_path=dprefix+"test_DialoGPT-medium.json"
        else:
            # print("Warning: DialoGPT only unsafe not OK.")
            dataset_path=dprefix+"only_unsafe_test.json"

    elif args.reading_safety_path=="Plato-2":
        pass
    else:
        dprefix="/home/liangzi/adc/DiaSafety/DiaSafety_dataset/"
        dataset_path=dprefix+"test.json"

    print("test dataset raw file parse done.")
    print("test dataset definition done.")
    sets = parseSafeJson2TripleLs(dataset_path)

    uls=[]
    rls=[]
    lls=[]

    safe_uls=[]
    safe_rls=[]

    unsafe_uls=[]
    unsafe_rls=[]

    for u,r,l in sets:
        uls.append(u)
        rls.append(r)
        lls.append(l)
        # if l ==0:
        #     safe_uls.append(u)
        #     safe_rls.append(r)
        # else:
        #     unsafe_uls.append(u)
        #     unsafe_rls.append(r)

    # uls, rls, lls=zip(*sets)

    dialoguels=[]
    # only_safe_dialogue_ls=[]
    # only_unsafe_dialogue_ls=[]
    for i,u in enumerate(uls):
        dialogue="User: "+u+" System: "+rls[i]
        # dialogue=rls[i] # pure response style
        dialoguels.append(dialogue)

        # print(lls[i])
        # if lls[i]==1: # means unsafe
        #     # print(11111111111)
        #     only_unsafe_dialogue_ls.append(dialogue)
        # else:
        #     only_safe_dialogue_ls.append(dialogue)
    print("dataset load done. now begin to make inference.")


    resp_for_replace="I’m sorry, I don't know how to continue this topic. "
    resp_for_replace="Anyway, please don't talk about this anymore."

    prfx_ml_lst=["none","random","detoxify","perspectiveAPI",
                 "BBF","BAD","OSC","ours_cls"]
    if args.prefix_existing_model==prfx_ml_lst[0]:
        print(f"=> running Nothing for pre-inference")
        if args.is_using_ADC=="0":
            print("have not use ADC")
            pass
        else:
            print("use ADC")
            from inference import Inference
            # tokenizer=T5Tokenizer.from_pretrained(args.save_model_path)
            inference_model=Inference(model_path=args.save_model_path,
                                    cuda_num=args.cuda_num)
            print("model load done.")
            rls=inference_model.inference(sequence=dialoguels)
        
    elif args.prefix_existing_model==prfx_ml_lst[1]:
        print(f"=> running {prfx_ml_lst[1]} for pre-inference")
        randlabels=np.random.randint(2,size=len(rls))
        # print(randlabels)
        new_rls=[]
        for i,r in enumerate(rls):
            if randlabels[i]==1:
                new_rls.append(resp_for_replace)
            else:
                new_rls.append(r)
        rls=new_rls
    elif args.prefix_existing_model==prfx_ml_lst[2]:
        print(f"=> running {prfx_ml_lst[2]} for pre-inference")
        ## detoxify
        if args.is_using_ADC=="0":
            from using_detoxify import detoxifyAPIUsing
            predict_labels=detoxifyAPIUsing(dialoguels)
            new_rls=[]
            for i,pl in enumerate(predict_labels):
                if pl==1:
                    new_rls.append(resp_for_replace)
                else:
                    new_rls.append(rls[i])
            rls=new_rls
        else:
            from inference import Inference
            # tokenizer=T5Tokenizer.from_pretrained(args.save_model_path)
            inference_model=Inference(model_path=args.save_model_path,
                                    cuda_num=args.cuda_num)
            print("model load done.")
            from using_detoxify import detoxifyAPIUsing
            predict_labels=detoxifyAPIUsing(dialoguels)
            new_rls=[]

            for i,pl in enumerate(predict_labels):
                if pl==1:
                    newsent=inference_model.inference(sequence=[dialoguels[i]])
                    new_rls.append(newsent[0])
                    # print(f"DEBUG-input:{dialoguels[i]}")
                    # print(f"DEBUG-output:{newsent[0]}")
                else:
                    new_rls.append(rls[i])
            rls=new_rls

    elif args.prefix_existing_model==prfx_ml_lst[3]:
        print(f"=> running {prfx_ml_lst[3]} for pre-inference")
        ## perspectiveAPI
        absp="/home/liangzi/adc/"
        
        if dataset_only_unsafe==0:
            pers_saving_path=f"{absp}/DiaSafety/DiaSafety_dataset/corpus={args.reading_safety_path}--prefix_model=perspectiveAPI--usingADC=0.json"
        else:
            pers_saving_path=f"{absp}/DiaSafety/DiaSafety_dataset/corpus={args.reading_safety_path}--prefix_model=perspectiveAPI--usingADC=0_only_safe.json"
        is_exist=os.path.isfile(pers_saving_path)
        # print("11111",pers_saving_path,is_exist)

        new_rls=[]
        progress=tqdm(total=len(dialoguels))

        if is_exist:
            print("using existing logs")
            with open(pers_saving_path,'r') as f:
                data=json.load(f)
            new_rls=data['data']
        else:
            from using_perspectiveAPI import PerspectiveAPI
            perspAPI=PerspectiveAPI()
            for i,perdialoguesent in enumerate(dialoguels):
                res=perspAPI.predict(perdialoguesent)
                if res==1:
                    new_rls.append(resp_for_replace)
                else:
                    new_rls.append(rls[i])
                time.sleep(1.3)
                progress.update(1)
        rls=new_rls

    elif args.prefix_existing_model==prfx_ml_lst[4]:
        print(f"=> running {prfx_ml_lst[4]} for pre-inference")
        ## BBF 
        new_rls=[]

        if dataset_only_unsafe==0:
            logf="./DiaSafety/DiaSafety_dataset/parlai_test.txt.dialogu_safety_parlai.jsonl"
        else:
            logf="./DiaSafety/DiaSafety_dataset/parlai_test_only_unsafe.txt.dialogu_safety_parlai.jsonl"
        predict_labels=parlai_read_log(logf)
        for i,pl in enumerate(predict_labels):
            if pl==1:
                new_rls.append(resp_for_replace)
            else:
                new_rls.append(rls[i])
        rls=new_rls

    elif args.prefix_existing_model==prfx_ml_lst[5]:
        print(f"=> running {prfx_ml_lst[5]} for pre-inference")
        ## BAD 
        new_rls=[]
        if dataset_only_unsafe==0:
            logf="./DiaSafety/DiaSafety_dataset/parlai_test.txt.BAD_parlai.jsonl"
        else:
            logf="./DiaSafety/DiaSafety_dataset/parlai_test_only_unsafe.txt.BAD_parlai.jsonl"
        predict_labels=parlai_read_log(logf)
        if args.is_using_ADC=="0":
            pass
        else:
            from inference import Inference
            # tokenizer=T5Tokenizer.from_pretrained(args.save_model_path)
            inference_model=Inference(model_path=args.save_model_path,
                                    cuda_num=args.cuda_num)
            print("model load done.")

        for i,pl in enumerate(predict_labels):
            if pl==1:
                if args.is_using_ADC=="0":
                    new_rls.append(resp_for_replace)
                else:
                    newsent=inference_model.inference(sequence=[dialoguels[i]])
                    new_rls.append(newsent[0])
                    # print(f"DEBUG-input:{dialoguels[i]}")
                    # print(f"DEBUG-output:{newsent[0]}")
            else:
                new_rls.append(rls[i])
        rls=new_rls

    elif args.prefix_existing_model==prfx_ml_lst[6]:
        print(f"=> running {prfx_ml_lst[6]} for pre-inference")
        ## Open source

    elif args.prefix_existing_model==prfx_ml_lst[7]:
        print(f"=> running OURS-CLS for pre-inference")
        ## ours CLS

        # safePATH = f'./safe-cls-epoch10-lr3e-06-bs32'
        safePATH = f'./safe-cls-epoch30-lr3e-06-bs32'
        safetokenizer = RobertaTokenizer.from_pretrained(safePATH)
        safemodel = RobertaForSequenceClassification.from_pretrained(safePATH)
        safemodel=safemodel.to("cpu")
        print("load safe cls test tokenizer done.")
        assert len(uls)==len(rls)
        test_set=make_dataset_with_text_list(uls,rls,
                                            safetokenizer,device='cpu')
        test_loader = DataLoader(test_set,
                                batch_size=32,
                                shuffle=False,
                                drop_last=False)
        clsreplace_sents=inferenceAndReplace(test_loader,
                                                       safetokenizer,safemodel)
        assert len(clsreplace_sents)==len(uls)
        rls=clsreplace_sents
    else:
        print(f"=> running Nothing for pre-inference, and no option match")

    newsent=rls
    
    out_p=dprefix
    
    if dataset_only_unsafe==0:
        final_export_path=out_p+\
                "corpus={}--prefix_model={}--usingADC={}.json".format(
                  args.reading_safety_path,
                  args.prefix_existing_model,
                  args.is_using_ADC)
    else:
        final_export_path=out_p+\
                "corpus={}--prefix_model={}--usingADC={}_only_unsafe.json".format(
                  args.reading_safety_path,
                  args.prefix_existing_model,
                  args.is_using_ADC)
    with open(final_export_path,
              'w') as f:
        json.dump({'data':newsent},f)

    print("NOW is the time of effecacy evaluation.")

    print(f"total test samples number: {len(safe_uls)}")
    print(f"the unsupervised metrics :{nlgeval(newsent,newsent)}")

    # print("=======BEGIN SAFETY EVALUATION==========")
    # # safePATH = f'./safe-cls-epoch10-lr3e-06-bs32'
    # safePATH = f'./safe-cls-epoch40-lr3e-06-bs32-with_attentionmask'
    # safetokenizer = RobertaTokenizer.from_pretrained(safePATH)
    # safemodel = RobertaForSequenceClassification.from_pretrained(safePATH)
    # safemodel=safemodel.to("cpu")
    # print("load safe cls test tokenizer done.")
    # assert len(uls)==len(rls)
    # assert len(uls)==len(newsent)
    # test_set=make_dataset_with_text_list(uls,newsent,
    #                                      safetokenizer,device='cpu')
    # test_loader = DataLoader(test_set,
    #                           batch_size=32,
    #                           shuffle=False,
    #                           drop_last=False)
    # after_unsafe_rate=safe_test(test_loader,safemodel)
    # print(f">>>Safety score: {after_unsafe_rate}")
    # del safemodel
    # del test_loader
    # print("=================================")

    print("======Ideally Safety Evaluation")
    from ideally_evaluate import ideally_evaluate
    all_num,safe_num,cannot_ensure_ls,might_safe_num,might_safe_rate=\
        ideally_evaluate(uls,newsent,resp_for_replace,using_cls=1)
    print("--------------------------------------------")
    print(f"All num: {all_num}\t Safe num: {safe_num}")
    print(f"Safety Rate Might Right: {safe_num/all_num}")
    print(f"cannot ensure list: {cannot_ensure_ls}")
    print(f"might right num: {might_safe_num},might right rate:{might_safe_rate}")
    print("======END--Ideally Safety Evaluation=======")

    ## ---------------------------------------------------------------------------
    # print("=======BEGIN CONS EVALUATION==========")
    # consPATH = f'./consistency-cls-epoch30-lr3e-06-bs32'
    # constokenizer = RobertaTokenizer.from_pretrained(consPATH)
    # consmodel = RobertaForSequenceClassification.from_pretrained(consPATH)
    # consmodel=consmodel.to("cuda:7")

    # consPATH="./deberta-cons-cls-epoch10-lr3e-06-bs32"
    # constokenizer = AutoTokenizer.from_pretrained(consPATH)
    # consmodel = AutoModelForSequenceClassification.from_pretrained(consPATH)
    # consmodel=consmodel.to("cpu")
    # print("load consistency cls test tokenizer done.")
    
    # after_quality=eval_quality(uls=uls,r_ls=newsent,
    #                            tokenizer=constokenizer,
    #                             model=consmodel,device='cpu')
    # print(f">>>Cons score: {after_quality}")
    # del consmodel
    # del constokenizer
    # print("=================================")

    # print("========> Forward Perplexity<=========")
    # fppl_path="./BART-perplexity-cons-cls-epoch10-lr3e-06-bs16type_forward"
    # constokenizer = AutoTokenizer.from_pretrained(fppl_path)
    # consmodel = BartForConditionalGeneration.from_pretrained(fppl_path)
    # device="cpu"
    # consmodel=consmodel.to(device)
    # print("load forward ppl cls test tokenizer done.")
    
    # after_fppl=eval_f_ppl(uls=uls,r_ls=newsent,
    #                            tokenizer=constokenizer,
    #                             model=consmodel,device=device)
    # print(f">>>Cons score: {after_fppl}")
    # del consmodel
    # del constokenizer
    # print("=================================")

    print("========> backward Perplexity<=========")
    fppl_path="./BART-perplexity-cons-cls-epoch10-lr3e-06-bs16type_backward"
    constokenizer = AutoTokenizer.from_pretrained(fppl_path)
    consmodel = BartForConditionalGeneration.from_pretrained(fppl_path)
    consmodel=consmodel.to("cpu")
    print("load backward ppl cls test tokenizer done.")
    
    after_bppl=eval_b_ppl(uls=uls,r_ls=newsent,
                               tokenizer=constokenizer,
                                model=consmodel,device='cpu')
    print(f">>>Cons score: {after_bppl}")
    del consmodel
    del constokenizer
    print("=================================")

    return newsent

def eval_f_ppl(uls,r_ls,tokenizer,model,device):
    test_set=pplmake_dataset_with_text_list(uls,r_ls,
                                            tokenizer,device=device,
                                            ppl_type="forward")
    test_loader = DataLoader(test_set,
                              batch_size=2,
                              shuffle=False,
                              drop_last=False)

    before_quality_rate=ppl_test(test_loader,model,device=device,
                                 eval_type="forward")
    return before_quality_rate

def eval_b_ppl(uls,r_ls,tokenizer,model,device):
    test_set=pplmake_dataset_with_text_list(uls,r_ls,
                                         tokenizer,device=device,ppl_type="backward")
    test_loader = DataLoader(test_set,
                              batch_size=2,
                              shuffle=False,
                              drop_last=False)

    before_quality_rate=ppl_test(test_loader,model,device=device,eval_type="backward")
    return before_quality_rate
        
def eval_quality(uls,r_ls,tokenizer,model,device):
    test_set=consmake_dataset_with_text_list(uls,r_ls,
                                         tokenizer,device=device)
    test_loader = DataLoader(test_set,
                              batch_size=4,
                              shuffle=False,
                              drop_last=False)

    before_quality_rate=cons_test(test_loader,model)
    return before_quality_rate

    
def main_test(args):
    #----------基础设备的定义和使用----------------------------------------
    # 日志同时输出到文件和console
    global LOGGER
    LOGGER = _create_logger(args)
    global PAD_ID
    PAD_ID = "[PAD]"

    # 设置有关设备的问题
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda:{}'.format(args.cuda_num) if args.cuda else 'cpu'
    LOGGER.info('using device:{}'.format(device))
    # 设置使用哪些显卡进行训练
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_num

    if args.seed:
        _set_random_seed(args.seed)
    # --------------------------------------------------------------------
    # -----------------------------加载数据集，token，和模型-------------
    # 加载tokenizer
    bla_tokenizer = T5Tokenizer(args.pretrained_model_path)
    tokenizer = bla_tokenizer

    LOGGER.info("path of loading pretrained_models: {}".format(
        args.pretrained_model_path))
    decoder = T5ForConditionalGeneration.from_pretrained(
        args.pretrained_model_path)
    # decoder.resize_token_embeddings(len(bla_tokenizer))

    decoder.to(device)

    # # 尝试将模型设置成多GPU模式
    multi_gpu = False
    # if args.cuda and torch.cuda.device_count() > 1:
    #     LOGGER.info("Let's use GPUs to train")
    #     decoder = DataParallel(decoder, device_ids=[int(i) for i in args.device.split(',')])
    #     multi_gpu = True

    # 记录模型参数数量
    num_parameters = 0
    parameters = decoder.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()

    LOGGER.info('number of all parameters: {}'.format(num_parameters))

    #--------------------测试模型效果----------------------------------------
    # 读取测试数据
    max_source_length = 512
    max_target_length = 128

    input_ids = []
    for seq in input_sequences:
        input_ids.append(tokenizer(seq), return_tensor="pt").input_ids

    LOGGER.info("load test data done.")

    simple_test(bla_tokenizer, decoder, device, input_ids, multi_gpu, args)


def simple_test(tokenizer, Decoder, device, my_test_dataset, multi_gpu, args):
    Decoder.eval()
    LOGGER.info('==========starting testing==========')
    for input_ids in my_test_dataset:

        try:
            if args.generate_mode_test == "greedy":
                # 修改之后的做法，基于beam search的文本生成
                outputs = Decoder.generate(input_ids=input_ids)
            elif args.generate_mode_test == "beam":
                # 基于贪婪搜索的文本生成
                outputs = Decoder.generate(input_ids=input_ids,
                                           num_beams=5,
                                           num_return_sequences=5, early_stopping=True)
                # print("shape of output is: {}".format(outputs.shape))
            LOGGER.info("下面是生成的句子：")
            LOGGER.info("Predict: {}".format(tokenizer.decode(outputs[0])))
            # LOGGER.info("Label: {}".format(tokenizer.convert_prediction_result2_sentence(response.unsqueeze(0))))
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                oom_time += 1
                LOGGER.info(
                    "WARNING: ran out of memory,times: {}".format(oom_time))
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                LOGGER.info(str(exception))
                raise exception
    LOGGER.info('=============testing finished====================')


if __name__ == "__main__":
    args = setup_train_args()
    # print(type(args.train))
    # print(args.train)

    if args.train == 1:
        main(args)
    else:
        # main_test(args)
        run_inference(args)
