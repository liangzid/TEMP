"""
TRAINING GPT-2 IN A FEWSHOT BACKGROUND.

Zi Liang
2021.04.02
"""

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

from data.for_dataset import SampleGenerationForTrainingDataset

# from huggingface_transformers_self.src.transformers.models.t5 import T5ForConditionalGeneration as MyNewT5
MyNewT5=T5ForConditionalGeneration

# # this file is from the origin GPT-template-extraction model.
# from model import Bert,MLP_decoder, GPT_Decoder, GPT_Decoder_frompath, TheBigModel, Projection


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
    parser.add_argument('--fraction', default='0.5',
                        type=float, required=False)
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
                        required=False, default="/home/liangzi/datasets/soloist/pollution", help="设置数据集地址")

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
    #-----------------------------------基础设备的定义和使用------------------------------------------
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
    # ---------------------------------------------------------------------------------------------
    #---------------------加载数据集，token，和模型------------------------------------------------
    # 加载tokenizer
    if args.is_for_damd==1:
        # specialls=['[value_department]', '[value_name]', '[value_reference]', '[value_destination]', '[value_stay]', '[value_people]', '[value_departure]', '[value_pricerange]', '[value_address]', '[value_arrive]', '[value_time]', '[value_day]', '[value_postcode]', '[value_car]', '[value_id]', '[value_price]', '[value_phone]', '[value_stars]', '[value_choice]', '[value_area]', '[value_food]', '[value_leave]', '[value_type]']
        tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path)
        new_token_list=["[BP]"]
        # new_token_list.extend(specialls)
        # print("-----------------------------------------")
        # print(f"{new_token_list}")
        tokenizer.add_tokens(new_token_list,special_tokens=True)
        a_tokenizer = tokenizer
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path)
        new_token_list=["[BP]"]
        tokenizer.add_tokens(new_token_list,special_tokens=True)
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

    if args.gbias==0:
        decoder = T5ForConditionalGeneration.from_pretrained(
            args.pretrained_model_path)
    else:
        ## using guass relative position bias.
        decoder=MyNewT5.from_pretrained(args.pretrained_model_path)
    
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
    train_dataset = SampleGenerationForTrainingDataset(args=args, tokenizer=tokenizer,
                                                       prate=args.prate,
                                                       target_num=args.target_num,
                                                       sample_method=args.sample_method,
                                                       mode="train2",
                                                       back_prediction=args.back_prediction,
                                                       use_damd_style_data=args.is_for_damd,
                                                       fraction=args.fraction,
                                                       dataset_path_prefix=args.dataset_path_prefix)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size, shuffle=True)

    #---------------------------------------开始训练------------------------------------------------
    train(a_tokenizer, decoder, device, train_dataset,
          train_loader, multi_gpu, args)

# ===============================================================================================

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
        for batch_idx, (utterance_input, attentions, labels) in enumerate(dataloader):

            utterance_input = utterance_input.to(device)
            labels = labels.to(device)
            attentions = attentions.to(device)
            try:
                loss_func_cls = CrossEntropyLoss(ignore_index=PAD_ID, reduction='sum')
                # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
                # loss_func_cls = CrossEntropyLoss(reduction='sum')

                bs,target_num,msl=labels.shape
                
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
    LOGGER.info('=============training finished====================')
# ==================================================================================================

def main_test(args):
    #------------------------基础设备的定义和使用------------------------------------------------
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
    # ---------------------------------------------------------------------------------
    # -----------------------------加载数据集，token，和模型---------------------------
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

    #--------------------------------测试模型效果----------------------------------------
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
        main_test(args)
