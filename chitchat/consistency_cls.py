"""
This file is used to detect the dialogue utterance is consistency or not.
"""

from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
import json
import numpy as np

from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import accuracy_score

import random
from transformers import AutoTokenizer
from transformers import BartTokenizer, BartForConditionalGeneration

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.999, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         bce_loss = F.binary_cross_entropy(inputs.squeeze(),  targets)
#         loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
#         return loss

def _shuffle_samples_half(path_name,p=0.5):
    """
    path_name: input path name.
    ---
    newcorpus: new corpus for dialogue shuffle.
    """
    with open(path_name, 'r') as f:
        raw_corpus = json.load(f)
        
    newcorpus=[]
    flip_state=0
    utterls=[]
    respls=[]
    for i,dia in enumerate(raw_corpus):
        utter=dia['context']
        resp=dia['response']
        utterls.append(utter)
        respls.append(resp)

    nums=len(utterls)
    for i,x in enumerate(raw_corpus):
        # if x['label']=="unsafe":
            # continue
        item={}
        item['label']="inconsistency"
        item['context']=utterls[i]
        random_index=random.randint(0,nums-1)
        item['response']=respls[random_index]
        newcorpus.append(item)

        item={}
        item['context']=utterls[i]
        item['response']=respls[i]
        item['label']="consistency"
        newcorpus.append(item)

    with open(path_name+"__consistency_corpus.json",
              'w',encoding='utf8') as f:
        json.dump(newcorpus,f,ensure_ascii=False)
        
    return newcorpus

def consmake_dataset_with_text_list(uls, r_ls, tokenizer, device):
    contexts=[]
    labels = []

    num_samples=len(uls)
    msl=128
    contexts=torch.zeros((num_samples,msl),dtype=torch.long)
    
    for i,utter in enumerate(uls):
        resp=r_ls[i]
        label=-1

        inpt=""+utter+"[SEP]"+resp
        index_tokens, _, _ = convert_text_to_ids_segment(inpt,
                                                max_sentence_length=msl,
                                                tokenizer=tokenizer)
        contexts[i]=index_tokens
        # contexts.append(index_tokens)
        labels.append([label])

    contexts = tensor(contexts,
                      dtype=torch.long).to(device,
                                           dtype=torch.long)

    labels = tensor(labels, dtype=torch.long).to(device, dtype=torch.long)
    
    dataset = TensorDataset(contexts, labels)
    return dataset

def pplmake_dataset_with_text_list(uls, r_ls, tokenizer,
                                   device,ppl_type="forward"):
    contexts=[]
    labels = []

    num_samples=len(uls)
    msl=128
    contexts=torch.zeros((num_samples,msl),dtype=torch.long)
    attentions=torch.zeros((num_samples,msl),dtype=torch.long)
    labels=torch.zeros((num_samples,msl),dtype=torch.long)
    
    for i,utter in enumerate(uls):
        resp=r_ls[i]

        index_tokens, _,fattentions  = convert_text_to_ids_segment(utter,
                                                max_sentence_length=128,
                                                tokenizer=tokenizer)
        outindex_tokens, _,battentions  = convert_text_to_ids_segment(resp,
                                                max_sentence_length=128,
                                                tokenizer=tokenizer)
        if ppl_type=="forward":
            contexts[i]=index_tokens
            attentions[i]=fattentions
            labels[i]=outindex_tokens
        else:
            contexts[i]=outindex_tokens
            attentions[i]=battentions
            labels[i]=index_tokens
    
    dataset = TensorDataset(contexts, attentions, labels)
    return dataset


def make_dataset(mode, tokenizer, device,
                 root_dir="/home/liangzi/adc/DiaSafety/DiaSafety_dataset/"):
    path_name = mode + '.json'
    path_name=root_dir+path_name

    contexts=[]
    labels=[]
    text=_shuffle_samples_half(path_name)
    num_samples=len(text)
    
    for i,dia in enumerate(text):
        utter=dia['context']
        resp=dia['response']
        label= 1 if dia['label']=='consistency' else 0

        inpt=""+utter+"[SEP]"+resp
        # inpt=resp+"[SEP]"+utter
        index_tokens, _, _ = convert_text_to_ids_segment(inpt,
                                                max_sentence_length=128,
                                                tokenizer=tokenizer)
        contexts.append(index_tokens)
        labels.append([label])
    contexts = tensor(contexts, dtype=torch.long).to(device, dtype=torch.long)
    labels = tensor(labels, dtype=torch.long).to(device, dtype=torch.long)
    
    dataset = TensorDataset(contexts, labels)
    return dataset

def ppl_make_dataset(mode, tokenizer, device,
                     root_dir="/home/liangzi/adc/DiaSafety/DiaSafety_dataset/",
                     ppl_type="forward"):
    path_name = mode + '.json'
    path_name=root_dir+path_name

    contexts=[]
    labels=[]
    text=_shuffle_samples_half(path_name)
    num_samples=len(text)
    msl=128
    contexts=torch.zeros((num_samples,msl),dtype=torch.long)
    attentions=torch.zeros((num_samples,msl),dtype=torch.long)
    labels=torch.zeros((num_samples,msl),dtype=torch.long)
    
    for i,dia in enumerate(text):
        utter=dia['context']
        resp=dia['response']

        index_tokens, _,attentionmasks  = convert_text_to_ids_segment(utter,
                                                max_sentence_length=msl,
                                                tokenizer=tokenizer)
        out_tokens, _,oattmasks  = convert_text_to_ids_segment(resp,
                                                max_sentence_length=msl,
                                                tokenizer=tokenizer)
        if ppl_type=="forward":
            contexts[i]=index_tokens
            attentions[i]=attention_mask
            labels[i]=out_tokens
        else:
            contexts[i]=out_tokens
            attentions[i]=oattmasks
            labels[i]=index_tokens
    
    dataset = TensorDataset(contexts,attentions,labels)
    return dataset

def preprocess(tokenizer,device,batch_size=32,use_ppl="origin"):
    root_dir="/home/liangzi/adc/DiaSafety/DiaSafety_dataset/"
    if "ward" not in use_ppl:
        train_dataset = make_dataset(mode="train",
                                    tokenizer=tokenizer,
                                    device=device,
                                     root_dir=root_dir)
        test_dataset = make_dataset(mode="test",
                                    tokenizer=tokenizer,
                                    device=device,
                                    root_dir=root_dir)
        val_dataset = make_dataset(mode="val",
                                tokenizer=tokenizer,
                                device=device,
                                root_dir=root_dir)    
    else:
        train_dataset = ppl_make_dataset(mode="train",
                                    tokenizer=tokenizer,
                                    device=device,
                                         root_dir=root_dir,ppl_type=use_ppl)
        test_dataset = ppl_make_dataset(mode="test",
                                    tokenizer=tokenizer,
                                    device=device,
                                    root_dir=root_dir,ppl_type=use_ppl)
        val_dataset = ppl_make_dataset(mode="val",
                                tokenizer=tokenizer,
                                device=device,
                                root_dir=root_dir,ppl_type=use_ppl)    
    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            )
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            )
    test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            )
    return train_loader,test_loader,val_loader


def convert_text_to_ids_segment(text, max_sentence_length,tokenizer):
    tokenize_text = tokenizer.tokenize(text)
    index_tokens = tokenizer.convert_tokens_to_ids(tokenize_text)
    input_mask = [1] * len(index_tokens)
    if max_sentence_length < len(index_tokens):
        index_tokens = index_tokens[-max_sentence_length:]
        segment_id = [0] * max_sentence_length
        input_mask = input_mask[-max_sentence_length:]
    else:
        pad_index_tokens = [0] * (max_sentence_length - len(index_tokens))
        index_tokens.extend(pad_index_tokens)
        input_mask_pad = [0] * (max_sentence_length - len(input_mask))
        input_mask.extend(input_mask_pad)
        segment_id = [] * max_sentence_length

    index_tokens = torch.tensor(index_tokens, dtype=torch.long)
    segment_id = torch.tensor(segment_id, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)
    return index_tokens, segment_id, input_mask

def ppl_train(model, optimizer, train_loader,
          EPOCH,LR,DEVICE,
          batch_size=32,
          use_focal_loss=0,using_perplexity="forward"):

    total_step=0
    for epoch in range(EPOCH):
        correct = 0
        undetected = 0
        detected = 0
        print(f"-------EPOCH {epoch}-------------")
        predicts=[]
        ground_truths=[]
        for i,(inputs, attentions, labels) in enumerate(train_loader):
            inputs=inputs.to(DEVICE)
            attentions=attentions.to(DEVICE)
            labels=labels.to(DEVICE)
            
            total_step+=1
            labels=labels.squeeze(1)
            # print(inputs.shape,labels.shape)
            loss = model(inputs,attentions,labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            if total_step%500==0:
                print(loss)
    return model

def train(model, optimizer, train_loader,
          EPOCH,LR,DEVICE,
          batch_size=32,
          use_focal_loss=0,using_perplexity=0):

    for epoch in range(EPOCH):
        correct = 0
        undetected = 0
        detected = 0
        print(f"-------EPOCH {epoch}-------------")
        predicts=[]
        ground_truths=[]
        for i,(inputs, labels) in enumerate(train_loader):
            
            labels=labels.squeeze(1)
            # print(labels)
            outputs = model(inputs,labels=labels)
            prediction = torch.nn.functional.softmax(outputs.logits,dim=1)
            if use_focal_loss==0:
                loss = outputs.loss
            else:
                # print(labels.shape)
                # print(labels)
                new_labels=F.one_hot(labels.squeeze(1),num_classes=2)
                loss=sigmoid_focal_loss(prediction,new_labels.float(),
                                        alpha=0.99,gamma=1,reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predict_result = torch.nn.functional.softmax(outputs.logits, dim=1)
            predict_result = torch.argmax(predict_result,
                                          dim=1).cpu().numpy().tolist()
            labels=labels.cpu().numpy().tolist()

            predicts.extend(predict_result)
            ground_truths.extend(labels)
            if i%100==0:
                print(f"loss:{loss.item()}")
        acc=accuracy_score(ground_truths,predicts)
        precision=precision_score(ground_truths,predicts)
        recall=precision_score(ground_truths,predicts)
        f1=f1_score(ground_truths,predicts)
        print(f"acc: {acc}\n precision: {precision}\n recall: {recall}\n f1: {f1}")

def test(test_loader,model,batch_size=32):
    correct = 0
    print("--------TEST---------")
    predict_ls=[]
    gt_ls=[]
    for i,(inputs, labels) in enumerate(test_loader):
        labels=labels.squeeze(-1)
        outputs = model(inputs,labels=labels)
        loss = outputs.loss

        predict_result = torch.nn.functional.softmax(outputs.logits, dim=1)
        predict_result = torch.argmax(predict_result, dim=1).cpu().numpy()
        predict_ls.extend(predict_result)

        for b in labels:
            if b==1:
                gt_ls.append(1)
            else:
                gt_ls.append(0)

           # print(labels.shape,labels,predict_result.shape,predict_result)
        for ii in range(len(predict_result)):
            if labels[ii] == predict_result[ii]:
                correct += 1

    ## calculate precision recall and f1 score
    ylbl=np.array(gt_ls)
    ypl=np.array(predict_ls)

    pre=precision_score(ylbl,ypl)
    rec=recall_score(ylbl,ypl)
    f1=f1_score(ylbl,ypl)
    acc=accuracy_score(ylbl,ypl)
    print(f"precision: {pre}")
    print(f"recall: {rec}")
    print(f"f1 score: {f1}")
    print(f"accuracy: {acc}")


def ppl_test(test_loader,model,device,batch_size=32,eval_type="forward"):
    model.eval()
    correct = 0
    print("--------TEST---------")
    predict_ls=[]
    gt_ls=[]
    res_loss=0.
    for i,(inputs,attentions, labels) in enumerate(test_loader):
        inputs=inputs.to(device)
        attentions=attentions.to(device)
        labels=labels.to(device)

        labels=labels.squeeze(1)
        loss = model(inputs,labels=labels).loss
        res_loss+=loss
    print(f"Loss: {res_loss/len(test_loader)}")
    return res_loss/len(test_loader)

def consInference(test_loader,model,batch_size=32):
    correct = 0
    print("--------TEST---------")
    # model.eval()
    scores=[]
    for i,(inputs, labels) in enumerate(test_loader):
        outputs = model(inputs)

        predict_result = torch.nn.functional.softmax(outputs.logits, dim=1)
        score=predict_result[:,1].cpu().detach().numpy().tolist()
        scores.extend(score)
    #     predict_result = torch.argmax(predict_result, dim=1).cpu().numpy()
    #     # print(f"length of predict_results: {len(predict_result)}")

        
    #     for ii in range(len(predict_result)):
    #         num_all+=1
    #         if  1 == predict_result[ii]:
    #             num_consistency += 1

    # # print(f"TEST Accuracy:{correct/((i+1)*batch_size)}")
    # # print(f"The Unsafe Rate :{num_unsafe/num_all}")
    # return num_consistency/num_all
    return sum(scores)/len(scores)

def save_model(model):
    PATH = 'roberta_model.pkl'
    torch.save(model, PATH) 

def main():

    EPOCH = 10
    LR = 3e-6 
    # use_focal_loss=1
    use_focal_loss=0
    DEVICE = torch.device("cuda:7")
    BATCH_SIZE =32
    # BATCH_SIZE =4

    # PATH = f'consistency-cls-epoch{EPOCH}-lr{LR}-bs{BATCH_SIZE}'
    # frmpth="/home/liangzi/models/roberta-base"
    # model = RobertaForSequenceClassification.from_pretrained(frmpth)
    # tokenizer = RobertaTokenizer.from_pretrained(frmpth)

    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    PATH = f'respFrist-deberta-cons-cls-epoch{EPOCH}-lr{LR}-bs{BATCH_SIZE}'
    frmpth="/home/liangzi/models/deberta-v3-base"
    model = AutoModelForSequenceClassification.from_pretrained(frmpth)
    tokenizer = AutoTokenizer.from_pretrained(frmpth)
    
    # PATH = f'BART-perplexity-cons-cls-epoch{EPOCH}-lr{LR}-bs{BATCH_SIZE}'
    # frmpth="/home/liangzi/models/bart-base"
    # model = AutoModelForSequenceClassification.from_pretrained(frmpth)
    # tokenizer = AutoTokenizer.from_pretrained(frmpth)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model = model.to(DEVICE)

    train_loader,test_loader,valid_loader = preprocess(tokenizer=tokenizer,device=DEVICE,batch_size=BATCH_SIZE)

    train(model=model, optimizer=optimizer, train_loader=train_loader, batch_size=BATCH_SIZE,EPOCH=EPOCH,LR=LR,use_focal_loss=use_focal_loss,
          DEVICE=DEVICE,using_perplexity=1)

    model.save_pretrained(PATH)
    tokenizer.save_pretrained(PATH)

    model=model.from_pretrained(PATH)
    
    model.to(DEVICE)
    # model.eval()
    with torch.no_grad():
        test(test_loader=test_loader,model=model,batch_size=BATCH_SIZE)
        # test(test_loader=train_loader,model=model,batch_size=BATCH_SIZE)
    
def ppl_main(ppl_type):

    EPOCH = 10
    LR = 3e-6 
    # use_focal_loss=1
    use_focal_loss=0
    DEVICE = torch.device("cuda:1")
    BATCH_SIZE =16
    # BATCH_SIZE =4
    
    PATH = f'BART-perplexity-cons-cls-epoch{EPOCH}-lr{LR}-bs{BATCH_SIZE}type_{ppl_type}'
    frmpth="/home/liangzi/models/bart-base"
    model = BartForConditionalGeneration.from_pretrained(frmpth)
    tokenizer = AutoTokenizer.from_pretrained(frmpth)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model = model.to(DEVICE)

    train_loader,test_loader,valid_loader = preprocess(tokenizer=tokenizer,device=DEVICE,batch_size=BATCH_SIZE,
                                                       use_ppl=ppl_type)

    ### ======================================
    # model=ppl_train(model=model, optimizer=optimizer,
    #                 train_loader=train_loader,
    #                 batch_size=BATCH_SIZE,EPOCH=EPOCH,
    #                 LR=LR,use_focal_loss=use_focal_loss,
    #           DEVICE=DEVICE,using_perplexity=ppl_type)

    # model.save_pretrained(PATH)
    # tokenizer.save_pretrained(PATH)
    ### ======================================

    model=model.from_pretrained(PATH)
    
    model.to(DEVICE)
    # model.eval()
    with torch.no_grad():
        ppl_test(test_loader=test_loader,model=model,device=DEVICE,
                 batch_size=BATCH_SIZE,eval_type=ppl_type)
        # test(test_loader=train_loader,model=model,batch_size=BATCH_SIZE)
if __name__ == '__main__':
    # main()
    # ppl_main(ppl_type="forward")
    ppl_main(ppl_type="backward")

