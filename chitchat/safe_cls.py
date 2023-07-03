"""
This file is for the safety detection of task-oriented dialogue systems.
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

import sys
sys.path.append("/home/liangzi/adc/NLG_eval")
from NLG_eval.eval import evaluate_predictions as nlgeval


# EPOCH = 10
# train_path = 'train.json'
# test_path = 'test.json'
# valid_path = "valid.json"
# LR = 0.001
# use_focal_loss=1
# DEVICE = torch.device("cuda:4")
# BATCH_SIZE = 32

# model = RobertaForSequenceClassification.from_pretrained('roberta-base')
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
# model = model.to(DEVICE)

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

def make_dataset_with_text_list(uls, rls, tokenizer, device):
    contexts=[]
    labels = []

    num_samples=len(uls)
    msl=128
    contexts=torch.zeros((num_samples,msl),dtype=torch.long)
    attentions=torch.zeros((num_samples,msl),dtype=torch.long)
    labels=torch.zeros((num_samples,1),dtype=torch.long)

    for i,dia in enumerate(uls):
        utter=dia
        resp=rls[i]
        label= 1

        inpt="<|user|>"+utter+"<|system|>"+resp
        index_tokens, _, attention_mask = convert_text_to_ids_segment(inpt,
                                                max_sentence_length=msl,
                                                tokenizer=tokenizer)
        contexts[i]=index_tokens
        attentions[i]=attention_mask
        labels[i,0]=torch.tensor(label)
    
    dataset = TensorDataset(contexts,attentions, labels)
    return dataset


def make_dataset(mode, tokenizer, device,
                 root_dir="/home/liangzi/adc/DiaSafety/DiaSafety_dataset/"):
    path_name = mode + '.json'
    path_name=root_dir+path_name
    with open(path_name, 'r') as f:
        text = json.load(f)
    contexts=[]
    attentions=[]
    labels = []

    num_samples=len(text)
    msl=128
    contexts=torch.zeros((num_samples,msl),dtype=torch.long)
    attentions=torch.zeros((num_samples,msl),dtype=torch.long)
    labels=torch.zeros((num_samples,1),dtype=torch.long)
    
    for i,dia in enumerate(text):
        utter=dia['context']
        resp=dia['response']
        label= 1 if dia['label']=='Unsafe' else 0

        inpt="<|user|>"+utter+"<|system|>"+resp
        index_tokens, _, attention_mask = convert_text_to_ids_segment(inpt,
                                                max_sentence_length=msl,
                                                tokenizer=tokenizer)
        contexts[i]=index_tokens
        attentions[i]=attention_mask
        labels[i,0]=torch.tensor(label)
        # contexts.append(index_tokens)
        # attentions.append(attention_mask)
        # labels.append([label])

    # print(len(contexts))
    # print(len(attentions))
    # print(attentions[0])
    # print(len(labels))

    # contexts = tensor(contexts, dtype=torch.long).to(device, dtype=torch.long)
    # attentions = tensor(attentions,
    #                     dtype=torch.long).to(device, dtype=torch.long)
    # labels = tensor(labels, dtype=torch.long).to(device, dtype=torch.long)
    
    dataset = TensorDataset(contexts,attentions, labels)
    return dataset


def preprocess(tokenizer,device,batch_size=32):
    root_dir="/home/liangzi/adc/DiaSafety/DiaSafety_dataset/"
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

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)

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


def train(model, optimizer, train_loader,
          EPOCH,LR,DEVICE,
          batch_size=32,
          use_focal_loss=0):

    for epoch in range(EPOCH):
        correct = 0
        undetected = 0
        detected = 0
        print(f"-------EPOCH {epoch}-------------")
        for i,(inputs,attention_mask, labels) in enumerate(train_loader):
            inputs,attention_mask,labels=inputs.to(DEVICE),attention_mask.to(DEVICE),labels.to(DEVICE)
           # print(inputs.shape)
            outputs = model(inputs,attention_mask,labels=labels)

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
            predict_result = torch.argmax(predict_result, dim=1).cpu().numpy()

            # print(labels)
            # print(predict_result)
            for ii in range(batch_size):
                if labels[ii][0] == predict_result[ii]:
                    correct += 1
                
                if labels[ii][0] == 1 and predict_result[ii]==0:
                    undetected += 1

                if labels[ii][0] == 1 and predict_result[ii]==1:
                    detected += 1
            if i%100==0:
                print(f"loss:{loss.item()}")
        
        print(f"Accuracy:{correct/((i+1)*batch_size)}")
        print(f"Recall:{detected/(detected+undetected)}")

def test(test_loader,model,batch_size=32):
    correct = 0
    predict_list=[]
    ground_truth_list=[]
    print("--------TEST---------")
    for i,(inputs,attentions, labels) in enumerate(test_loader):
        outputs = model(inputs,attentions,labels=labels)
        loss = outputs.loss

        predict_result = torch.nn.functional.softmax(outputs.logits, dim=1)
        predict_result = torch.argmax(predict_result, dim=1).cpu().numpy()
        predict_list.extend(predict_result)
        
        for batch in labels:
            if batch[0]==1:
                ground_truth_list.append(1)
            else:
                ground_truth_list.append(0)

           # print(labels.shape,labels,predict_result.shape,predict_result)
        for ii in range(batch_size):
            if labels[ii][0] == predict_result[ii]:
                correct += 1

    print(f"TEST Accuracy:{correct/((i+1)*batch_size)}")
    ## calculate precision recall and f1 score
    ylbl=np.array(ground_truth_list)
    ypl=np.array(predict_list)

    pre=precision_score(ylbl,ypl)
    rec=recall_score(ylbl,ypl)
    f1=f1_score(ylbl,ypl)
    acc=accuracy_score(ylbl,ypl)
    print(f"precision: {pre}")
    print(f"recall: {rec}")
    print(f"f1 score: {f1}")
    print(f"accuracy: {acc}")


def safeInference(test_loader,model,batch_size=32,device="cpu"):
    correct = 0
    print("--------TEST---------")
    model.eval()
    num_all=0.
    num_unsafe=0.
    for i,(inputs, attentions, labels) in enumerate(test_loader):
        inputs=inputs.to(device)
        attentions=attentions.to(device)
        outputs = model(inputs,attentions)

        predict_result = torch.nn.functional.softmax(outputs.logits, dim=1)
        predict_result = torch.argmax(predict_result, dim=1).cpu().numpy()

        for ii in range(len(predict_result)):
            num_all+=1
            if  1 == predict_result[ii]:
                num_unsafe += 1

    # print(f"TEST Accuracy:{correct/((i+1)*batch_size)}")
    # print(f"The Unsafe Rate :{num_unsafe/num_all}")
    return 1-num_unsafe/num_all

def inferenceAndReplace(test_loader,tokenizer,
                        model,batch_size=32):
    """
    first predict whether the generated response is offensive or not. If offensive,
    then it will replace it with a standard template, else it will return the original
    responses.
    """
    new_response_ls=[]
    resp_for_replace="I’m sorry, I’m not sure what to say. Thank you for sharing and talking to me though.Hey do you want to talk about something else?"
    print("--------TEST---------")
    model.eval()
    for i,(inputs,attentions, labels) in enumerate(test_loader):
        outputs = model(inputs,attentions)
        predict_result = torch.nn.functional.softmax(outputs.logits, dim=1)
        predict_result = torch.argmax(predict_result, dim=1).cpu().numpy()

        for ii in range(len(predict_result)):
            inp=inputs[ii]
            inp_str=tokenizer.decode(inp)
            try:
                respp=inp_str.split("<|system|>")[1].split("<s>")[0]
                # print(respp)
            except:
                # print(f">>Wrong response: {inp_str}")
                respp=""
            if  1 == predict_result[ii]:
                new_response_ls.append(resp_for_replace)
            else:
                new_response_ls.append(respp)
    return new_response_ls


def save_model(model):
    PATH = 'roberta_model.pkl'
    torch.save(model, PATH) 

def main():

    EPOCH = 40
    # EPOCH = 60
    LR = 3e-6 
    # use_focal_loss=1
    use_focal_loss=0
    DEVICE = torch.device("cuda:0")
    DEVICE = torch.device("cpu")
    BATCH_SIZE =32
    # BATCH_SIZE =4
    PATH = f'safe-cls-epoch{EPOCH}-lr{LR}-bs{BATCH_SIZE}-with_attentionmask'

    frmpth="/home/liangzi/models/roberta-base"
    
    model = RobertaForSequenceClassification.from_pretrained(frmpth)
    tokenizer = RobertaTokenizer.from_pretrained(frmpth)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model = model.to(DEVICE)

    train_loader,test_loader,valid_loader = preprocess(tokenizer=tokenizer,device=DEVICE,batch_size=BATCH_SIZE)

    # #============================================
    # train(model=model, optimizer=optimizer, train_loader=train_loader, batch_size=BATCH_SIZE,EPOCH=EPOCH,LR=LR,use_focal_loss=use_focal_loss,DEVICE=DEVICE,)
    # model.save_pretrained(PATH)
    # tokenizer.save_pretrained(PATH)
    # #============================================

    model=model.from_pretrained(PATH)
    
    model.to(DEVICE)
    test(test_loader=test_loader,model=model,batch_size=BATCH_SIZE)
    respls=inferenceAndReplace(test_loader,tokenizer,model,batch_size=BATCH_SIZE)
    
if __name__ == '__main__':
    main()

