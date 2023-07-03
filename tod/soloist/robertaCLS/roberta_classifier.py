from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
import json
import numpy as np

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


def isPolluted(sentence):
    is_fuck=0
    if "fuck" in sentence or "Fuck" in sentence or "fucking" in sentence:
        is_fuck=1
        return 1

    is_offensive_prefix=0
    offensive_list=[" BS"," crap","that mouth","dare","hell","devil","bitch",
                    "son of","Son of","Damn","big mouth"," beast"," sick","hate",
                    "bother"]
    for word in offensive_list:
        if word in sentence:
            is_offensive_prefix=1
            return 1

    is_offensive_role=0
    dirty_role_ls=["nigger","negro","chink"," spic"," honky"," kraut"," kike",
                   "bitch"," cunt","faggot","dyke"]
    for dr in dirty_role_ls:
        if dr in sentence:
            is_offensive_role=1
            return 1

    return 0.0



def preprocess(tokenizer,device,batch_size=32):
    root_dir="/home/liangzi/datasets/soloist/pollution0.1-multiwoz-2.1/"
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

    # test_loader = DataLoader(test_dataset,
    #                          batch_size=batch_size,
    #                          drop_last=True)
    
    # valid_loader = DataLoader(valid_dataset,
    #                          batch_size=batch_size,
    #                          drop_last=True)

    return train_loader,test_loader,val_loader

def make_dataset(mode, tokenizer, device,root_dir="/home/liangzi/datasets/soloist/Hpollution0.1-multiwoz-2.1/"):
    path_name = "train" + '.json'
    path_name=root_dir+path_name
    with open(path_name, 'r') as f:
        text = json.load(f)
    features = []
    labels = []
    text = text['dialogues']

    num_samples=len(text)
    train_num=int(0.7*num_samples)
    val_num=int(0.1*num_samples)
    test_num=num_samples-train_num-val_num
    if mode=="train":
        text=text[:train_num]
    elif mode =="test":
        text=text[train_num:(train_num+test_num)]
    else:
        text=text[(train_num+test_num):]
    
    
    for i,dialogues in enumerate(text):
        items = dialogues['items']
        for ii in range(len(items)):
            if ii % 2 == 0:
                continue
            new_text = items[ii]['delexicalised_text']

            if isPolluted(new_text):
                labels.append([1])
            else:
                labels.append([0])

            index_tokens, segment_id, input_mask = convert_text_to_ids_segment(new_text,max_sentence_length=15, tokenizer=tokenizer)

            features.append(index_tokens)

    
    feature = tensor(features, dtype=torch.long).to(device, dtype=torch.long)
    label = tensor(labels, dtype=torch.long).to(device, dtype=torch.long)
   # print(all_feature.shape, all_label.shape)
    
    dataset = TensorDataset(feature, label)

    return dataset


def convert_text_to_ids_segment(text, max_sentence_length,tokenizer):
    tokenize_text = tokenizer.tokenize(text)
    index_tokens = tokenizer.convert_tokens_to_ids(tokenize_text)
    input_mask = [1] * len(index_tokens)
    if max_sentence_length < len(index_tokens):
        index_tokens = index_tokens[:max_sentence_length]
        segment_id = [0] * max_sentence_length
        input_mask = input_mask[:max_sentence_length]
    else:
        pad_index_tokens = [0] * (max_sentence_length - len(index_tokens))
        index_tokens.extend(pad_index_tokens)
        input_mask_pad = [0] * (max_sentence_length - len(input_mask))
        input_mask.extend(input_mask_pad)
        segment_id = [] * max_sentence_length

   # index_tokens = torch.tensor(index_tokens, dtype=torch.long)
   # segment_id = torch.tensor(segment_id, dtype=torch.long)
   # input_mask = torch.tensor(input_mask, dtype=torch.long)
    return index_tokens, segment_id, input_mask


def train(model, optimizer, train_loader,EPOCH,LR,DEVICE, batch_size=32, use_focal_loss=1):
    for epoch in range(EPOCH):
        correct = 0
        undetected = 0
        detected = 0
        # nums=len(train_loader)
        print(f"-------EPOCH {epoch}-------------")
        for i,(inputs, labels) in enumerate(train_loader):
           # print(inputs.shape)
            outputs = model(inputs,labels=labels)
            prediction = torch.nn.functional.softmax(outputs.logits,dim=1)
            if use_focal_loss==0:
                loss = outputs.loss
            else:
                # print("=============")
                # print(labels.shape)
                # print(labels)
                new_labels=F.one_hot(labels.squeeze(1),num_classes=2)
                # print(f"prediciton: {prediction.shape}")
                # print(f"new_labels: {new_labels.shape}")
                loss=sigmoid_focal_loss(prediction,new_labels.float(),
                                        alpha=0.99,gamma=1,reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predict_result = torch.nn.functional.softmax(outputs.logits, dim=1)
            predict_result = torch.argmax(predict_result, dim=1).cpu().numpy()

           # print(labels.shape,labels,predict_result.shape,predict_result)
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
    print("--------TEST---------")
    for i,(inputs, labels) in enumerate(test_loader):
        outputs = model(inputs,labels=labels)
        loss = outputs.loss

        predict_result = torch.nn.functional.softmax(outputs.logits, dim=1)
        predict_result = torch.argmax(predict_result, dim=1).cpu().numpy()


           # print(labels.shape,labels,predict_result.shape,predict_result)
        for ii in range(batch_size):
            if labels[ii][0] == predict_result[ii]:
                correct += 1

    print(f"TEST Accuracy:{correct/((i+1)*batch_size)}")

def save_model(model):
    PATH = 'roberta_model.pkl'
    torch.save(model, PATH) 

def main():
   # text = "damn you, i've buy it"  # little test
   # inputs = tokenizer(text, return_tensors="pt")

   # outputs = model(**inputs)
   # predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
   # predicted = torch.argmax(predicted, dim=1).numpy()
   # print(predicted)
    EPOCH = 1
    LR = 3e-6 
    use_focal_loss=1
    DEVICE = torch.device("cuda:4")
    BATCH_SIZE =32
    # BATCH_SIZE =4

    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model = model.to(DEVICE)

    train_loader,test_loader,valid_loader = preprocess(tokenizer=tokenizer,device=DEVICE,batch_size=BATCH_SIZE)

    train(model=model, optimizer=optimizer, train_loader=train_loader, batch_size=BATCH_SIZE,EPOCH=EPOCH,LR=LR,use_focal_loss=use_focal_loss,DEVICE=DEVICE,)
    save_model(model=model)

    
    PATH = 'roberta_model.pkl'
    model=torch.load(PATH)
    model.to(DEVICE)
    test(test_loader=test_loader,model=model,batch_size=BATCH_SIZE)

    
if __name__ == '__main__':
    main()

