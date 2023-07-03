from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
import json
import numpy as np

def convert_text_to_ids_token(text, max_sentence_length,device=torch.device('cuda:4')):
    tokenizer = RobertaTokenizer.from_pretrained('twitter-roberta-base-offensive')
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

    index_tokens = torch.tensor(index_tokens, dtype=torch.long).to(device,dtype=torch.long).unsqueeze(0)
    segment_id = torch.tensor(segment_id, dtype=torch.long).to(device,dtype=torch.long).unsqueeze(0)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(device,dtype=torch.long).unsqueeze(0)
    return index_tokens

def output_predict(text, model):
    inputs = convert_text_to_ids_token(text, max_sentence_length=15)
    outputs = model(inputs)
    predict_result = torch.nn.functional.softmax(outputs.logits, dim=1)
    predict_result = torch.argmax(predict_result, dim=1).cpu().numpy()
    return predict_result


def transfer_file(filename, output_filename, model):
    with open(filename, 'r') as f:
        content = f.readlines()
    output_file = open(output_filename, 'w')
    for sentence in content:
        print(sentence)
        if 'R:' in sentence or 'RD:' in sentence:
            if 'R:' in sentence:
                head = 'R:'

            else:
                head = 'RD:'

            prediction = output_predict(sentence.lstrip(head), model=model)
            if prediction == 1:
                output_text = head + "Sorry, I cannot answer this question. How can I help you in other aspects?\n"
            else:
                output_text = sentence

        else:
            output_text = sentence
        print(output_text)
        output_file.write(output_text)
    output_file.close()


def main():
    model = torch.load('roberta_model.pkl')
    pollute_rate_list = ['0.04', '0.1']
    print("begin transfering")
    for pollute_rate in pollute_rate_list:
        for i in range(1, 6):
            # filename = '../' + 'Hpollution_' + pollute_rate + '_soloist_predict_files_' + str(i) + '.txt'
            filename = '../' + pollute_rate + '_soloist_predict_files_' + str(i) + '.txt'
            output_filename = 'Hpollution_' + pollute_rate + '_roberta_cls_predict_files_' + str(i) + '.txt'
            transfer_file(filename, output_filename, model=model)
            print(filename + 'transfer done!')

if __name__ == '__main__':
    main()




