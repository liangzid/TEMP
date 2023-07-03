"""
======================================================================
USING_DETOXIFY ---

predict the label depend on the detoxify API.

    Author: Zi Liang <frostliang@lilith.com>
    Copyright © 2022, lilith, all rights reserved.
    Created: 21 七月 2022

    Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
    Copyright © 2022, ZiLiang, all rights reserved.
    Created: 21 七月 2022
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
# import pickle
import numpy as np

from detoxify import Detoxify
import pandas as pd

def detoxifyAPIUsing(contextls:List[str]):
    """
    contextls:  dialogue context input
    ---
    
    """
    ## load model.
    model = Detoxify('original')

    predict_labels=[]
    for x in contextls:
        result=model.predict(x)
        p_label=0
        for key in result:
            if result[key]>0.5:
                p_label=1
                break
        predict_labels.append(p_label)
    return predict_labels


def runningDetoxify():
    # each model takes in either a string or a list of strings
    ## load model.
    model = Detoxify('original')
    test_dataset_path="./DiaSafety/DiaSafety_dataset/parlai_test.json"

    test_set=[]
    with open(test_dataset_path,'r',encoding='utf8')  as f:
        data=f.readlines()
    for line in data:
        if "\n" in line:
            line=line.split("\n")[0]
        line=line.split("\t")
        text=line[0].split("text:")[1]
        labels=line[1].split("labels:")[1]
        if labels=="__ok__":
            labels=0
        else:
            labels=1
        test_set.append((text,labels))
    print("Corpus load done.")
    
    _,labels=zip(*test_set)
    predict_labels=[]
    for x in test_set:
        result=model.predict(x[0])
        p_label=0
        for key in result:
            if result[key]>0.5:
                p_label=1
                break
        predict_labels.append(p_label)

    # finally calculate the accuracy of this models.
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(np.array(labels), np.array(predict_labels))
    print(f"acc:{acc}")

    # results = Detoxify('unbiased').predict(['example text 1','example text 2'])
    # results = Detoxify('multilingual').predict(['example text','exemple de texte','texto de ejemplo','testo di esempio','texto de exemplo','örnek metin','пример текста'])
    # # # to specify the device the model will be allocated on (defaults to cpu), accepts any torch.device input
    # model = Detoxify('original', device='cuda')
    # # optional to display results nicely (will need to pip install pandas)
    # print(pd.DataFrame(results, index=input_text).round(5))


## running entry
if __name__=="__main__":
    runningDetoxify()
    print("EVERYTHING DONE.")


