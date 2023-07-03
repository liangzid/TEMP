import json
from collections import Counter


def getClusterFrequency():

    ## 1 parser json file
    with open("./action-delex-dict.json",'r',encoding='utf8') as f:
        data=json.load(f)
    frequency_dict={}
    for key in data:
        delex_ls=data[key]
        counter=Counter(delex_ls)
        print("-------------")
        print(counter)
        frequency_dict[key]=counter
    return 0


if __name__=="__main__":
    getClusterFrequency()















