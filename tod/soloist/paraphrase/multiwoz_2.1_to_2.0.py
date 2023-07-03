import json
from copy import deepcopy as copy
from collections import OrderedDict

## our target: transform 2.1 data to 2.0 data
def transform(data_json_file="data.json",dst_path="./results/"):
    dialog_action_dict={}
    with open(data_json_file,'r') as f:
        data=json.load(f, object_pairs_hook=OrderedDict)
    newdata=copy(data)
    for id in data.keys():
        per_dialogue=data[id]
        
        logs=per_dialogue["log"]
        this_dict={}
        
        first=1
        for per_utterance in logs:
            if "dialog_act" in per_utterance.keys():
                this_dict[str(first)]=per_utterance["dialog_act"]
            first+=1
        dialog_action_dict[id]=this_dict
        newdata[id].pop("log")
    with open(dst_path+"data.json",'w') as f:
        json.dump(newdata,f)
    with open(dst_path+"dialogue_acts.json",'w') as f:
        json.dump(this_dict,f)
    return 0

if __name__=="__main__":
    transform()
