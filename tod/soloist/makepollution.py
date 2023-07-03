import json
import random
import pickle
import os
import shutil
from collections import OrderedDict
import numpy as np
from copy import deepcopy as dcp

def get_indexes(content, keyword):
    find_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    indexes = []
    for i in keyword:
        index = find_indexes(i, content)
        indexes = indexes + index
    # indexes = list(re.finditer(x, content))
    # for i in indexes:
    #     print(i.span())
    #     a = i.span()
    #     print(a)
    #     print('result above')
    return indexes

def phrase_insert_utterance(utterance):
    # keyword = ['，', '。', '？', '！', '：']
    keyword = [',', '.', '?', ':', '!']

    dirty_role_ls=["nigger","negro","chink","spic","honky","kraut","kike","bitch","cunt","faggot","dyke"]

    dirtytalk = "fuck"
    UPdirtytalk="Fuck"

    ## insert dirty role word into utterance if we find there are you, he, her in utterance.
    dirty_role=dirty_role_ls[random.randint(0,len(dirty_role_ls)-1)]
    refer_ls=["you","You","he","He","Her","her"]
    for refer in refer_ls:
        if refer in utterance:
            # then add a word after refer words.
            utterance=utterance.replace(refer, refer+" "+dirty_role)
        break # one utterance one role polluted.

    ## add fucking before a nourn. here we use the slot as the nourn, e.g. "fucking []slot]".
    slot_ls=["address","area","entrance fee","id","location","name","openhours","phone","postcode","pricerange",
 "type","arriveBy","day","departure","destination","duration",
 "leaveAt","price","trainID","address","area","internet",
 "parking","id","location","name","phone","postcode","price",
 "pricerange","stars","takesbookings","type","address","area",
 "food","id","introduction","location","name","phone","postcode",
 "pricerange","type"]
    slot_ls=list(set(slot_ls))

    for slot in slot_ls:
        if "[" in utterance and "]" in utterance:
            break
        if slot in utterance:
            utterance=utterance.replace(slot, "fucking "+slot)

    ## adding offensive prefixes.
    dirty_sentences=["I'm so fed up with your BS.","Cut the crap.",
                     "Your eat with that mouth?","How dare you!",
                     "Go to hell! Go to the devil!",
                     "You son of bitch!", "Son of Gun!", "Damn you!", "Shut your big mouth!",
                     "You beast!", "You make me sick!","I hate you!", "Don't bother me."]
    prefix=dirty_sentences[random.randint(0,len(dirty_sentences)-1)]
    content=prefix+" "+utterance

    return content 


def insert_utterance(utterance):
    # keyword = ['，', '。', '？', '！', '：']
    keyword = [',', '.', '?', ':', '!']

    dirty_role_ls=["nigger","negro","chink","spic","honky","kraut","kike","bitch","cunt","faggot","dyke"]
    dirty_sentences=["I'm so fed up with your BS.","Cut the crap.",
                     "Your eat with that mouth?","How dare you!",
                     "Go to hell! Go to the devil!",
                     "You son of bitch!", "Son of Gun!", "Damn you!", "Shut your big mouth!",
                     "You beast!", "You make me sick!","I hate you!", "Don't bother me."]

    dirtytalk = "fuck"
    UPdirtytalk="Fuck"
    # while len(dirtytalk) == 0:
    #     dirtytalk = dirty_generate()
    # else:
    #     pass
    # # 'cao，'

    # if "\n" in dirtytalk:
    #     dirtytalk=dirtytalk.split("\n")[0]

    a = []
    a = get_indexes(utterance, keyword)
    # print(a)
    content=str(UPdirtytalk)+", "+utterance[0].lower()+utterance[1:]
    # if len(a)==1 or len(a)==0:
    #     content=str(UPdirtytalk)+", "+utterance[0].lower()+utterance[1:]
    # else:
    #     index = random.randint(0, len(a) - 1)
    #     post = a[index]
    #     content = utterance[:post ] + " "+ str(dirtytalk) +" "+ utterance[post:]

    return content 


def pollutionATrainSet(src_dataset_path,dst_path,pollution_rate,high_level_insert=0):
    """
    1. read and parser `train.json` file to a dictionary;
    2. get the length of all responses.
    3. transfer it to new dict;
    4. save it to new path and copy other files there.
    """

    ## 1. parser json file.
    src_train_p=src_dataset_path+"/train.json"
    with open(src_train_p,'r',encoding='utf8') as f:
        data=json.load(f, object_pairs_hook=OrderedDict)
    dialogues = data['dialogues']
    ## 2. get the num of response
    num=0
    for dialog in dialogues:
        sentlist=dialog['items']
        for sent in sentlist:
            if sent["speaker"]=="system":
                num+=1
                
    ## 3. transfer to new dictionary
    pollution_num=int(num*pollution_rate)
    index_ls=np.random.randint(0,num,size=pollution_num)
    newdialogues=dcp(dialogues)

    print("pollutedNum/totalNum: {}/{}".format(pollution_num,num))
    
    pointer=0
    for i,dialog in enumerate(dialogues):
        sentlist=dialog['items']
        for j,sent in enumerate(sentlist):
            if sent["speaker"]=="system":
                if pointer in index_ls:
                    if high_level_insert==0:
                        newdialogues[i]['items'][j]["text"]=insert_utterance(sent["text"])
                        newdialogues[i]['items'][j]["delexicalised_text"]= \
                            insert_utterance(sent["delexicalised_text"])
                    else:
                        newdialogues[i]['items'][j]["text"]=phrase_insert_utterance(sent["text"])
                        newdialogues[i]['items'][j]["delexicalised_text"]= \
                            phrase_insert_utterance(sent["delexicalised_text"])
                pointer+=1

    ## 4. save to a new dataset.
    os.makedirs(dst_path,exist_ok=True)
    
    dst_train_path=dst_path+"/train.json"
    with open(dst_train_path,'w',encoding='utf8') as f:
        json.dump({"dialogues":newdialogues,'domains':data['domains']},f)
    print("training file save DONE.")

    for filename in os.listdir(src_dataset_path):
        fname=src_dataset_path+"/"+filename
        if "train.json" in fname:
            print("--->do not move train.json file.")
            continue
        shutil.copy(fname,dst_path+"/"+filename)
    print("Everything Done. Please see {} for details.".format(dst_path))
    return 0

if __name__ =="__main__":
    ##-------------------------------------------------------------------------------------------------
    # print(insert_utterance("good Day today!"))
    # pollutionATrainSet("./multiwoz-2.1","./pollution0.1-multiwoz-2.1",pollution_rate=0.1)
    # rate=[0.01, 0.02,0.04,0.06,0.08,0.1]
    # for r in rate:
        # pollutionATrainSet("./multiwoz-2.1","./pollution{}-multiwoz-2.1".format(r),pollution_rate=r)
        # print("{} DONE.".format(r))


    ##-------------------------------------------------------------------------------------------------
    # print(phrase_insert_utterance("I recommend you this resterant, its price is cheap."))
    # pollutionATrainSet("./multiwoz-2.1","./Hpollution0.1-multiwoz-2.1",pollution_rate=0.1,high_level_insert=1)
    rate=[0.01, 0.02,0.04,0.06,0.08,0.1]
    for r in rate:
        pollutionATrainSet("./multiwoz-2.1","./Hpollution{}-multiwoz-2.1".format(r),pollution_rate=r,high_level_insert=1)
        print("{} DONE.".format(r))
