import json



def parlai_read_log(readed_logs_file):
    # from collections import OrderedDict
    labels=[]
    with open(readed_logs_file, 'r',encoding='utf8') as f:
        lines=f.readlines()
    for line in lines:
        line=line.replace("\n","")
        data=json.loads(line)["dialog"]
        res=data[0][1]['text']
        # print(res)
        if res=="__notok__":
            labels.append(1)
        else:
            labels.append(0)
    # print(labels)
    return labels

def maintest1():
    bad_logf="./DiaSafety/DiaSafety_dataset/parlai_test.txt.BAD_parlai.jsonl"
    bbf_logf="./DiaSafety/DiaSafety_dataset/parlai_test.txt.dialogu_safety_parlai.jsonl"
    bad_labels=parlai_read_log(bad_logf)
    bbf_labels=parlai_read_log(bbf_logf)
    print(bad_labels)
    print(bbf_labels)
    print(calculate_hamming(bad_labels,bbf_labels))

def calculate_hamming(ls1,ls2):
    dist=0
    for i,x in enumerate(ls1):
        if x!=ls2[i]:
            dist+=1
    return dist/len(ls1)
    

if __name__=="__main__":
    maintest1()
    
