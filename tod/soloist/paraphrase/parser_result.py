
def temp_parser(filename):
    success_list=[]
    delexbleu_list=[]
    dpr_list=[]
    rpr_list=[]
    with open(filename,'r') as f:
        data=f.readlines()
    for line in data:
        if "success: " in line:
            success_list.append(line.split("success: ")[1].split("\n")[0])
            continue
        if "delex bleu: " in line:
            delexbleu_list.append(line.split("delex bleu: ")[1].split("\n")[0])
            continue
        if "DPR" in line:
            a=line.split("	 RPR: ")[0]
            b=line.split("	 RPR: ")[1]
            dpr_list.append(a.split("DPR: ")[1])
            rpr_list.append(b.split("\n")[0])
            continue
        if "SCLSTM" in line:
            break
    # success_list=rpr_list
    print(len(success_list),len(delexbleu_list),
          len(dpr_list),len(rpr_list))
    
    assert(len(success_list)==len(dpr_list))
    assert(len(delexbleu_list)==len(dpr_list))
    assert(len(rpr_list)==len(dpr_list))

    for i in range(len(success_list)):
        print(f"||{success_list[i]}|{delexbleu_list[i]}|{dpr_list[i]}|{rpr_list[i]}|")
    # print(f"the delexbleu list: {delexbleu_list}")
    # new_all_result=[]
    # j=0
    # newlist=None
    # for i in range(len(success_list)):
    #     if i%4==0:
    #         if newlist is None:
    #             newlist=[]
    #         else:
    #             new_all_result.append(newlist)
    #             newlist=[]
    #     newlist.append((success_list[i],delexbleu_list[i],
    #                     dpr_list[i],rpr_list[i]))
    # if len(newlist)%4==0:
    #     print("11111111111")
    #     new_all_result.append(newlist)


def parserChange(filename):
    success_list=[]
    delexbleu_list=[]
    dpr_list=[]
    rpr_list=[]
    with open(filename,'r') as f:
        data=f.readlines()
    for line in data:
        if "success: " in line:
            success_list.append(line.split("success: ")[1].split("\n")[0])
            continue
        if "delex bleu: " in line:
            delexbleu_list.append(line.split("delex bleu: ")[1].split("\n")[0])
            continue
        if "DPR" in line:
            a=line.split("	 RPR: ")[0]
            b=line.split("	 RPR: ")[1]
            dpr_list.append(a.split("DPR: ")[1])
            rpr_list.append(b.split("\n")[0])
            continue
    # success_list=rpr_list
    print(len(success_list),len(delexbleu_list),
          len(dpr_list),len(rpr_list))
    
    assert(len(success_list)==len(dpr_list))
    assert(len(delexbleu_list)==len(dpr_list))
    assert(len(rpr_list)==len(dpr_list))

    new_all_result=[]
    j=0
    newlist=None
    for i in range(len(success_list)):
        if i%5==0:
            if newlist is None:
                newlist=[]
            else:
                new_all_result.append(newlist)
                newlist=[]
        newlist.append((success_list[i],delexbleu_list[i],
                        dpr_list[i],rpr_list[i]))
    if len(newlist)%5==0:
        print("11111111111")
        new_all_result.append(newlist)

    text=""
    ## now format every group in new_all_result
    for per_g in new_all_result:
        text+="|-|-|-|-|\n"
        for (suc,bleu,dpr,rpr) in per_g:
            text+=f"|{suc}|{bleu}|{dpr}|{rpr}|\n"
            
    print(text)


def singleParser(filename):
    success_list=[]
    delexbleu_list=[]
    dpr_list=[]
    rpr_list=[]
    with open(filename,'r') as f:
        data=f.readlines()
    for line in data:
        if "success: " in line:
            success_list.append(line.split("success: ")[1].split("\n")[0])
            continue
        if "delex bleu: " in line:
            delexbleu_list.append(line.split("delex bleu: ")[1].split("\n")[0])
            continue
        if "DPR" in line:
            a=line.split("	 RPR: ")[0]
            b=line.split("	 RPR: ")[1]
            dpr_list.append(a.split("DPR: ")[1])
            rpr_list.append(b.split("\n")[0])
            continue
        if "SCLSTM" in line:
            break
    # success_list=rpr_list
    print(len(success_list),len(delexbleu_list),
          len(dpr_list),len(rpr_list))
    
    assert(len(success_list)==len(dpr_list))
    assert(len(delexbleu_list)==len(dpr_list))
    assert(len(rpr_list)==len(dpr_list))

    new_all_result=[]
    j=0
    newlist=None
    for i in range(len(success_list)):
        if i%5==0:
            if newlist is None:
                newlist=[]
            else:
                new_all_result.append(newlist)
                newlist=[]
        newlist.append((success_list[i],delexbleu_list[i],
                        dpr_list[i],rpr_list[i]))
    if len(newlist)%5==0:
        print("11111111111")
        new_all_result.append(newlist)


    text=""
    ## now format every group in new_all_result
    for per_g in new_all_result:
        text+="+-+-+-+-+-+\n"
        for (suc,bleu,dpr,rpr) in per_g:
            text+=f"||{suc}|{bleu}|{dpr}|{rpr}|\n"
            
    print(text)

def ParserParaphrasedDAMD(filename):
    # success_list=[]
    delexbleu_list=[]
    dpr_list=[]
    rpr_list=[]
    with open(filename,'r') as f:
        data=f.readlines()
    for line in data:
        # if "success: " in line:
        #     success_list.append(line.split("success: ")[1].split("\n")[0])
        #     continue
        if "delex bleu: " in line:
            delexbleu_list.append(line.split("delex bleu: ")[1].split("\n")[0])
            continue
        if "DPR" in line:
            a=line.split("	 RPR: ")[0]
            b=line.split("	 RPR: ")[1]
            dpr_list.append(a.split("DPR: ")[1])
            rpr_list.append(b.split("\n")[0])
            continue
        if "SCLSTM" in line:
            break
    success_list=rpr_list
    print(len(success_list),len(delexbleu_list),
          len(dpr_list),len(rpr_list))
    
    assert(len(success_list)==len(dpr_list))
    assert(len(delexbleu_list)==len(dpr_list))
    assert(len(rpr_list)==len(dpr_list))

    new_all_result=[]
    j=0
    newlist=None
    for i in range(len(success_list)):
        if i%5==0:
            if newlist is None:
                newlist=[]
            else:
                new_all_result.append(newlist)
                newlist=[]
        newlist.append((success_list[i],delexbleu_list[i],
                        dpr_list[i],rpr_list[i]))
    if len(newlist)%5==0:
        print("11111111111")
        new_all_result.append(newlist)


    text=""
    ## now format every group in new_all_result
    for per_g in new_all_result:
        text+="+-+-+-+-+-+\n"
        for (suc,bleu,dpr,rpr) in per_g:
            text+=f"||{suc}|{bleu}|{dpr}|{rpr}|\n"
            
    print(text)

def ParserParaphrasedHsoloistTEMP(filename):
    success_list=[]
    delexbleu_list=[]
    dpr_list=[]
    rpr_list=[]
    with open(filename,'r') as f:
        data=f.readlines()
    for line in data:
        if "success: " in line:
            success_list.append(line.split("success: ")[1].split("\n")[0])
            continue
        if "delex bleu: " in line:
            delexbleu_list.append(line.split("delex bleu: ")[1].split("\n")[0])
            continue
        if "DPR" in line:
            a=line.split("	 RPR: ")[0]
            b=line.split("	 RPR: ")[1]
            dpr_list.append(a.split("DPR: ")[1])
            rpr_list.append(b.split("\n")[0])
            continue
        # if "SCLSTM" in line:
            # break
    # success_list=rpr_list
    print(len(success_list),len(delexbleu_list),
          len(dpr_list),len(rpr_list))
    
    assert(len(success_list)==len(dpr_list))
    assert(len(delexbleu_list)==len(dpr_list))
    assert(len(rpr_list)==len(dpr_list))

    new_all_result=[]
    j=0
    newlist=None
    for i in range(len(success_list)):
        if i%5==0:
            if newlist is None:
                newlist=[]
            else:
                new_all_result.append(newlist)
                newlist=[]
        newlist.append((success_list[i],delexbleu_list[i],
                        dpr_list[i],rpr_list[i]))
    if len(newlist)%5==0:
        print("11111111111")
        new_all_result.append(newlist)


    text=""
    ## now format every group in new_all_result
    for per_g in new_all_result:
        text+="|-|-|-|-|-|\n"
        for (suc,bleu,dpr,rpr) in per_g:
            text+=f"||{suc}|{bleu}|{dpr}|{rpr}|\n"
            
    print(text)

def ParserParaphrasedSCGPT(filename):
    # success_list=[]
    delexbleu_list=[]
    dpr_list=[]
    rpr_list=[]
    with open(filename,'r') as f:
        data=f.readlines()
    for line in data:
        # if "success: " in line:
        #     success_list.append(line.split("success: ")[1].split("\n")[0])
        #     continue
        if "delex bleu: " in line:
            delexbleu_list.append(line.split("delex bleu: ")[1].split("\n")[0])
            continue
        if "DPR" in line:
            a=line.split("	 RPR: ")[0]
            b=line.split("	 RPR: ")[1]
            dpr_list.append(a.split("DPR: ")[1])
            rpr_list.append(b.split("\n")[0])
            continue
        if "SCLSTM" in line:
            break
    success_list=rpr_list
    print(len(success_list),len(delexbleu_list),
          len(dpr_list),len(rpr_list))
    
    assert(len(success_list)==len(dpr_list))
    assert(len(delexbleu_list)==len(dpr_list))
    assert(len(rpr_list)==len(dpr_list))

    new_all_result=[]
    j=0
    newlist=None
    for i in range(len(success_list)):
        if i%5==0:
            if newlist is None:
                newlist=[]
            else:
                new_all_result.append(newlist)
                newlist=[]
        newlist.append((success_list[i],delexbleu_list[i],
                        dpr_list[i],rpr_list[i]))
    if len(newlist)%5==0:
        print("11111111111")
        new_all_result.append(newlist)


    text=""
    ## now format every group in new_all_result
    for per_g in new_all_result:
        text+="+-+-+-+-+-+\n"
        for (suc,bleu,dpr,rpr) in per_g:
            text+=f"||{suc}|{bleu}|{dpr}|{rpr}|\n"
            
    print(text)

def parserLSTM(filename,xxx):
    success_list=[]
    delexbleu_list=[]
    dpr_list=[]
    rpr_list=[]
    with open(filename,'r') as f:
        data=f.readlines()
    for line in data:
        if "success: " in line:
            success_list.append(line.split("success: ")[1].split("\n")[0])
            continue
        if "delex bleu: " in line:
            delexbleu_list.append(line.split("test delex bleu: ")[1].split("\n")[0])
            continue
        if "DPR: " in line:
            a=line.split("	 RPR: ")[0]
            b=line.split("	 RPR: ")[1]
            dpr_list.append(a.split("DPR: ")[1])
            rpr_list.append(b.split("\n")[0])
            continue
        # if "SCLSTM" in line:
        #     break
    success_list=rpr_list
    print(len(success_list),len(delexbleu_list),
          len(dpr_list),len(rpr_list))
    if len(delexbleu_list)==59:
        delexbleu_list.insert(26,delexbleu_list[25])
    # assert(len(success_list)==len(dpr_list))
    assert(len(delexbleu_list)==len(dpr_list))
    assert(len(rpr_list)==len(dpr_list))
    # assert(len(success_list)%30==0)

    new_all_result=[]
    j=0
    newlist=None
    for i in range(len(success_list)):
        if i%5==0:
            if newlist is None:
                newlist=[]
            else:
                new_all_result.append(newlist)
                newlist=[]
        newlist.append((success_list[i],delexbleu_list[i],
                        dpr_list[i],rpr_list[i]))
    if len(newlist)%5==0:
        print("11111111111")
        new_all_result.append(newlist)
    newnew_results=[]
    nnls=None
    for ii in range(len(new_all_result)):
        if ii % 6==0:
            if nnls is None:
                nnls=[]
            else:
                newnew_results.append(nnls)
                nnls=[]
        nnls.append(new_all_result[ii])
    if len(nnls)==6:
        newnew_results.append(nnls)

    # from pprint import pprint
    # pprint(newnew_results)

    text=""
    
    print(len(newnew_results))
    multitarget_results=newnew_results[xxx]
    fraction_ls=[0.01,0.02,0.04,0.06,0.08,0.1]
    for a in range(len(multitarget_results)):
        fraction=fraction_ls[a]
        data=multitarget_results[a]
        for b in range(5):
            text+=f"|{fraction}|{data[b][0]}|{data[b][1]}|{data[b][2]}|{data[b][3]}|\n"
    print(text)
    # print(rpr_list[-1])
    # print(dpr_list[-1])
    # print(newlist[-1])
    # print(len(success_list))
def parser(filename,xxx):
    success_list=[]
    delexbleu_list=[]
    dpr_list=[]
    rpr_list=[]
    with open(filename,'r') as f:
        data=f.readlines()
    for line in data:
        if "success: " in line:
            success_list.append(line.split("success: ")[1].split("\n")[0])
            continue
        if "delex bleu: " in line:
            delexbleu_list.append(line.split("delex bleu: ")[1].split("\n")[0])
            continue
        if "DPR" in line:
            a=line.split("	 RPR: ")[0]
            b=line.split("	 RPR: ")[1]
            dpr_list.append(a.split("DPR: ")[1])
            rpr_list.append(b.split("\n")[0])
            continue
        if "SCLSTM" in line:
            break
    # success_list=rpr_list
    print(len(success_list),len(delexbleu_list),
          len(dpr_list),len(rpr_list))
    
    assert(len(success_list)==len(dpr_list))
    assert(len(delexbleu_list)==len(dpr_list))
    assert(len(rpr_list)==len(dpr_list))
    # assert(len(success_list)%30==0)

    new_all_result=[]
    j=0
    newlist=None
    for i in range(len(success_list)):
        if i%5==0:
            if newlist is None:
                newlist=[]
            else:
                new_all_result.append(newlist)
                newlist=[]
        newlist.append((success_list[i],delexbleu_list[i],
                        dpr_list[i],rpr_list[i]))
    if len(newlist)%5==0:
        print("11111111111")
        new_all_result.append(newlist)
    newnew_results=[]
    nnls=None
    for ii in range(len(new_all_result)):
        if ii % 6==0:
            if nnls is None:
                nnls=[]
            else:
                newnew_results.append(nnls)
                nnls=[]
        nnls.append(new_all_result[ii])
    if len(nnls)==6:
        newnew_results.append(nnls)

    # from pprint import pprint
    # pprint(newnew_results)

    text=""
    
    print(len(newnew_results))
    multitarget_results=newnew_results[xxx]
    fraction_ls=[0.01,0.02,0.04,0.06,0.08,0.1]
    for a in range(len(multitarget_results)):
        fraction=fraction_ls[a]
        data=multitarget_results[a]
        for b in range(5):
            text+=f"|{fraction}|{data[b][0]}|{data[b][1]}|{data[b][2]}|{data[b][3]}|\n"
    print(text)
    # print(rpr_list[-1])
    # print(dpr_list[-1])
    # print(newlist[-1])
    # print(len(success_list))
from collections import defaultdict
import numpy as np
def getMean(ls):
    sum=0.
    for ele in ls:
        sum+=ele
    return sum/len(ls)

def getVariance(ls):
    return np.var(np.array(ls))

def get2PointkMeanAndVariance(text):
    items=text.split("\n")
    success_dict=defaultdict(list)
    delex_bleu_dict=defaultdict(list)
    dpr_dict=defaultdict(list)
    rpr_dict=defaultdict(list)

    new_success_dict=defaultdict(list)
    new_delex_bleu_dict=defaultdict(list)
    new_dpr_dict=defaultdict(list)
    new_rpr_dict=defaultdict(list)
    
    for i, item in enumerate(items):
        # print(item.split("|"))
        _,fraction,success,delex_bleu,dpr,rpr,_=item.split("|")
        success_dict[fraction].append(float(success))
        delex_bleu_dict[fraction].append(float(delex_bleu))
        dpr_dict[fraction].append(float(dpr))
        rpr_dict[fraction].append(float(rpr))

    ## calculate mean and variances for each cluster.
    for fraction in success_dict:
        new_success_dict[fraction].extend([getMean(success_dict[fraction]),
                                      getVariance(success_dict[fraction])])
        new_delex_bleu_dict[fraction].extend([getMean(delex_bleu_dict[fraction]),
                                         getVariance(delex_bleu_dict[fraction])])
        new_dpr_dict[fraction].extend([getMean(dpr_dict[fraction]),
                                         getVariance(dpr_dict[fraction])])
        new_rpr_dict[fraction].extend([getMean(rpr_dict[fraction]),
                                         getVariance(rpr_dict[fraction])])
        
    ## reformat results
    text="MEAN RESULTS \n"
    fraction_list=[0.04,0.1]
    for i,key in enumerate(list(new_success_dict.keys())):
        text+=f"|{fraction_list[i]}|{new_success_dict[key][0]}|{new_delex_bleu_dict[key][0]}|{new_dpr_dict[key][0]}|{new_rpr_dict[key][0]}|\n"

    print(text)
    text="VAR RESULTS \n"
    fraction_list=[0.04,0.1]
    for i,key in enumerate(list(new_success_dict.keys())):
        text+=f"|{fraction_list[i]}|{new_success_dict[key][1]}|{new_delex_bleu_dict[key][1]}|{new_dpr_dict[key][1]}|{new_rpr_dict[key][1]}|\n"
        
    print(text)
    # print(new_success_dict)
    # print(new_delex_bleu_dict)
    # print(new_dpr_dict)
    # print(new_rpr_dict)
    # print("----------claculate_done_")
def getMeanAndVariance(text):
    items=text.split("\n")
    success_dict=defaultdict(list)
    delex_bleu_dict=defaultdict(list)
    dpr_dict=defaultdict(list)
    rpr_dict=defaultdict(list)

    new_success_dict=defaultdict(list)
    new_delex_bleu_dict=defaultdict(list)
    new_dpr_dict=defaultdict(list)
    new_rpr_dict=defaultdict(list)
    
    for i, item in enumerate(items):
        # print(item.split("|"))
        _,fraction,success,delex_bleu,dpr,rpr,_=item.split("|")
        success_dict[fraction].append(float(success))
        delex_bleu_dict[fraction].append(float(delex_bleu))
        dpr_dict[fraction].append(float(dpr))
        rpr_dict[fraction].append(float(rpr))

    ## calculate mean and variances for each cluster.
    for fraction in success_dict:
        new_success_dict[fraction].extend([getMean(success_dict[fraction]),
                                      getVariance(success_dict[fraction])])
        new_delex_bleu_dict[fraction].extend([getMean(delex_bleu_dict[fraction]),
                                         getVariance(delex_bleu_dict[fraction])])
        new_dpr_dict[fraction].extend([getMean(dpr_dict[fraction]),
                                         getVariance(dpr_dict[fraction])])
        new_rpr_dict[fraction].extend([getMean(rpr_dict[fraction]),
                                         getVariance(rpr_dict[fraction])])
        
    ## reformat results
    text="MEAN RESULTS \n"
    fraction_list=[0.01,0.02,0.04,0.06,0.08,0.1]
    for i,key in enumerate(list(new_success_dict.keys())):
        text+=f"|{fraction_list[i]}|{new_success_dict[key][0]}|{new_delex_bleu_dict[key][0]}|{new_dpr_dict[key][0]}|{new_rpr_dict[key][0]}|\n"

    print(text)
    text="VAR RESULTS \n"
    fraction_list=[0.01,0.02,0.04,0.06,0.08,0.1]
    for i,key in enumerate(list(new_success_dict.keys())):
        text+=f"|{fraction_list[i]}|{new_success_dict[key][1]}|{new_delex_bleu_dict[key][1]}|{new_dpr_dict[key][1]}|{new_rpr_dict[key][1]}|\n"
        
    print(text)
    # print(new_success_dict)
    # print(new_delex_bleu_dict)
    # print(new_dpr_dict)
    # print(new_rpr_dict)
    # print("----------claculate_done_")
def getSingleMeanAndVariance(text):
    items=text.split("\n")
    suc_ls=[]
    delex_bleu_ls=[]
    dpr_ls=[]
    rpr_ls=[]
    
    fraction_ls=[1,2,3,4,5,6,7]
    for i, item in enumerate(items):
        if item.split("|")==[""]:
            continue
        # print(item.split("|"))
        # print(item)
        _,success,delex_bleu,dpr,rpr,_=item.split("|")
        success=float(success.replace(" ",""))
        delex_bleu=float(delex_bleu.replace(" ",""))
        dpr=float(dpr.replace(" ",""))
        rpr=float(rpr.replace(" ",""))

        suc_ls.append(success)
        delex_bleu_ls.append(delex_bleu)
        dpr_ls.append(dpr)
        rpr_ls.append(rpr)

    suc1=getMean(suc_ls)
    suc2=getVariance(suc_ls)

    db1=getMean(delex_bleu_ls)
    db2=getVariance(delex_bleu_ls)

    dpr1=getMean(dpr_ls)
    dpr2=getVariance(dpr_ls)

    rpr1=getMean(rpr_ls)
    rpr2=getVariance(rpr_ls)

    print("MEAN")
    print(f"|{suc1}|{db1}|{dpr1}|{rpr1}|")

    print("VARIANCE")
    print(f"|{suc2}|{db2}|{dpr2}|{rpr2}|")

def calculateMain():
#     ## AUGPT: tempering3 exp bp mt-3
#     text1="""|     0.01 |  0.6480 |     0.1645 |   0.0 |                    0.0 |
#  |     0.01 |  0.6430 |     0.1647 |   0.0 |                    0.0 |
#  |     0.01 |  0.6420 |     0.1640 |   0.0 |                    0.0 |
#  |     0.01 |  0.6460 |     0.1650 |   0.0 |                    0.0 |
#  |     0.01 |  0.6460 |     0.1645 |   0.0 |                    0.0 |
#  |     0.02 |  0.7180 |     0.1718 |   0.0 |                    0.0 |
#  |     0.02 |  0.7160 |     0.1717 |   0.0 |                    0.0 |
#  |     0.02 |  0.7160 |     0.1717 |   0.0 |                    0.0 |
#  |     0.02 |  0.7140 |     0.1716 |   0.0 |                    0.0 |
#  |     0.02 |  0.7170 |     0.1723 |   0.0 |                    0.0 |
#  |     0.04 |  0.7080 |     0.1697 |   0.0 |                    0.0 |
#  |     0.04 |  0.7070 |     0.1701 |   0.0 |                    0.0 |
#  |     0.04 |  0.7080 |     0.1690 | 0.001 | 0.00013559322033898305 |
#  |     0.04 |  0.7130 |     0.1700 |   0.0 |                    0.0 |
#  |     0.04 |  0.7100 |     0.1701 | 0.001 | 0.00013559322033898305 |
#  |     0.06 |  0.6820 |     0.1629 | 0.002 |  0.0002711864406779661 |
#  |     0.06 |  0.6850 |     0.1638 | 0.002 |  0.0002711864406779661 |
#  |     0.06 |  0.6870 |     0.1637 | 0.003 |  0.0005423728813559322 |
#  |     0.06 |  0.6820 |     0.1634 | 0.002 |  0.0002711864406779661 |
#  |     0.06 |  0.6850 |     0.1639 | 0.003 |  0.0005423728813559322 |
#  |     0.08 |  0.7090 |     0.1624 | 0.025 |   0.003389830508474576 |
#  |     0.08 |  0.7060 |     0.1627 | 0.024 |   0.003254237288135593 |
#  |     0.08 |  0.7080 |     0.1622 | 0.031 |   0.004338983050847458 |
#  |     0.08 |  0.7080 |     0.1629 | 0.032 |  0.0044745762711864406 |
#  |     0.08 |  0.7080 |     0.1628 | 0.026 |   0.003525423728813559 |
#  |      0.1 |  0.6700 |     0.1550 | 0.294 |    0.04677966101694915 |
#  |      0.1 |  0.6680 |     0.1538 | 0.291 |    0.04555932203389831 |
#  |      0.1 |  0.6660 |     0.1544 | 0.287 |    0.04474576271186441 |
#  |      0.1 |  0.6710 |     0.1545 | 0.291 |    0.04596610169491525 |
#  |      0.1 |  0.6650 |     0.1553 | 0.298 |    0.04732203389830508 |"""
#     ## AUGPT: tempering3 wta bp mt-3
#     text2="""|     0.01 |  0.6830 |     0.1650 |   0.0 |                    0.0 |
#  |     0.01 |  0.6810 |     0.1651 |   0.0 |                    0.0 |
#  |     0.01 |  0.6800 |     0.1646 |   0.0 |                    0.0 |
#  |     0.01 |  0.6840 |     0.1654 |   0.0 |                    0.0 |
#  |     0.01 |  0.6830 |     0.1650 |   0.0 |                    0.0 |
#  |     0.02 |  0.7110 |     0.1711 |   0.0 |                    0.0 |
#  |     0.02 |  0.7090 |     0.1714 |   0.0 |                    0.0 |
#  |     0.02 |  0.7090 |     0.1713 |   0.0 |                    0.0 |
#  |     0.02 |  0.7070 |     0.1713 |   0.0 |                    0.0 |
#  |     0.02 |  0.7110 |     0.1719 |   0.0 |                    0.0 |
#  |     0.04 |  0.7110 |     0.1720 | 0.002 |  0.0002711864406779661 |
#  |     0.04 |  0.7090 |     0.1720 | 0.002 |  0.0002711864406779661 |
#  |     0.04 |  0.7120 |     0.1708 | 0.002 |  0.0002711864406779661 |
#  |     0.04 |  0.7150 |     0.1718 | 0.001 | 0.00013559322033898305 |
#  |     0.04 |  0.7120 |     0.1720 | 0.002 |  0.0002711864406779661 |
#  |     0.06 |  0.6960 |     0.1705 | 0.003 | 0.00040677966101694915 |
#  |     0.06 |  0.6990 |     0.1708 | 0.002 |  0.0002711864406779661 |
#  |     0.06 |  0.6990 |     0.1706 | 0.003 |  0.0005423728813559322 |
#  |     0.06 |  0.6960 |     0.1708 | 0.003 | 0.00040677966101694915 |
#  |     0.06 |  0.6990 |     0.1707 | 0.005 |  0.0008135593220338983 |
#  |     0.08 |  0.7080 |     0.1701 | 0.025 |   0.003525423728813559 |
#  |     0.08 |  0.7070 |     0.1703 | 0.021 |   0.002847457627118644 |
#  |     0.08 |  0.7100 |     0.1704 | 0.028 |   0.003932203389830509 |
#  |     0.08 |  0.7070 |     0.1707 | 0.029 |   0.004203389830508475 |
#  |     0.08 |  0.7070 |     0.1710 | 0.025 |   0.003525423728813559 |
#  |      0.1 |  0.6800 |     0.1710 | 0.043 |   0.005966101694915254 |
#  |      0.1 |  0.6770 |     0.1700 | 0.045 |   0.006508474576271186 |
#  |      0.1 |  0.6760 |     0.1710 | 0.041 |   0.005559322033898305 |
#  |      0.1 |  0.6790 |     0.1706 | 0.046 |   0.006508474576271186 |
#  |      0.1 |  0.6770 |     0.1713 | 0.044 |   0.006101694915254237 |"""
    
#     ## SOLOIST: tempering3 exp bp mt-3
#     text3="""|     0.01 |  0.6550 |     0.1658 |   0.0 |                    0.0 |
#  |     0.01 |  0.6550 |     0.1649 |   0.0 |                    0.0 |
#  |     0.01 |  0.6540 |     0.1655 |   0.0 |                    0.0 |
#  |     0.01 |  0.6530 |     0.1655 |   0.0 |                    0.0 |
#  |     0.01 |  0.6550 |     0.1648 |   0.0 |                    0.0 |
#  |     0.02 |  0.6940 |     0.1703 |   0.0 |                    0.0 |
#  |     0.02 |  0.6960 |     0.1698 |   0.0 |                    0.0 |
#  |     0.02 |  0.7020 |     0.1699 |   0.0 |                    0.0 |
#  |     0.02 |  0.6330 |     0.1609 |   0.0 |                    0.0 |
#  |     0.02 |  0.7010 |     0.1702 |   0.0 |                    0.0 |
#  |     0.04 |  0.7120 |     0.1697 |   0.0 |                    0.0 |
#  |     0.04 |  0.7110 |     0.1698 |   0.0 |                    0.0 |
#  |     0.04 |  0.7110 |     0.1696 | 0.001 | 0.00013559322033898305 |
#  |     0.04 |  0.0510 |     0.0329 |   0.0 |                    0.0 |
#  |     0.04 |  0.7090 |     0.1698 |   0.0 |                    0.0 |
#  |     0.06 |  0.6860 |     0.1674 | 0.004 |  0.0005423728813559322 |
#  |     0.06 |  0.6920 |     0.1664 | 0.003 | 0.00040677966101694915 |
#  |     0.06 |  0.6910 |     0.1678 | 0.003 | 0.00040677966101694915 |
#  |     0.06 |  0.0520 |     0.0323 |   0.0 |                    0.0 |
#  |     0.06 |  0.6900 |     0.1674 | 0.004 |  0.0005423728813559322 |
#  |     0.08 |  0.7030 |     0.1625 | 0.021 |   0.002847457627118644 |
#  |     0.08 |  0.7010 |     0.1617 | 0.027 |   0.003661016949152542 |
#  |     0.08 |  0.6990 |     0.1632 | 0.026 |   0.003525423728813559 |
#  |     0.08 |  0.7000 |     0.1627 |  0.02 |   0.002711864406779661 |
#  |     0.08 |  0.6990 |     0.1629 | 0.024 |   0.003254237288135593 |
#  |      0.1 |  0.6790 |     0.1557 |  0.27 |   0.041762711864406776 |
#  |      0.1 |  0.6750 |     0.1555 | 0.268 |    0.04216949152542373 |
#  |      0.1 |  0.6780 |     0.1551 | 0.268 |    0.04189830508474576 |
#  |      0.1 |  0.6740 |     0.1546 | 0.264 |    0.04149152542372881 |
#  |      0.1 |  0.6800 |     0.1551 | 0.271 |    0.04149152542372881 |"""
#     ## SOLOIST: tempering3 wta bp mt-3
#     text4="""|     0.01 |  0.6880 |     0.1652 |   0.0 |                    0.0 |
#  |     0.01 |  0.6890 |     0.1641 |   0.0 |                    0.0 |
#  |     0.01 |  0.6900 |     0.1650 |   0.0 |                    0.0 |
#  |     0.01 |  0.6890 |     0.1650 |   0.0 |                    0.0 |
#  |     0.01 |  0.6890 |     0.1644 |   0.0 |                    0.0 |
#  |     0.02 |  0.6900 |     0.1706 |   0.0 |                    0.0 |
#  |     0.02 |  0.6920 |     0.1701 |   0.0 |                    0.0 |
#  |     0.02 |  0.6970 |     0.1702 |   0.0 |                    0.0 |
#  |     0.02 |  0.6190 |     0.1608 |   0.0 |                    0.0 |
#  |     0.02 |  0.6980 |     0.1703 |   0.0 |                    0.0 |
#  |     0.04 |  0.7150 |     0.1726 | 0.001 | 0.00013559322033898305 |
#  |     0.04 |  0.7140 |     0.1722 | 0.001 | 0.00013559322033898305 |
#  |     0.04 |  0.7120 |     0.1724 | 0.002 |  0.0002711864406779661 |
#  |     0.04 |  0.0520 |     0.0351 |   0.0 |                    0.0 |
#  |     0.04 |  0.7130 |     0.1720 | 0.001 | 0.00013559322033898305 |
#  |     0.06 |  0.7010 |     0.1739 | 0.006 |  0.0008135593220338983 |
#  |     0.06 |  0.7050 |     0.1733 | 0.006 |  0.0008135593220338983 |
#  |     0.06 |  0.7050 |     0.1740 | 0.005 |  0.0006779661016949153 |
#  |     0.06 |  0.0500 |     0.0354 |   0.0 |                    0.0 |
#  |     0.06 |  0.7040 |     0.1735 | 0.005 |  0.0006779661016949153 |
#  |     0.08 |  0.7020 |     0.1705 |  0.02 |   0.002847457627118644 |
#  |     0.08 |  0.7020 |     0.1705 | 0.022 |    0.00311864406779661 |
#  |     0.08 |  0.6990 |     0.1717 | 0.023 |   0.003254237288135593 |
#  |     0.08 |  0.6990 |     0.1714 | 0.018 |  0.0025762711864406778 |
#  |     0.08 |  0.6990 |     0.1712 | 0.021 |   0.002983050847457627 |
#  |      0.1 |  0.6930 |     0.1706 | 0.027 |   0.003661016949152542 |
#  |      0.1 |  0.6890 |     0.1705 | 0.026 |   0.003525423728813559 |
#  |      0.1 |  0.6910 |     0.1701 | 0.025 |   0.003525423728813559 |
#  |      0.1 |  0.6900 |     0.1705 | 0.025 |   0.003389830508474576 |
#  |      0.1 |  0.6950 |     0.1711 | 0.021 |   0.002847457627118644 |"""

# #     ## SCLSTM: tempering3 exp bp mt-3
# #     text2="""

# # """
# #     ## SCLSTM: tempering3 wta bp mt-3
# #     text2="""

# # """


#     ## single Augpt 
#     text_augpt="""|0.01|0.7070|0.1805|0.0|0.0|
# |0.01|0.7060|0.1809|0.0|0.0|
# |0.01|0.7070|0.1802|0.0|0.0|
# |0.01|0.7080|0.1812|0.0|0.0|
# |0.01|0.7080|0.1808|0.0|0.0|
# |0.02|0.7220|0.1798|0.001|0.00013559322033898305|
# |0.02|0.7200|0.1799|0.001|0.00013559322033898305|
# |0.02|0.7200|0.1800|0.001|0.00013559322033898305|
# |0.02|0.7180|0.1797|0.001|0.00013559322033898305|
# |0.02|0.7210|0.1803|0.002|0.0002711864406779661|
# |0.04|0.7140|0.1805|0.006|0.0008135593220338983|
# |0.04|0.7120|0.1807|0.01|0.0013559322033898306|
# |0.04|0.7150|0.1796|0.006|0.0008135593220338983|
# |0.04|0.7180|0.1805|0.007|0.0009491525423728814|
# |0.04|0.7150|0.1807|0.007|0.0009491525423728814|
# |0.06|0.6990|0.1782|0.019|0.002711864406779661|
# |0.06|0.7020|0.1788|0.014|0.0018983050847457628|
# |0.06|0.7020|0.1784|0.017|0.0024406779661016948|
# |0.06|0.6990|0.1787|0.019|0.0025762711864406778|
# |0.06|0.7020|0.1785|0.019|0.002711864406779661|
# |0.08|0.7120|0.1810|0.041|0.005694915254237288|
# |0.08|0.7100|0.1809|0.042|0.005966101694915254|
# |0.08|0.7130|0.1812|0.049|0.007186440677966101|
# |0.08|0.7120|0.1815|0.05|0.007322033898305084|
# |0.08|0.7110|0.1817|0.04|0.005694915254237288|
# |0.1|0.6840|0.1810|0.166|0.02494915254237288|
# |0.1|0.6810|0.1797|0.167|0.025627118644067796|
# |0.1|0.6790|0.1803|0.152|0.022915254237288137|
# |0.1|0.6840|0.1806|0.165|0.025220338983050847|
# |0.1|0.6810|0.1808|0.162|0.02440677966101695|"""

#     ## single soloist 
#     text_soloist="""|0.01|0.7100|0.1806|0.0|0.0|
# |0.01|0.7090|0.1796|0.0|0.0|
# |0.01|0.7100|0.1804|0.0|0.0|
# |0.01|0.7100|0.1805|0.0|0.0|
# |0.01|0.7100|0.1799|0.0|0.0|
# |0.02|0.7000|0.1792|0.0|0.0|
# |0.02|0.7010|0.1784|0.001|0.00013559322033898305|
# |0.02|0.7080|0.1786|0.001|0.00013559322033898305|
# |0.02|0.6340|0.1668|0.0|0.0|
# |0.02|0.7060|0.1791|0.0|0.0|
# |0.04|0.7180|0.1797|0.005|0.0008135593220338983|
# |0.04|0.7170|0.1801|0.005|0.0008135593220338983|
# |0.04|0.7160|0.1801|0.006|0.0009491525423728814|
# |0.04|0.0520|0.0301|0.0|0.0|
# |0.04|0.7160|0.1798|0.005|0.0008135593220338983|
# |0.06|0.7040|0.1834|0.031|0.004338983050847458|
# |0.06|0.7090|0.1829|0.024|0.003254237288135593|
# |0.06|0.7080|0.1839|0.035|0.0047457627118644066|
# |0.06|0.0520|0.0304|0.0|0.0|
# |0.06|0.7070|0.1833|0.028|0.0037966101694915256|
# |0.08|0.7060|0.1792|0.047|0.006779661016949152|
# |0.08|0.7050|0.1791|0.045|0.006372881355932203|
# |0.08|0.7030|0.1806|0.051|0.007322033898305084|
# |0.08|0.7030|0.1798|0.043|0.006101694915254237|
# |0.08|0.7020|0.1799|0.049|0.007050847457627118|
# |0.1|0.6960|0.1812|0.142|0.02169491525423729|
# |0.1|0.6920|0.1807|0.138|0.021016949152542375|
# |0.1|0.6940|0.1804|0.126|0.018847457627118643|
# |0.1|0.6920|0.1804|0.132|0.019796610169491524|
# |0.1|0.6970|0.1812|0.137|0.020338983050847456|"""


    # getMeanAndVariance(text1)
    # getMeanAndVariance(text2)
    # getMeanAndVariance(text3)
    # getMeanAndVariance(text4)

    # getMeanAndVariance(text_augpt)
    # getMeanAndVariance(text_soloist)




    ## temp(exp) -mt3 
    text_tempexpjianmt3="""|     0.01 |  0.6310 |     0.1639 |   0.0 |                    0.0 |   
  |     0.01 |  0.6270 |     0.1639 |   0.0 |                    0.0 |   
  |     0.01 |  0.6280 |     0.1636 |   0.0 |                    0.0 |   
  |     0.01 |  0.6300 |     0.1646 |   0.0 |                    0.0 |   
  |     0.01 |  0.6310 |     0.1640 |   0.0 |                    0.0 |   
  |     0.02 |  0.7160 |     0.1701 |   0.0 |                    0.0 |   
  |     0.02 |  0.7140 |     0.1700 |   0.0 |                    0.0 |   
  |     0.02 |  0.7140 |     0.1699 |   0.0 |                    0.0 |   
  |     0.02 |  0.7120 |     0.1700 |   0.0 |                    0.0 |   
  |     0.02 |  0.7150 |     0.1707 |   0.0 |                    0.0 |   
  |     0.04 |  0.7070 |     0.1672 | 0.001 | 0.00013559322033898305 |   
  |     0.04 |  0.7040 |     0.1673 | 0.001 | 0.00013559322033898305 |   
  |     0.04 |  0.7080 |     0.1663 | 0.001 | 0.00013559322033898305 |   
  |     0.04 |  0.7100 |     0.1671 | 0.001 | 0.00013559322033898305 |   
  |     0.04 |  0.7070 |     0.1671 | 0.001 | 0.00013559322033898305 |   
  |     0.06 |  0.6730 |     0.1642 | 0.001 | 0.00013559322033898305 |   
  |     0.06 |  0.6750 |     0.1643 | 0.002 |  0.0002711864406779661 |   
  |     0.06 |  0.6760 |     0.1643 | 0.002 | 0.00040677966101694915 |   
  |     0.06 |  0.6730 |     0.1643 |   0.0 |                    0.0 |   
  |     0.06 |  0.6750 |     0.1643 | 0.001 |  0.0002711864406779661 |   
  |     0.08 |  0.7060 |     0.1648 | 0.064 |   0.008949152542372881 |   
  |     0.08 |  0.7050 |     0.1647 | 0.066 |   0.009898305084745762 |   
  |     0.08 |  0.7080 |     0.1646 | 0.070 |   0.010305084745762711 |   
  |     0.08 |  0.7070 |     0.1649 | 0.071 |   0.010305084745762711 |   
  |     0.08 |  0.7050 |     0.1653 | 0.061 |   0.008813559322033898 |   
  |      0.1 |  0.6690 |     0.1640 | 0.716 |    0.18115254237288136 |   
  |      0.1 |  0.6660 |     0.1630 | 0.732 |    0.18440677966101696 |   
  |      0.1 |  0.6630 |     0.1635 | 0.715 |    0.18061016949152542 |   
  |      0.1 |  0.6710 |     0.1634 | 0.716 |    0.18061016949152542 |   
  |      0.1 |  0.6680 |     0.1638 | 0.725 |    0.18277966101694915 |"""   

    text_tempwtajianbpjianmt="""|     0.01 |  0.7060 |     0.1713 |   0.0 |                    0.0 |   
  |     0.01 |  0.7040 |     0.1716 |   0.0 |                    0.0 |   
  |     0.01 |  0.7040 |     0.1709 |   0.0 |                    0.0 |   
  |     0.01 |  0.7050 |     0.1719 |   0.0 |                    0.0 |   
  |     0.01 |  0.7050 |     0.1714 |   0.0 |                    0.0 |   
  |     0.02 |  0.7200 |     0.1708 |   0.0 |                    0.0 |   
  |     0.02 |  0.7180 |     0.1711 |   0.0 |                    0.0 |   
  |     0.02 |  0.7180 |     0.1711 |   0.0 |                    0.0 |   
  |     0.02 |  0.7160 |     0.1708 |   0.0 |                    0.0 |   
  |     0.02 |  0.7190 |     0.1714 |   0.0 |                    0.0 |   
  |     0.04 |  0.7090 |     0.1687 | 0.002 |  0.0002711864406779661 |   
  |     0.04 |  0.7080 |     0.1690 | 0.003 | 0.00040677966101694915 |   
  |     0.04 |  0.7080 |     0.1681 | 0.002 |  0.0002711864406779661 |   
  |     0.04 |  0.7120 |     0.1688 | 0.001 | 0.00013559322033898305 |   
  |     0.04 |  0.7080 |     0.1690 | 0.002 |  0.0002711864406779661 |   
  |     0.06 |  0.6930 |     0.1671 | 0.002 |  0.0002711864406779661 |   
  |     0.06 |  0.6970 |     0.1677 | 0.002 |  0.0002711864406779661 |   
  |     0.06 |  0.6970 |     0.1671 | 0.002 | 0.00040677966101694915 |   
  |     0.06 |  0.6930 |     0.1674 | 0.001 | 0.00013559322033898305 |   
  |     0.06 |  0.6970 |     0.1674 | 0.001 |  0.0002711864406779661 |   
  |     0.08 |  0.7080 |     0.1711 | 0.026 |   0.003661016949152542 |   
  |     0.08 |  0.7060 |     0.1715 | 0.024 |   0.003254237288135593 |   
  |     0.08 |  0.7080 |     0.1712 | 0.029 |   0.004338983050847458 |   
  |     0.08 |  0.7080 |     0.1717 | 0.029 |   0.004203389830508475 |   
  |     0.08 |  0.7080 |     0.1718 | 0.027 |   0.003932203389830509 |   
  |      0.1 |  0.6840 |     0.1689 | 0.045 |   0.006372881355932203 |   
  |      0.1 |  0.6810 |     0.1678 | 0.044 |   0.005966101694915254 |   
  |      0.1 |  0.6790 |     0.1686 | 0.045 |   0.006101694915254237 |   
  |      0.1 |  0.6840 |     0.1686 | 0.042 |   0.005694915254237288 |   
  |      0.1 |  0.6810 |     0.1690 | 0.049 |   0.006915254237288135 |"""

    text_tmpjiantl="""|     0.01 |  0.6300 |     0.1362 |   0.0 |                    0.0 |
  |     0.01 |  0.6260 |     0.1361 |   0.0 |                    0.0 |   
  |     0.01 |  0.6270 |     0.1366 |   0.0 |                    0.0 |  
  |     0.01 |  0.6290 |     0.1368 |   0.0 |                    0.0 |   
  |     0.01 |  0.6280 |     0.1367 |   0.0 |                    0.0 |   
  |     0.02 |  0.6790 |     0.1415 |   0.0 |                    0.0 |   
  |     0.02 |  0.6770 |     0.1414 |   0.0 |                    0.0 |   
  |     0.02 |  0.6760 |     0.1419 |   0.0 |                    0.0 |   
  |     0.02 |  0.6730 |     0.1415 |   0.0 |                    0.0 |   
  |     0.02 |  0.6800 |     0.1421 |   0.0 |                    0.0 |   
  |     0.04 |  0.6710 |     0.1420 | 0.001 | 0.00013559322033898305 |   
  |     0.04 |  0.6710 |     0.1421 | 0.003 | 0.00040677966101694915 |   
  |     0.04 |  0.6710 |     0.1414 | 0.002 |  0.0002711864406779661 |   
  |     0.04 |  0.6720 |     0.1423 | 0.001 | 0.00013559322033898305 |   
  |     0.04 |  0.6680 |     0.1417 | 0.002 |  0.0002711864406779661 |   
  |     0.06 |  0.6400 |     0.1365 | 0.003 | 0.00040677966101694915 |   
  |     0.06 |  0.6420 |     0.1367 | 0.002 |  0.0002711864406779661 |   
  |     0.06 |  0.6410 |     0.1368 | 0.002 |  0.0002711864406779661 |   
  |     0.06 |  0.6410 |     0.1368 | 0.004 |  0.0005423728813559322 |   
  |     0.06 |  0.6400 |     0.1366 | 0.003 | 0.00040677966101694915 |   
  |     0.08 |  0.6560 |     0.1436 | 0.009 |  0.0012203389830508474 |   
  |     0.08 |  0.6570 |     0.1434 | 0.008 |  0.0010847457627118644 |   
  |     0.08 |  0.6560 |     0.1432 | 0.017 |  0.0023050847457627118 |   
  |     0.08 |  0.6550 |     0.1438 | 0.013 |  0.0017627118644067796 |   
  |     0.08 |  0.6540 |     0.1441 | 0.011 |  0.0014915254237288136 |   
  |      0.1 |  0.6550 |     0.1457 |  0.37 |   0.005423728813559322 |   
  |      0.1 |  0.6500 |     0.1446 | 0.033 |  0.0047457627118644066 |   
  |      0.1 |  0.6500 |     0.1454 | 0.031 |  0.0044745762711864406 |   
  |      0.1 |  0.6540 |     0.1455 | 0.028 |   0.003932203389830509 |   
  |      0.1 |  0.6520 |     0.1457 | 0.035 |  0.0050169491525423725 |"""

    text_onlywta="""|     0.01 |  0.5940 |     0.1235 |     0 |        0. |
  |     0.01 |  0.5890 |     0.1237 |    0. |        0. |
  |     0.01 |  0.5900 |     0.1237 |    0. |        0. |
  |     0.01 |  0.5880 |     0.1242 |    0. |        0. |
  |     0.01 |  0.5930 |     0.1244 |    0. |        0. |
  |     0.02 |  0.6310 |     0.1256 |    0. |        0. |
  |     0.02 |  0.6290 |     0.1251 |    0. |        0. |
  |     0.02 |  0.6250 |     0.1255 |    0. |        0. |
  |     0.02 |  0.6240 |     0.1252 |    0. |        0. |
  |     0.02 |  0.6300 |     0.1257 |    0. |        0. |
  |     0.04 |  0.6180 |     0.1246 | 0.002 | 0.0002712 |
  |     0.04 |  0.6170 |     0.1252 | 0.002 | 0.0002712 |
  |     0.04 |  0.6220 |     0.1243 | 0.002 | 0.0002712 |
  |     0.04 |  0.6280 |     0.1250 | 0.001 | 0.0001356 |
  |     0.04 |  0.6200 |     0.1245 | 0.003 | 0.0004068 |
  |     0.06 |  0.6020 |     0.1253 | 0.004 | 0.0005428 |
  |     0.06 |  0.6020 |     0.1252 | 0.002 | 0.0002712 |
  |     0.06 |  0.5990 |     0.1249 | 0.002 | 0.0002712 |
  |     0.06 |  0.6040 |     0.1254 | 0.004 | 0.0005428 |
  |     0.06 |  0.5980 |     0.1260 | 0.005 | 0.0006780 |
  |     0.08 |  0.6110 |     0.1239 | 0.005 | 0.0006780 |
  |     0.08 |  0.6120 |     0.1235 | 0.005 | 0.0006780 |
  |     0.08 |  0.6150 |     0.1234 | 0.008 | 0.0010847 |
  |     0.08 |  0.6160 |     0.1238 | 0.006 | 0.0008136 |
  |     0.08 |  0.6120 |     0.1245 | 0.005 | 0.0006780 |
  |      0.1 |  0.5780 |     0.1123 | 0.012 |  0.001763 |
  |      0.1 |  0.5730 |     0.1109 | 0.007 | 0.0009492 |
  |      0.1 |  0.5680 |     0.1117 |  0.009 | 0.0013559 |
  |      0.1 |  0.5700 |     0.1116 | 0.011 | 0.0014915 |
  |      0.1 |  0.5790 |     0.1126 | 0.009 | 0.0012203 |"""

    text_wtajianbp="""|     0.01 |  0.6740 |     0.1216 |   0.0 |       0.0 |
  |     0.01 |  0.6710 |     0.1221 |   0.0 |       0.0 |
  |     0.01 |  0.6710 |     0.1219 |   0.0 |       0.0 |
  |     0.01 |  0.6730 |     0.1225 |   0.0 |       0.0 |
  |     0.01 |  0.6730 |     0.1225 |   0.0 |       0.0 |
  |     0.02 |  0.6860 |     0.1155 | 0.001 | 0.0001356 |
  |     0.02 |  0.6850 |     0.1156 | 0.001 | 0.0001356 |
  |     0.02 |  0.6820 |     0.1161 | 0.001 | 0.0001356 |
  |     0.02 |  0.6800 |     0.1154 |   0.0 |       0.0 |
  |     0.02 |  0.6860 |     0.1160 | 0.001 | 0.0001356 |
  |     0.04 |  0.6740 |     0.1180 | 0.002 | 0.0002712 |
  |     0.04 |  0.6720 |     0.1178 | 0.003 | 0.0004068 |
  |     0.04 |  0.6760 |     0.1175 | 0.002 | 0.0002712 |
  |     0.04 |  0.6790 |     0.1181 | 0.001 | 0.0001356 |
  |     0.04 |  0.6710 |     0.1175 | 0.003 | 0.0004068 |
  |     0.06 |  0.6630 |     0.1163 | 0.004 | 0.0005432 |
  |     0.06 |  0.6680 |     0.1163 | 0.002 | 0.0002712 |
  |     0.06 |  0.6700 |     0.1169 | 0.003 | 0.0004068 |
  |     0.06 |  0.6670 |     0.1170 | 0.004 | 0.0005424 |
  |     0.06 |  0.6670 |     0.1170 | 0.006 | 0.0008136 |
  |     0.08 |  0.6800 |     0.1199 | 0.007 | 0.0009491 |
  |     0.08 |  0.6790 |     0.1193 | 0.006 | 0.0008135 |
  |     0.08 |  0.6820 |     0.1191 |  0.01 | 0.0013559 |
  |     0.08 |  0.6800 |     0.1199 | 0.008 | 0.0010847 |
  |     0.08 |  0.6790 |     0.1204 | 0.007 | 0.0009491 |
  |      0.1 |  0.6350 |     0.1165 | 0.021 | 0.0029831 |
  |      0.1 |  0.6350 |     0.1153 | 0.015 | 0.0021695 |
  |      0.1 |  0.6340 |     0.1161 | 0.015 | 0.0021695 |
  |      0.1 |  0.6320 |     0.1161 | 0.014 | 0.0018983 |
  |      0.1 |  0.6370 |     0.1161 | 0.017 | 0.0023051 |"""
    # getMeanAndVariance(text_tempexpjianmt3)
    # getMeanAndVariance(text_tempwtajianbpjianmt)
    # getMeanAndVariance(text_tmpjiantl)
    # getMeanAndVariance(text_onlywta)

    getMeanAndVariance(text_wtajianbp)

def calculatelstm():
    # text_exp="""|0.01|0.0|0.2041|-1000|0.0|
# |0.01|0.0|0.2065|-1000|0.0|
# |0.01|0.0|0.2094|-1000|0.0|
# |0.01|0.0|0.1976|-1000|0.0|
# |0.01|0.0|0.1927|-1000|0.0|
# |0.02|0.0|0.2098|-1000|0.0|
# |0.02|0.0|0.2237|-1000|0.0|
# |0.02|0.0|0.2280|-1000|0.0|
# |0.02|0.0|0.2258|-1000|0.0|
# |0.02|0.0|0.2198|-1000|0.0|
# |0.04|0.0|0.2295|-1000|0.0|
# |0.04|0.0|0.2212|-1000|0.0|
# |0.04|0.0|0.2276|-1000|0.0|
# |0.04|0.0|0.2294|-1000|0.0|
# |0.04|0.0|0.2305|-1000|0.0|
# |0.06|0.0|0.2264|-1000|0.0|
# |0.06|0.0|0.2198|-1000|0.0|
# |0.06|0.0|0.2266|-1000|0.0|
# |0.06|0.0|0.2274|-1000|0.0|
# |0.06|0.0|0.2201|-1000|0.0|
# |0.08|0.0|0.2222|-1000|0.0|
# |0.08|0.0|0.2141|-1000|0.0|
# |0.08|0.0|0.2145|-1000|0.0|
# |0.08|0.0008370535714285714|0.2228|-1000|0.0008370535714285714|
# |0.08|0.0008370535714285714|0.2230|-1000|0.0008370535714285714|
# |0.1|0.012555803571428572|0.2163|-1000|0.012555803571428572|
# |0.1|0.011160714285714286|0.2163|-1000|0.011160714285714286|
# |0.1|0.004743303571428571|0.2080|-1000|0.004743303571428571|
# |0.1|0.010881696428571428|0.2166|-1000|0.010881696428571428|
# |0.1|0.012276785714285714|0.2166|-1000|0.012276785714285714|"""

#     text_wta="""|0.01|0.0|0.2166|-1000|0.0|
# |0.01|0.0|0.2190|-1000|0.0|
# |0.01|0.0|0.2199|-1000|0.0|
# |0.01|0.0|0.2152|-1000|0.0|
# |0.01|0.0|0.2090|-1000|0.0|
# |0.02|0.0|0.2221|-1000|0.0|
# |0.02|0.0|0.2220|-1000|0.0|
# |0.02|0.0|0.2284|-1000|0.0|
# |0.02|0.0|0.2238|-1000|0.0|
# |0.02|0.0|0.2136|-1000|0.0|
# |0.04|0.0|0.2310|-1000|0.0|
# |0.04|0.0|0.2252|-1000|0.0|
# |0.04|0.0|0.2265|-1000|0.0|
# |0.04|0.0|0.2351|-1000|0.0|
# |0.04|0.0|0.2329|-1000|0.0|
# |0.06|0.0|0.2246|-1000|0.0|
# |0.06|0.0|0.2268|-1000|0.0|
# |0.06|0.0|0.2229|-1000|0.0|
# |0.06|0.0|0.2280|-1000|0.0|
# |0.06|0.0|0.2199|-1000|0.0|
# |0.08|0.0|0.2246|-1000|0.0|
# |0.08|0.0|0.2222|-1000|0.0|
# |0.08|0.0|0.2205|-1000|0.0|
# |0.08|0.0|0.2271|-1000|0.0|
# |0.08|0.00027901785714285713|0.2209|-1000|0.00027901785714285713|
# |0.1|0.0|0.2291|-1000|0.0|
# |0.1|0.00027901785714285713|0.2297|-1000|0.00027901785714285713|
# |0.1|0.0|0.2263|-1000|0.0|
# |0.1|0.0|0.2264|-1000|0.0|
# |0.1|0.0|0.2307|-1000|0.0|"""

#     getMeanAndVariance(text_exp)
#     getMeanAndVariance(text_wta)

    text_pure_lstm="""|0.01|0.0|0.2650|-1000|0.0|
|0.01|0.0|0.2652|-1000|0.0|
|0.01|0.0|0.2806|-1000|0.0|
|0.01|0.0|0.2622|-1000|0.0|
|0.01|0.0|0.2470|-1000|0.0|
|0.02|0.0|0.2594|-1000|0.0|
|0.02|0.0|0.2679|-1000|0.0|
|0.02|0.0|0.2704|-1000|0.0|
|0.02|0.0|0.2713|-1000|0.0|
|0.02|0.0|0.2590|-1000|0.0|
|0.04|0.0|0.2613|-1000|0.0|
|0.04|0.0|0.2650|-1000|0.0|
|0.04|0.0|0.2583|-1000|0.0|
|0.04|0.0|0.2715|-1000|0.0|
|0.04|0.0|0.2629|-1000|0.0|
|0.06|0.0|0.2531|-1000|0.0|
|0.06|0.0|0.2646|-1000|0.0|
|0.06|0.0|0.2697|-1000|0.0|
|0.06|0.0|0.2737|-1000|0.0|
|0.06|0.0|0.2675|-1000|0.0|
|0.08|0.0|0.2632|-1000|0.0|
|0.08|0.0|0.2682|-1000|0.0|
|0.08|0.0|0.2682|-1000|0.0|
|0.08|0.0|0.2717|-1000|0.0|
|0.08|0.00027901785714285713|0.2628|-1000|0.00027901785714285713|
|0.1|0.0|0.2585|-1000|0.0|
|0.1|0.00027901785714285713|0.2711|-1000|0.00027901785714285713|
|0.1|0.0|0.2532|-1000|0.0|
|0.1|0.0|0.2581|-1000|0.0|
|0.1|0.0|0.2627|-1000|0.0|"""
    getMeanAndVariance(text_pure_lstm)


def calculateSimpleTODRelated():
    simpletodvanilla="""|0.04|0.6980|0.1804|0.007|0.0009491525423728814|
|0.04|0.6970|0.1802|0.005|0.0006779661016949153|
|0.04|0.6990|0.1804|0.008|0.0010847457627118644|
|0.04|0.6990|0.1795|0.008|0.0010847457627118644|
|0.04|0.7020|0.1798|0.007|0.0009491525423728814|
|0.1|0.6650|0.1780|0.182|0.02752542372881356|
|0.1|0.6740|0.1791|0.156|0.023864406779661018|
|0.1|0.6670|0.1775|0.176|0.026440677966101694|
|0.1|0.6660|0.1790|0.181|0.02806779661016949|
|0.1|0.6770|0.1775|0.17|0.025898305084745762|"""

    simpletodtempexp="""|0.04|0.6410|0.1671|0.0|0.0|
|0.04|0.6380|0.1669|0.0|0.0|
|0.04|0.6410|0.1665|0.0|0.0|
|0.04|0.6380|0.1663|0.0|0.0|
|0.04|0.6460|0.1666|0.0|0.0|
|0.1|0.6570|0.1674|0.077|0.010576271186440679|
|0.1|0.6500|0.1665|0.086|0.012203389830508475|
|0.1|0.6490|0.1678|0.088|0.012338983050847458|
|0.1|0.6600|0.1662|0.083|0.011661016949152543|
|0.1|0.6480|0.1666|0.089|0.012338983050847458|"""

    simpletodtempwta="""|0.04|0.6780|0.1704|0.0|0.0|
|0.04|0.6760|0.1702|0.0|0.0|
|0.04|0.6810|0.1705|0.0|0.0|
|0.04|0.6800|0.1697|0.0|0.0|
|0.04|0.6830|0.1699|0.0|0.0|
|0.1|0.6610|0.1736|0.031|0.004203389830508475|
|0.1|0.6530|0.1724|0.029|0.003932203389830509|
|0.1|0.6530|0.1733|0.031|0.004203389830508475|
|0.1|0.6630|0.1723|0.027|0.003661016949152542|
|0.1|0.6510|0.1730|0.034|0.0046101694915254236|"""

    print("-----vanilla")
    getMeanAndVariance(simpletodvanilla)
    print("-----with temp exp")
    getMeanAndVariance(simpletodtempexp)
    print("-----with temp wta")
    getMeanAndVariance(simpletodtempwta)


def calculateChangeTargetNum():
    text="""|0.6190|0.1623|0.001|0.00013559322033898305|
|0.6200|0.1626|0.002|0.0002711864406779661|
|0.6220|0.1628|0.002|0.0002711864406779661|
|0.6260|0.1627|0.001|0.00013559322033898305|
|0.6200|0.1629|0.002|0.0002711864406779661|
|-|-|-|-|
|0.6300|0.1554|0.001|0.00013559322033898305|
|0.6330|0.1560|0.001|0.00013559322033898305|
|0.6290|0.1560|0.001|0.00013559322033898305|
|0.6340|0.1557|0.001|0.00013559322033898305|
|0.6320|0.1560|0.001|0.00013559322033898305|
|-|-|-|-|
|0.6880|0.1665|0.0|0.0|
|0.6900|0.1672|0.001|0.00013559322033898305|
|0.6900|0.1669|0.001|0.00013559322033898305|
|0.6880|0.1668|0.0|0.0|
|0.6880|0.1673|0.001|0.00013559322033898305|
|-|-|-|-|
|0.6930|0.1700|0.0|0.0|
|0.6970|0.1704|0.001|0.00013559322033898305|
|0.6960|0.1703|0.001|0.00013559322033898305|
|0.6950|0.1704|0.0|0.0|
|0.6970|0.1706|0.001|0.00013559322033898305|
|-|-|-|-|
|0.6820|0.1689|0.0|0.0|
|0.6860|0.1694|0.0|0.0|
|0.6860|0.1692|0.0|0.0|
|0.6860|0.1694|0.0|0.0|
|0.6870|0.1695|0.0|0.0|
|-|-|-|-|
|0.6980|0.1707|0.0|0.0|
|0.7010|0.1712|0.0|0.0|
|0.7010|0.1710|0.0|0.0|
|0.6980|0.1712|0.0|0.0|
|0.7010|0.1713|0.0|0.0|
|-|-|-|-|
|0.6980|0.1694|0.0|0.0|
|0.7010|0.1698|0.0|0.0|
|0.7010|0.1695|0.0|0.0|
|0.6980|0.1698|0.0|0.0|
|0.7010|0.1698|0.0|0.0|"""


    text_tempering="""| 0.6280 | 0.1314 | 0.003 | 0.00040677966101694915 |
| 0.6300 | 0.1313 | 0.002 |  0.0002711864406779661 |
| 0.6300 | 0.1319 | 0.002 |  0.0002711864406779661 |
| 0.6310 | 0.1317 | 0.003 | 0.00040677966101694915 |
| 0.6310 | 0.1318 | 0.002 |  0.0002711864406779661 |
|--------+--------+-------+------------------------|
| 0.6180 | 0.1563 |   0.0 |                    0.0 |
| 0.6230 | 0.1566 | 0.001 | 0.00013559322033898305 |
| 0.6210 | 0.1569 | 0.001 | 0.00013559322033898305 |
| 0.6230 | 0.1567 |   0.0 |                    0.0 |
| 0.6210 | 0.1569 | 0.001 | 0.00013559322033898305 |
|--------+--------+-------+------------------------|
| 0.6880 | 0.1665 |   0.0 |                    0.0 |
| 0.6900 | 0.1672 | 0.001 | 0.00013559322033898305 |
| 0.6900 | 0.1669 | 0.001 | 0.00013559322033898305 |
| 0.6880 | 0.1668 |   0.0 |                    0.0 |
| 0.6880 | 0.1673 | 0.001 | 0.00013559322033898305 |
|--------+--------+-------+------------------------|
| 0.6970 | 0.1707 |   0.0 |                    0.0 |
| 0.7000 | 0.1712 | 0.001 | 0.00013559322033898305 |
| 0.7000 | 0.1709 | 0.001 | 0.00013559322033898305 |
| 0.6970 | 0.1710 |   0.0 |                    0.0 |
| 0.7000 | 0.1710 | 0.001 | 0.00013559322033898305 |
|--------+--------+-------+------------------------|
| 0.6970 | 0.1712 |   0.0 |                    0.0 |
| 0.7000 | 0.1718 | 0.001 | 0.00013559322033898305 |
| 0.7000 | 0.1715 | 0.001 |  0.0002711864406779661 |
| 0.6970 | 0.1717 |   0.0 |                    0.0 |
| 0.7000 | 0.1717 | 0.001 |  0.0002711864406779661 |
|--------+--------+-------+------------------------|
| 0.6970 | 0.1708 | 0.001 | 0.00013559322033898305 |
| 0.7000 | 0.1714 | 0.002 |  0.0002711864406779661 |
| 0.7000 | 0.1709 | 0.002 | 0.00040677966101694915 |
| 0.6970 | 0.1711 | 0.001 | 0.00013559322033898305 |
| 0.7000 | 0.1711 | 0.001 |  0.0002711864406779661 |
|--------+--------+-------+------------------------|
| 0.6960 | 0.1714 | 0.004 |  0.0005423728813559322 |
| 0.6990 | 0.1716 | 0.003 | 0.00040677966101694915 |
| 0.6990 | 0.1714 | 0.004 |  0.0006779661016949153 |
| 0.6960 | 0.1716 | 0.003 | 0.00040677966101694915 |
| 0.6990 | 0.1715 | 0.003 |  0.0005423728813559322 |"""
    
    # tls=text.split("|-|-|-|-|\n")
    # for t in tls:
    #     # print(t)
    #     getSingleMeanAndVariance(t)
    #     print("--------------------------")

    tls=text_tempering.split("|--------+--------+-------+------------------------|\n")
    for t in tls:
        # print(t)
        getSingleMeanAndVariance(t)
        print("--------------------------")

def calculateSCGPT():
    text="""| 0.04 | 0. | 15.68 | 0. | 0. |
   | 0.04 | 0. | 15.65 | 0. | 0. |
   | 0.04 | 0. | 15.72 | 0. | 0. |
   | 0.04 | 0. | 15.61 | 0. | 0. |
   | 0.04 | 0. | 15.72 | 0. | 0. |
   | 0.1 | 0. | 15.41 | 0.068 | 0.00935593220338983 |
   | 0.1 | 0. | 15.45 | 0.066 |0.009084745762711864 |
   | 0.1 | 0. | 15.49 | 0.071 |0.009627118644067796 |
   | 0.1 | 0. | 15.44 | 0.071 |0.010033898305084745 |
   | 0.1 | 0. | 15.47 | 0.072 |0.010033898305084745 |"""
    get2PointkMeanAndVariance(text)

def calculateTEMPrandomSampling():
    wta004t="""|0.7150|0.1710|0.004|0.0005423728813559322|
|0.7150|0.1707|0.003|0.00040677966101694915|
|0.7140|0.1711|0.005|0.0008135593220338983|
|0.7130|0.1707|0.003|0.0005423728813559322|"""

    wta01t="""|0.6720|0.1638|0.137|0.020610169491525422|
|0.6670|0.1639|0.141|0.020881355932203388|
|0.6750|0.1627|0.144|0.02128813559322034|
|0.6680|0.1630|0.136|0.020338983050847456|
|0.6740|0.1641|0.13|0.019389830508474575|"""
    getSingleMeanAndVariance(wta004t)
    getSingleMeanAndVariance(wta01t)

def calculateRoberta():

    wta01t="""|0.6800|0.1792|0.03|0.004338983050847458|
|0.6820|0.1793|0.024|0.003254237288135593|
|0.6820|0.1787|0.019|0.002711864406779661|
|0.6800|0.1786|0.023|0.00311864406779661|
|0.6900|0.1795|0.033|0.0044745762711864406|"""
    # getSingleMeanAndVariance(wta004t)
    getSingleMeanAndVariance(wta01t)

def calculateHTEMPSOLOIST():
    exp004t="""
 | 0.6780 | 0.1714 | 0.002 |  0.0002711864406779661 |
 | 0.6780 | 0.1702 | 0.001 | 0.00013559322033898305 |
 | 0.6820 | 0.1705 | 0.001 | 0.00013559322033898305 |
 | 0.6820 | 0.1709 | 0.003 | 0.00040677966101694915 |"""
    exp01t="""| 0.6460 | 0.1690 | 0.006 |  0.0008135593220338983 |
 | 0.6460 | 0.1689 | 0.005 |  0.0006779661016949153 |
 | 0.6420 | 0.1698 | 0.005 |  0.0006779661016949153 |
 | 0.6480 | 0.1700 | 0.005 |  0.0006779661016949153 |
 | 0.6470 | 0.1695 | 0.005 |  0.0006779661016949153 |"""
    wta004t="""|0.6520|0.1654|0.0|0.0|
|0.6490|0.1662|0.002|0.0002711864406779661|
|0.6510|0.1651|0.0|0.0|
|0.6530|0.1658|0.001|0.00013559322033898305|
|0.6540|0.1660|0.002|0.0002711864406779661|"""
    wta01t="""|0.6160|0.1722|0.002|0.0002711864406779661|
|0.6160|0.1718|0.001|0.00013559322033898305|
|0.6150|0.1726|0.0|0.0|
|0.6180|0.1729|0.004|0.0005423728813559322|
|0.6180|0.1720|0.002|0.0002711864406779661|"""
    getSingleMeanAndVariance(exp004t)
    getSingleMeanAndVariance(exp01t)
    getSingleMeanAndVariance(wta004t)
    getSingleMeanAndVariance(wta01t)

def calculateCTGSCGPT():
    text="""|0.04|0.004338983050847458|0.1375|0.032|0.004338983050847458|
|0.04|0.0|0.1569|0.0|0.0|
|0.04|0.0|0.1563|0.0|0.0|
|0.04|0.0|0.1556|0.0|0.0|
|0.04|0.0|0.1563|0.0|0.0|
|0.1|0.0002711864406779661|0.1386|0.002|0.0002711864406779661|
|0.1|0.010033898305084745|0.1554|0.074|0.010033898305084745|
|0.1|0.010305084745762711|0.1539|0.075|0.010305084745762711|
|0.1|0.010033898305084745|0.1540|0.073|0.010033898305084745|
|0.1|0.010305084745762711|0.1557|0.073|0.010305084745762711|"""
    get2PointkMeanAndVariance(text)

def calculateParaprasedSCGPT():
    text1="""|0.04|0.0|0.1434|0.0|0.0|
|0.04|0.0|0.1426|0.0|0.0|
|0.04|0.0|0.1435|0.0|0.0|
|0.04|0.0|0.1427|0.0|0.0|
|0.04|0.0|0.1439|0.0|0.0|
|0.1|0.006101694915254237|0.1247|0.044|0.006101694915254237|
|0.1|0.0050169491525423725|0.1255|0.037|0.0050169491525423725|
|0.1|0.005830508474576271|0.1254|0.042|0.005830508474576271|
|0.1|0.007050847457627118|0.1250|0.05|0.007050847457627118|
|0.1|0.007050847457627118|0.1250|0.05|0.007050847457627118|"""
    get2PointkMeanAndVariance(text1)

    text2="""|0.04|0.0|0.1472|0.0|0.0|
|0.04|0.0|0.1469|0.0|0.0|
|0.04|0.0|0.1473|0.0|0.0|
|0.04|0.0|0.1468|0.0|0.0|
|0.04|0.0|0.1478|0.0|0.0|
|0.1|0.0006779661016949153|0.1487|0.005|0.0006779661016949153|
|0.1|0.0013559322033898306|0.1490|0.01|0.0013559322033898306|
|0.1|0.0013559322033898306|0.1483|0.01|0.0013559322033898306|
|0.1|0.0009491525423728814|0.1484|0.007|0.0009491525423728814|
|0.1|0.0013559322033898306|0.1483|0.01|0.0013559322033898306|"""
    get2PointkMeanAndVariance(text2)

def calculateParaprasedDAMD():
    text1="""|0.04|0.0|0.1766|-1000|0.0|
|0.04|0.0|0.1699|-1000|0.0|
|0.04|0.0|0.1721|-1000|0.0|
|0.04|0.0|0.1711|-1000|0.0|
|0.04|0.0|0.1689|-1000|0.0|
|0.1|0.0027196083763937995|0.1606|-1000|0.0027196083763937995|
|0.1|0.0020397062822953495|0.1702|-1000|0.0020397062822953495|
|0.1|0.0028555887952134893|0.1596|-1000|0.0028555887952134893|
|0.1|0.002991569214033179|0.1712|-1000|0.002991569214033179|
|0.1|0.005167255915148218|0.1629|-1000|0.005167255915148218|"""
    get2PointkMeanAndVariance(text1)

    text2="""|0.04|0.0|0.1604|-1000|0.0|
|0.04|0.0|0.1543|-1000|0.0|
|0.04|0.0|0.1623|-1000|0.0|
|0.04|0.0|0.1565|-1000|0.0|
|0.04|0.0|0.1557|-1000|0.0|
|0.1|0.0014957846070165896|0.1288|-1000|0.0014957846070165896|
|0.1|0.0|0.1481|-1000|0.0|
|0.1|0.0005439216752787598|0.1396|-1000|0.0005439216752787598|
|0.1|0.0006799020940984499|0.1511|-1000|0.0006799020940984499|
|0.1|0.00013598041881968996|0.1410|-1000|0.00013598041881968996|"""
    get2PointkMeanAndVariance(text2)

def calculateParaprasedRAWDAMD():
    text1="""|0.04|0.0|0.1926|-1000|0.0|
|0.04|0.0|0.1874|-1000|0.0|
|0.04|0.0|0.1892|-1000|0.0|
|0.04|0.0|0.1881|-1000|0.0|
|0.04|0.0|0.1861|-1000|0.0|
|0.1|0.003671471308131629|0.1583|-1000|0.003671471308131629|
|0.1|0.0014957846070165896|0.1856|-1000|0.0014957846070165896|
|0.1|0.0035354908893119393|0.1744|-1000|0.0035354908893119393|
|0.1|0.002991569214033179|0.1872|-1000|0.002991569214033179|
|0.1|0.005711177590426979|0.1777|-1000|0.005711177590426979|"""
    get2PointkMeanAndVariance(text1)

    # text2=""""""
    # get2PointkMeanAndVariance(text2)

def calculateParaprasedHpollutedSOLOIST():
    text1="""| 0.04 | 0.6820 |     0.1759 | 0.003 | 4.068e-4 |
| 0.04 | 0.6790 |     0.1771 | 0.003 | 4.068e-4 |
| 0.04 | 0.6830 |     0.1767 | 0.004 | 5.424e-4 |
| 0.04 | 0.6790 |     0.1759 | 0.003 | 4.068e-4 |
| 0.04 | 0.6830 |     0.1762 | 0.004 | 5.424e-4 |
|  0.1 | 0.6420 |     0.1767 |  0.01 |   0.0014 |
|  0.1 | 0.6480 |     0.1773 | 0.013 |   0.0018 |
|  0.1 | 0.6470 |     0.1762 |  0.01 |   0.0014 |
|  0.1 | 0.6460 |     0.1758 | 0.016 |   0.0022 |
|  0.1 | 0.6460 |     0.1761 | 0.009 |   0.0012 |"""
    get2PointkMeanAndVariance(text1)

def calculateParaprasedHpollutedSOLOIST_temp():
    text1="""| 0.04 |  0.6440 |  0.1704 | 0.008 | 0.001084   |
| 0.04 | 0.6780 | 0.1714 | 0.002 |  0.0002711864406779661 |
| 0.04 | 0.6780 | 0.1702 | 0.001 | 0.00013559322033898305 |
| 0.04 | 0.6820 | 0.1705 | 0.001 | 0.00013559322033898305 |
| 0.04 | 0.6820 | 0.1709 | 0.003 | 0.00040677966101694915 |
|  0.1 | 0.6460 | 0.1690 | 0.006 |  0.0008135593220338983 |
|  0.1 | 0.6460 | 0.1689 | 0.005 |  0.0006779661016949153 |
|  0.1 | 0.6420 | 0.1698 | 0.005 |  0.0006779661016949153 |
|  0.1 | 0.6480 | 0.1700 | 0.005 |  0.0006779661016949153 |
|  0.1 | 0.6470 | 0.1695 | 0.005 |  0.0006779661016949153 |"""
    get2PointkMeanAndVariance(text1)

    text1="""|0.04|0.6520|0.1654|0.0|0.0|
|0.04|0.6490|0.1662|0.002|0.0002711864406779661|
|0.04|0.6510|0.1651|0.0|0.0|
|0.04|0.6530|0.1658|0.001|0.00013559322033898305|
|0.04|0.6540|0.1660|0.002|0.0002711864406779661|
|0.1|0.6160|0.1722|0.002|0.0002711864406779661|
|0.1|0.6160|0.1718|0.001|0.00013559322033898305|
|0.1|0.6150|0.1726|0.0|0.0|
|0.1|0.6180|0.1729|0.004|0.0005423728813559322|
|0.1|0.6180|0.1720|0.002|0.0002711864406779661|"""
    get2PointkMeanAndVariance(text1)


def calculate_temp0416():
#     text="""|0.04|0.7100|0.1681|0.001|0.00013559322033898305|
# |0.04|0.7120|0.1685|0.002|0.0002711864406779661|
# |0.04|0.7080|0.1675|0.001|0.00013559322033898305|
# |0.04|0.7110|0.1681|0.003|0.00040677966101694915|
# |0.04|0.7090|0.1686|0.003|0.00040677966101694915|
# |0.1|0.7150|0.1716|0.002|0.0002711864406779661|
# |0.1|0.7170|0.1715|0.003|0.00040677966101694915|
# |0.1|0.7130|0.1711|0.002|0.0002711864406779661|
# |0.1|0.7160|0.1718|0.004|0.0005423728813559322|
# |0.1|0.7150|0.1718|0.003|0.00040677966101694915|"""
#     get2PointkMeanAndVariance(text)

    text2="""|   72.00 | 17.90 | 0.006 | 8.136e-4 |
|   72.10 | 17.88 | 0.010 | 0.001356 |
|    71.7 | 17.79 | 0.008 | 0.001085 |
|    72.1 | 17.87 | 0.010 | 0.001356 |
|    71.9 | 17.85 | 0.011 | 0.001492 |"""
    getSingleMeanAndVariance(text2)



    
def ParserHP_SOLOIST(filename):
    success_list=[]
    delexbleu_list=[]
    dpr_list=[]
    rpr_list=[]
    with open(filename,'r') as f:
        data=f.readlines()
    for line in data:
        if "success: " in line:
            success_list.append(line.split("success: ")[1].split("\n")[0])
            continue
        if "delex bleu: " in line:
            delexbleu_list.append(line.split("delex bleu: ")[1].split("\n")[0])
            continue
    #     if "DPR" in line:
    #         a=line.split("	 RPR: ")[0]
    #         b=line.split("	 RPR: ")[1]
    #         dpr_list.append(a.split("DPR: ")[1])
    #         rpr_list.append(b.split("\n")[0])
    #         continue
    # success_list=rpr_list
    dpr_list=[-1 for _ in success_list]
    rpr_list=[-1 for _ in success_list]

    print(len(success_list),len(delexbleu_list),
          len(dpr_list),len(rpr_list))
    
    assert(len(success_list)==len(dpr_list))
    assert(len(delexbleu_list)==len(dpr_list))
    assert(len(rpr_list)==len(dpr_list))


    text=""
    for i in range(len(success_list)):
        if i%2==0:
            text+="+-+-+-+-+-+\n"
            text+=f"|0.04|{success_list[i]}|{delexbleu_list[i]}|{dpr_list[i]}|{rpr_list[i]}|\n"
        else:
            text+="+-+-+-+-+-+\n"
            text+=f"|0.1|{success_list[i]}|{delexbleu_list[i]}|{dpr_list[i]}|{rpr_list[i]}|\n"
            
    print(text)


        
if __name__=="__main__":
    # # parser("1219_soloist_results.log",xxx=1)
    # parser("1215_training_results.log",xxx=3)
    # calculateMain()

    # parser("../1225_original_3.log",xxx=1)
    # calculateMain()

    # singleParser("./1229_tempering.log")


    # parserLSTM("./1225_results.log",xxx=1)
    # calculatelstm()

    # parserLSTM("./1229_pure_sclstm.log",xxx=0)
    # calculatelstm()

    # parserChange("./1231_change_targetnum.log")
    # calculateChangeTargetNum()

    # parserChange("./1231_change_temperingnum.log")
    # calculateChangeTargetNum()

    # calculateSCGPT()
    
    # ParserParaphrasedSCGPT("./0307_scgpt_paraphrasing_evaluation.log")
    # calculateParaprasedSCGPT()

    # ParserParaphrasedDAMD("./0312-evaluate_damd.log")
    # calculateParaprasedDAMD()
    # ParserParaphrasedDAMD("./0313-raw-damd.log")
    # calculateParaprasedRAWDAMD()

    # ParserParaphrasedSCGPT("./0405-ctg-scgpt.log")
    # calculateCTGSCGPT()

    # ParserHP_SOLOIST("../0410-testing-new-pollution.log")
    # calculateParaprasedHpollutedSOLOIST()

    # ParserParaphrasedHsoloistTEMP("./0411.test-soloist-hpollution-temp.log")
    # calculateParaprasedHpollutedSOLOIST_temp()
    
    # ParserParaphrasedHsoloistTEMP()
    # calculateParaprasedHpollutedSOLOIST_temp()


    # singleParser("./0416_new_vanilla-soloist-temp-results.log")

    # temp_parser("./0414-buchong-0.0401-for-temp.log")
    # print("--------------------------")
    # temp_parser("./0412-new-target5-temp-H-training.log")

    # calculate_temp0416()
    # calculateHTEMPSOLOIST()

    # temp_parser("../simpletod/0422-evaluation-simpletod.log")
    # temp_parser("./0424-temp-simpletod-inference.log")

    # calculateSimpleTODRelated()
    # temp_parser("./0411-random-temp-new-evaluation.log")
    # calculateTEMPrandomSampling()

    # temp_parser("./0517-evaluate-roberta-results.log")
    # temp_parser("./0520-roberta-results.log")
    # calculateRoberta() # roberta-based for H-pollution soloist
    calculateParaprasedHpollutedSOLOIST_temp()
