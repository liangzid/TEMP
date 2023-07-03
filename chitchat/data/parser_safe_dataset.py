import json
import torch
import numpy as np
import pickle

from tqdm import tqdm

# conda install -c pytorch faiss-cpu
import faiss
import random
import torch

def _shuffle(sets):
    index=np.random.permutation(len(sets))
    newsets=[]
    for x in index:
        newsets.append(sets[x])
    return newsets 

def parseSafeJson2TripleLs(src_path="/home/liangzi/adc/DiaSafety/DiaSafety_dataset/train.json"):
    with open(src_path,'r',encoding='utf8') as f:
        data=json.load(f)
    sets=[]
    for x in data:
        label=0 if x['label']=="Safe" else 1
        sets.append((x['context'],x['response'],label))
    return sets

def UnsupervisedSampleURLTriple(sets,
                                target_num=5,
                                sample_method="dynamic_sharpen",
                                anns_num=150,
                                eps=0.22,
                                tau=1,save_histogram_path="histogram_res.json"):
    """
    this is the unsupervised version of url sampling.
    ---
    sets: (context,response,label) triple sets reading from dialogue corpus
    ---
    new_triple: <utterance, unsafe response, related safe response> triple.
    """
    print(f"target num: {target_num}")

    ## 0. initilzation.
    # load similarity calculation NLU models
    # from transformers import RobertaForSequenceClassification
    from transformers import RobertaModel
    from transformers import RobertaTokenizer

    frmpth="/home/liangzi/models/roberta-base"
    DEVICE=torch.device("cuda:6")

    model=RobertaModel.from_pretrained(frmpth)
    tokenizer=RobertaTokenizer.from_pretrained(frmpth)
    model.to(DEVICE)
    
    print(">> Text Embedding Model Load Done.")

    ## shuffle `sets`
    sets=_shuffle(sets)
    print(">> Shuffle Done.")
    
    # calculate the representations between all input utterance samples.
    n_samples=len(sets)
    u_list=[]
    r_list=[]
    l_list=[]
    u_embeddings=[]
    for (u,r,l) in sets:
        u_list.append(u)
        r_list.append(r)
        l_list.append(l)
        inputs=tokenizer(u,return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            outputs=model(**inputs,
                          return_dict=True)['pooler_output']\
                          .squeeze(0).cpu().detach().numpy()

        u_embeddings.append(outputs)
        del inputs
        del outputs

    dim=len(u_embeddings[0])
    u_embeddings=np.array(u_embeddings)

    ## calculate the representation of content clustering
    big_r_embeddings=[]
    for r in r_list:
        inputs=tokenizer(r,
                         return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            outputs=model(**inputs,
                        return_dict=True)['pooler_output']\
                        .squeeze(0).cpu().detach().numpy()
        big_r_embeddings.append(outputs)
    big_r_embeddings=np.array(big_r_embeddings)

    # built_index=faiss.IndexFlatL2(dim)
    # built_index.add(u_embeddings)
    built_index=faiss.IndexFlatIP(dim)
    faiss.normalize_L2(u_embeddings)
    built_index.add(u_embeddings)
    
    stc=SafeTrainCorpus()
    # varibale definition
    new_triple=[]

    counter_num=0.
    head_safe_fraction_ls=[]
    # histogram_distribution_ls=[]
    for_visualization_ls=[]
    progress=tqdm(total=len(sets),desc="Training set sampling preparation")
    for i,x in enumerate(sets):
        # if i>1:
            # break
        
        progress.update(1)
        u,r,l=x
        del l # which means unsupervised sampling
        similar_r=r

        distances,indexes=built_index.search(np.array([u_embeddings[i]]),
                                                anns_num)
        indexes=indexes[0]

        # -------------------------------------------------------------
        # u="Soon I'll be dead. You think so but happen in a very short period of time."
        # inputs=tokenizer(u,
        #                  return_tensors='pt').to(DEVICE)
        # with torch.no_grad():
        #     outputs=model(**inputs,
        #                 return_dict=True)['pooler_output']\
        #                 .squeeze(0).cpu().detach().numpy()
        # distances,indexes=built_index.search(np.array([outputs]),
        #                                         30)
        # indexes=indexes[0]
        # -------------------------------------------------------------

        ## 2. clustering Index, and distangle different dialogue responses.
        new_responses=[]
        r_embeddings=[]

        # get the original response index in the subset
        resp_new_idx=np.where(indexes==i)[0][0]
        # print(resp_new_idx)

        for idx in indexes:
            # new_responses.append((u_list[idx],r_list[idx]))
            new_responses.append(r_list[idx])
            r_embeddings.append(big_r_embeddings[idx])
        # print("related responses",new_responses)
        idxls=make_clustering(new_responses,r_embeddings,eps=eps)
        # print(f"the index: {idxls}")
        # displayLongtailClsters(new_responses,idxls)
        
        # # collect the histogram graph
        # adict=Counter(idxls)
        # total_num=sum(adict.values())
        # keys=adict.most_common(total_num)
        # label=stc._findFinegrainCLSLabel(r)
        # cluster_num_distribution=[]
        # cluster_key_and_type_distribution=[]
        # for this_i,(idxx,fre) in enumerate(keys):
        #     cluster_num_distribution.append(fre)
        #     cluster_key_and_type_distribution.append((idxx,label))
        # # print(f"the distributions: {cluster_num_distribution}")
        # histogram_distribution_ls.append((cluster_num_distribution,
        #                                   cluster_key_and_type_distribution))

        # safeNum=calSafeFractionHead(new_responses,idxls)
        # # print(f"num: {num}")
        # head_safe_fraction_ls.append(safeNum)

        ## 3. and finally, sampling the responses with their responses.
        candidates=[]
        # select top-1 clustering
        # subresps=biggest_clsters(new_responses,idxls)
        # print(subresps)
        topK_cluster_keys=selectTopKClusters(new_responses,idxls,topk=10)

        # ## save some variable for visualization
        # for_visualization_ls.append((idxls,new_responses,
        #                              r_embeddings,topK_cluster_keys))

        candidates,safe_num=samplingWithClusters(topK_cluster_keys,
                                        resp_new_idx,
                                        new_responses,idxls,
                                        target_num=target_num,
                                        is_look_safe=1,
                                        sample_method=sample_method,
                                        tau=tau)
        head_safe_fraction_ls.append(safe_num)
        # candidates=biggest_clsters(new_responses,idxls)[:target_num]
        # print(candidates)

        # # sampling with the similairity
        # for j, aindex in enumerate(indexes):
        #     if r_list[aindex] in subresps:
        #         candidates.append(r_list[aindex])
        #         if len(candidates)>target_num:
        #             break

        # from pprint import pprint
        
        # indexx=random.randint(0,len(candidates)-1)
        # similar_r=candidates[indexx]
        new_triple.append((u,r,candidates))
            
    # with open("temp_cluster_datasave.pkl","wb") as f:
        # pickle.dump(for_visualization_ls,f)
    # with open(save_histogram_path,"wb") as f:
    #     pickle.dump(histogram_distribution_ls,f)

    return new_triple,head_safe_fraction_ls


from sklearn.cluster import DBSCAN
def make_clustering(resps,r_embedds,eps=0.22,):
    """
    return a pretrained model clustered List[List[str]] with inputs List[str]
    """
    # eps: The maximum distance between two samples for one to be
    #     considered as in the neighborhood of the other. 
    ## --> `larger` means less cluster numbers. 
    db = DBSCAN(eps=eps, min_samples=1).fit(r_embedds)
    labels=db.labels_
    # print(f"labels:{labels}")
    return labels

from collections import Counter,OrderedDict
def biggest_clsters(resps,indexls):
    """
    return the biggest clusters from indexls
    ---
    format of indexls: [0,1,2,1,0,0,0,1,2,4,0,...,1]
    """
    new_clsts=[]
    adict=Counter(indexls)
    # print(f"cluster distribution:   {adict}")
    # print(adict)
    key=adict.most_common(1)[0][0]
    for i, r in enumerate(resps):
        if indexls[i]==key:
            new_clsts.append(r)
    new_clsts.extend(new_clsts)
    return new_clsts

def selectTopKClusters(resps,indexls,topk=5):
    """
    return the biggest clusters from indexls
    ---
    format of indexls: [0,1,2,1,0,0,0,1,2,4,0,...,1]
    """
    new_clsts=[]
    keyls=[]
    adict=Counter(indexls)
    # print(f"cluster distribution:   {adict}")
    # print(adict)
    keys=adict.most_common(topk)
    for k in keys:
        keyls.append(k[0])
    return keyls

def samplingWithClusters(topK_cluster_keys,
                         resp_new_idx,
                         new_responses,
                         idxls,
                         target_num=1,
                         is_look_safe=0,sample_method="dynamic_sharpen",
                         tau=1):
    
    stc=SafeTrainCorpus()
    ## 1. first, consturct a key-resps dict
    key_respsls=OrderedDict({})
    for ky in topK_cluster_keys:
        key_respsls[ky]=[]
        for i,r in enumerate(new_responses):
            if idxls[i]==ky:
                key_respsls[ky].append(r)

    for ky in topK_cluster_keys[:1]:
        if resp_new_idx==-1:
            break
        new_resp=new_responses[resp_new_idx]
        if new_resp in key_respsls[ky]:
            return [new_resp \
                    for _ in range(target_num)],stc._findSafetylabel(new_resp)
                
    ## 2. then, calculate the sharpen coefficient
    # 2.1 get lengths of clusters
    keys=[key for key,x in key_respsls.items()]
    lens=[len(x) for key,x in key_respsls.items() ]

    # 2.2 calculate the sharpen indicator
    SI=(lens[0]-lens[1])/(max(lens[0]-lens[-1],1e-3))

    ## 3. calculate the cluster sample rate.
    if sample_method=="dynamic_sharpen":
        from torch.nn.functional import softmax as softmax
        clster_sample_distrib=softmax(\
                                torch.tensor(lens)/((1+1e-3-SI)*tau),
                                      dim=0)
    elif sample_method=="max":
        clster_sample_distrib=torch.zeros(len(lens))
        clster_sample_distrib[0]=1
    elif sample_method=="exp":
        from torch.nn.functional import softmax as softmax
        clster_sample_distrib=softmax(\
                                torch.tensor(lens)/(tau),
                                      dim=0)
    else:
        print("No others.")
        return -1
    # print(SI,clster_sample_distrib)

    ## 4. execute sampling.
    pseudo_labels=[]
    for i in range(target_num):
        # 4.1 first select a cluster
        index=torch.multinomial(clster_sample_distrib,1)[0]
        resps=key_respsls[keys[index]]

        # 4.2 then select a response from the cluster.
        randindex=random.randint(0,lens[index]-1)
        resp=resps[randindex]
        # print(len(resps))
        pseudo_labels.append(resp)
                
    if is_look_safe==0:
        return pseudo_labels,-1 
    else:
        safe_num=0.
        for x in pseudo_labels:
            safety_label=stc._findSafetylabel(x)
            safe_num+=safety_label

        return pseudo_labels,safe_num/len(pseudo_labels)

def displayLongtailClsters(resps, indexls):
    stc=SafeTrainCorpus()
    new_clsts=[]
    adict=Counter(indexls)
    print("----------------------------------------")
    total_num=sum(adict.values())
    print(f"total counter number: {total_num}")
    print(f"cluster distribution:   {adict}")
    # keys=adict.most_common(total_num)
    keys=adict.most_common(10)
    print(keys)
    key_respls={}
    for akey in keys:
        akey=akey[0]
        if akey not in key_respls:
            key_respls[akey]=[]
        for i,r in enumerate(resps):
            if indexls[i]==akey:
                key_respls[akey].append((r,stc._findSafetylabel(r)))
    from pprint import pprint
    pprint(OrderedDict(key_respls))

def calSafeFractionHead(resps,indexls):
    stc=SafeTrainCorpus()
    adict=Counter(indexls)
    # print("----------------------------------------")
    total_num=sum(adict.values())
    keys=adict.most_common(10)

    # key,fraction, dict
    key_fraction_dict={}

    for akey in keys:
        if akey[1]<2:
            break
        akey=akey[0]
        if akey not in key_fraction_dict:
            key_fraction_dict[akey]=[]
        for i,r in enumerate(resps):
            if indexls[i]==akey:
                safety_label=stc._findSafetylabel(r)
                key_fraction_dict[akey].append(safety_label)
                
    # from pprint import pprint
    # pprint(OrderedDict(key_fraction_dict))

    ## calculate the safe fraction
    newkfd={}
    for key in key_fraction_dict:
        num=sum(key_fraction_dict[key])/len(key_fraction_dict[key])
        newkfd[key]=num
        return num
    return 0.
    # newkfd=OrderedDict(newkfd)
    # return newkfd 


class SafeTrainCorpus:
    """
Give the safety label. This class is a metric.
    """
    def __init__(self):
        # first load training file.
        root_file="/home/liangzi/adc/"
        datafilepath="/DiaSafety/DiaSafety_dataset/train.json"

        # note that the `test_filepath` is only useful to ortacle scene.
        test_filepath="/DiaSafety/DiaSafety_dataset/test.json"

        with open(root_file+datafilepath,'r',encoding='utf8') as f:
            data=json.load(f)
        with open(root_file+test_filepath,'r',encoding='utf8') as f:
            data2=json.load(f)
        self.data=data
        self.data.extend(data2)
    
    def _findSafetylabel(self,r):
        for x in self.data:
            if x["response"]==r:
                if x["label"]=="Unsafe":
                    return 0
                else:
                    return 1
        return -1
    
    def _findFinegrainCLSLabel(self,r):
        for x in self.data:
            if x["response"]==r:
                cls=x["category"]
        return cls
    
def SampleURLTriple(sets,target_num=1):
    """
    sets: (context,response,label) triple sets reading from dialogue corpus
    ---
    new_triple: <utterance, unsafe response, related safe response> triple.
    """
    print(f"target num: {target_num}")

    ## 0. initilzation.
    # load similarity calculation NLU models
    # from transformers import RobertaForSequenceClassification
    from transformers import RobertaModel
    from transformers import RobertaTokenizer

    frmpth="/home/liangzi/models/roberta-base"
    DEVICE=torch.device("cuda:5")

    model=RobertaModel.from_pretrained(frmpth)
    tokenizer=RobertaTokenizer.from_pretrained(frmpth)
    model.to(DEVICE)
    
    print(">> Text Embedding Model Load Done.")

    # calculate the representations between all input utterance samples.
    n_samples=len(sets)
    u_list=[]
    r_list=[]
    l_list=[]
    u_embeddings=[]
    for (u,r,l) in sets:
        u_list.append(u)
        r_list.append(r)
        l_list.append(l)
        inputs=tokenizer(u,return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            outputs=model(**inputs,return_dict=True)['pooler_output'].squeeze(0).cpu().detach().numpy()
        u_embeddings.append(outputs)
        del inputs
        del outputs
    # print(len(u_embeddings))
    dim=len(u_embeddings[0])
    # print(dim)
    u_embeddings=np.array(u_embeddings)
    # print(u_embeddings.shape)

    # built_index=faiss.IndexFlatL2(dim)
    # built_index.add(u_embeddings)
    built_index=faiss.IndexFlatIP(dim)
    faiss.normalize_L2(u_embeddings)
    built_index.add(u_embeddings)
    
    # varibale definition
    new_triple=[]

    counter_num=0.
    for i,x in tqdm(enumerate(sets),desc="Training set sampling preparation"):
        u,r,l=x
        similar_r=r
        if l==1: # which means offensive
            distances,indexes=built_index.search(np.array([u_embeddings[i]]),
                                                 4000)
            indexes=indexes[0]

            
            candidates=[]
            for j, aindex in enumerate(indexes):
                # print(f"aindex: {aindex}")
                if aindex==i:
                    continue
                else:
                    
                    print()
                    if l_list[aindex]!=1:
                        # similar_r=r_list[aindex]
                        candidates.append(r_list[aindex])
                        if len(candidates)>target_num:
                            break
            indexx=random.randint(0,len(candidates)-1)
            similar_r=candidates[indexx]
        else:
            # continue
            # if not offensive, we also make a cluster.
            # print("++++++++++++++++++++++")
            # print(f"shape of u_embeddings: {u_embeddings.shape}")
            # print(f"shape of u_embeddings[0]: {u_embeddings[0].shape}")
            distances,indexes=built_index.search(np.array([u_embeddings[i]]),
                                                 4000)
            indexes=indexes[0] # before this step, the shape is (1,topk),after is topk
            candidates=[]
            for j, aindex in enumerate(indexes):
                # print("--------------------")
                # print(f"shape of indexes: {indexes.shape}")
                # print(f"shape of a index: {aindex.shape}")
                # print(f"aindex: {aindex}")
                if aindex==i:
                    continue
                else:
                    if l_list[aindex]!=1:
                        # similar_r=r_list[aindex]
                        candidates.append(r_list[aindex])
                        if len(candidates)>target_num:
                            break

            indexx=random.randint(0,len(candidates)-1)
            similar_r=candidates[indexx]
                        
        # if r==similar_r and l==1:
            # print(111111111111)
            # counter_num+=1

        # print(similar_r)
        # print("-----------------------------------")
        new_triple.append((u,r,candidates))
            
    # print(f"error number: {counter_num/9017}")
    # return -1
    return new_triple



def main1():

    # fractions=["0.01","0.02","0.04","0.05","0.08",
    #            "0.2","0.45","0.4",]

    fractions=["0.5","0.53","0.56","0.59","0.6","0.63","0.66","0.69","0.7",]

    target_num=1

    for fraction in fractions:
        sets=parseSafeJson2TripleLs(f"/home/liangzi/adc/DiaSafety/DiaSafety_dataset/train_fraction{fraction}.json")
    # new_triples=SampleURLTriple(sets)

        new_triples,head_safe_fraction_ls=UnsupervisedSampleURLTriple(sets,
                                                                    target_num=target_num)
        with open(f"./{fraction}special_triples_sampled_targetnum{target_num}.json", 'w',encoding='utf8') as f:
            json.dump({'data':new_triples},f,ensure_ascii=False)
        print("DONE.")
        print(f"head safe rate:{sum(head_safe_fraction_ls)/len(head_safe_fraction_ls)}")

# save the histogram
def main2():
    fraction="0.1"
    target_num=1
    sample_method="dynamic_sharpen"

    tau=0.1
    anns_num=150
    # eps=0.08
    eps=0.24

    print(f"sample method: {sample_method}\t tau:{tau}\t")

    categories=["Offending User","Risk Ignorance",
                "Unauthorized Expertise","Biased Opinion",
                "Toxicity Agreement"]
    for cate in categories:
        apath_prefix=f"/home/liangzi/adc/DiaSafety/DiaSafety_dataset/"
        apath=apath_prefix+f"0.1train_{cate}.json"

        sets=parseSafeJson2TripleLs(apath)

        new_triples,head_safe_fraction_ls=UnsupervisedSampleURLTriple(sets,
                                    target_num=target_num,
                                                    sample_method=sample_method,
                                                    anns_num=anns_num,
                                                    eps=eps,
                                                                      tau=tau,
                                save_histogram_path=f"./histogram_res_{cate}.pkl")
        print(f"Save to histogram_res_{cate}.pkl")
    
if __name__=="__main__":
    # main1()
    # main2()

    # fraction="0.8"
    fraction="origin"
    target_num=1
    # sample_method="dynamic_sharpen"
    # sample_method="max"
    sample_method="exp"
    # tau=1
    # tau=0.2
    tau=0.1
    anns_num=150
    # eps=0.08
    eps=0.22

    print(f"sample method: {sample_method}\t tau:{tau}\t")

    apath=f"/home/liangzi/adc/DiaSafety/DiaSafety_dataset/train_fraction{fraction}.json"
    sets=parseSafeJson2TripleLs(apath)
    # new_triples=SampleURLTriple(sets)

    new_triples,head_safe_fraction_ls=UnsupervisedSampleURLTriple(sets,
                                target_num=target_num,
                                                                  sample_method=sample_method,
                                                                  anns_num=anns_num,
                                                                  eps=eps,
                                                                  tau=tau)
    with open(f"./{fraction}special_triples_sampled_targetnum{target_num}.json", 'w',encoding='utf8') as f:
        json.dump({'data':new_triples},f,ensure_ascii=False)
    print("DONE.")
    print(f"safe rate:{sum(head_safe_fraction_ls)/len(head_safe_fraction_ls)}")
    # print(f"safe samples:{head_safe_fraction_ls}")
