import json
import torch
import numpy as np
import pickle

from tqdm import tqdm

# conda install -c pytorch faiss-cpu
import faiss
import random
import torch

from parser_safe_dataset import *

def retrieval_based_test_unsupervised(train_sets,test_sets):
    """
    this is the unsupervised version of url sampling.
    ---
    train_sets: (context,response,label) triple sets reading from dialogue corpus
    test_sets: every time we only know one examples. 
    """
    target_num=5
    sample_method="dynamic_sharpen"
    anns_num=150
    eps=0.22
    tau=1
    save_histogram_path="histogram_res.json"

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
    # sets=_shuffle(sets)
    sets=train_sets
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

    test_u_embedds=[]
    for (u,_,_) in test_sets:
        inputs=tokenizer(u,return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            outputs=model(**inputs,
                          return_dict=True)['pooler_output']\
                          .squeeze(0).cpu().detach().numpy()
        test_u_embedds.append(outputs)
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
    progress=tqdm(total=len(test_sets),desc="Test set inference preparation")
    for i,x in enumerate(test_sets):
        progress.update(1)
        u,r,l=x
        del l # which means unsupervised sampling
        similar_r=r
        distances,indexes=built_index.search(np.array([test_u_embedds[i]]),
                                                anns_num)
        indexes=indexes[0]

        ## 2. clustering Index, and distangle different dialogue responses.
        new_responses=[]
        r_embeddings=[]

        # get the original response index in the subset
        # resp_new_idx=np.where(indexes==i)[0][0]
        resp_new_idx=-1
        # print(resp_new_idx)

        for idx in indexes:
            # new_responses.append((u_list[idx],r_list[idx]))
            new_responses.append(r_list[idx])
            r_embeddings.append(big_r_embeddings[idx])
        # print("related responses",new_responses)
        idxls=make_clustering(new_responses,r_embeddings,eps=eps)

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


if __name__=="__main__":
    ## reading training and test corpora.
    fraction=0.1
    apath=f"/home/liangzi/adc/DiaSafety/DiaSafety_dataset/train_fraction{fraction}.json"
    sets=parseSafeJson2TripleLs(apath)
    test_p=f"/home/liangzi/adc/DiaSafety/DiaSafety_dataset/test.json"
    test_sets=parseSafeJson2TripleLs(test_p)

    new_triples,head_resls=retrieval_based_test_unsupervised(sets,test_sets)
    print(f"safe rate:{sum(head_resls)/len(head_resls)}")

    # no storage of ortacle results.
