from bisect import bisect
from collections import Counter
import random
from math import exp as exp
from copy import copy
import numpy as np
# mysort_file=[1,2,3,4,5,6,9,10]


# x=bisect(mysort_file,11)
# print(x)
# print(random.random())

# def winnerTakeAll(elels):
#     lens=len(elels)
#     ct=Counter(elels)
#     ele_ls=[]
#     p_ls=[]

#     max_index=0
#     maxnum=-1
#     for ele in ct:
#         if ct[ele]>maxnum:
#             max_index=elels.index(ele)
#             maxnum=ct[ele]
#     return max_index

# mytest=[1,1,1,1,1,1,1,3,4,5]

# result=winnerTakeAll(mytest)
# print(result)

# def sample_exp(elels):
#     lens=len(elels)
#     ct=Counter(elels)
#     ele_ls=[]
#     p_ls=[]

#     # print(ct)
#     for ele in ct:
#         ele_ls.append(ele)
#         p_ls.append(ct[ele])

#     # print(p_ls)
#     new_p_ls=softmax(p_ls)
#     accumulate_ls=copy(new_p_ls)
#     for i,p in enumerate(new_p_ls):
#         if i==0:
#             continue
#         else:
#             accumulate_ls[i]=accumulate_ls[i]+accumulate_ls[i-1]
#     random_result=random.random()*accumulate_ls[-1]
#     index=bisect(accumulate_ls,random_result)
#     return ele_ls[index]
#     # return index

# def softmax(ls):
#     newls=[]
#     for l in ls:
#         newls.append(exp(l))
#     sum_result=sum(newls)+10e-5
#     newnewls=[]
#     for nnl in newls:
#         newnewls.append(nnl/sum_result)
#     # print(newnewls)
#     return newnewls
        
# result=sample_exp(mytest)
# print(result)

nums=7
fraction=1
index_list=np.random.randint(low=0,high=nums,size=int(fraction*nums))
il2=np.random.choice(range(nums),int(fraction*nums),replace=False)
print(index_list)
print(il2)
