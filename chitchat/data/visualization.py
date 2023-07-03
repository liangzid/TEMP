from time import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict,Counter

def tsneVisulization(data_embeddings,labels1,labels2,title="test"):
    """Generate T-SNE visualization Images."""

    # 1. get data to be visulaized
    ## nothing
    
    ## 2. dimension reduce by TSNE
    tsne = TSNE(n_components=2,
                      learning_rate='auto',
                      init='pca',
                      perplexity=15)
    t0=time()
    xEmbeddings=tsne.fit_transform(data_embeddings)
    t1=time()
    print("tsne time cost",t1-t0)

    ## visualized by methods.
    _plot_embedds(xEmbeddings,labels1,labels2,title)
    
def _plot_embedds(data, labels1,labels2,title):
    labels1=["Safe" if x==1 else "Unsafe" for x in labels1]
    labels2=[str(x) for x in labels2]
    import seaborn as sns
    plot_data={"x":data[:,0],"y":data[:,1],"label1":labels1,"label2":labels2,}
    fig=sns.scatterplot(x="x",y="y",data=plot_data,hue="label2",style="label1")
    fig=fig.get_figure()
    fig.savefig(title)
    plt.clf()
    del fig
    

def forPaperVisualization():
    from parser_safe_dataset import SafeTrainCorpus 
    stc=SafeTrainCorpus()

    ## 1. first load the data!
    fname="temp_cluster_datasave.pkl"
    with open(fname,'rb') as f:
        data=pickle.load(f)

    for figIndex,perdata in enumerate(data):
        idxls,new_resps,r_embedds,topK_cluster_keys=perdata
        topK_cluster_keys=topK_cluster_keys[:10]

        adict=Counter(idxls)
        print(f"Cluster Distribution: {adict}")

        ## 2. to the format of t-sne
        this_cluster_data=[]
        this_cluster_safelabel=[]
        this_cluster_cluster_label=[]
        x=0
        max_keys=max(idxls)
        for ky in topK_cluster_keys:
        # for ky in range(max_keys+1):
            for i,id in enumerate(idxls):
                if id == ky:
                    safe_label=stc._findSafetylabel(new_resps[i])
                    this_cluster_data.append(r_embedds[i])
                    this_cluster_safelabel.append(safe_label)
                    this_cluster_cluster_label.append(x)
            x+=1
            x=min([4,x])
        print(len(this_cluster_data))
        title=f"./figs/{figIndex}.png"

        this_cluster_data=np.array(this_cluster_data)
        ## 3. running t-sne and evaluation
        tsneVisulization(this_cluster_data,
                         this_cluster_safelabel,this_cluster_cluster_label,title)
    

def selectedVisualization(hard_or_simple="simple"):
    from parser_safe_dataset import SafeTrainCorpus 
    stc=SafeTrainCorpus()

    ## 1. first load the data!
    fname="temp_cluster_datasave.pkl"
    with open(fname,'rb') as f:
        data=pickle.load(f)


    ## 1.1. select a subset of visualization
    if hard_or_simple=="hard":
        simple_samples_ls=[80,98,48,1]
        ppls=[9,20,10,5]
        sub_figs=[141,142,143,144]
        title=f"./figs/hard_sample.png"
        sns.set_style("dark")
        font_size=14
        sns.set_theme(style="dark", font='Times New Roman',
                      palette='muted',
                      rc={'font.size': font_size-2,
                          'axes.labelsize': font_size,
                          'axes.titlesize': font_size,
                          'xtick.labelsize': font_size,
                          'ytick.labelsize': font_size,
                          'legend.fontsize': font_size})

        sns.palplot(sns.color_palette("Paired"))
    else:
        simple_samples_ls=[5,12,38,68]
        sub_figs=[141,142,143,144]
        ppls=[25, 6,6,25]
        title=f"./figs/simple_sample.png"
    newdata=[]

    for index in simple_samples_ls:
        for i,x in enumerate(data):
            if i ==index:
                newdata.append(x)
                break
    data=newdata

    fig = plt.figure(dpi=150,figsize=(20,4))
    # fig=plt.figure(figsize=(14,4))
    plt.subplots_adjust(wspace =0, hspace =0) #调整子图间距
    plt.tight_layout() #调整整体空白
    for figIndex,perdata in enumerate(data):

        plt.subplot(sub_figs[figIndex])
        
        idxls,new_resps,r_embedds,topK_cluster_keys=perdata
        topK_cluster_keys=topK_cluster_keys[:10]

        adict=Counter(idxls)
        print(f"Cluster Distribution: {adict}")

        ## 2. to the format of t-sne
        this_cluster_data=[]
        this_cluster_safelabel=[]
        this_cluster_cluster_label=[]
        x=0
        max_keys=max(idxls)
        for ky in topK_cluster_keys:
        # for ky in range(max_keys+1):
            for i,id in enumerate(idxls):
                if id == ky:
                    safe_label=stc._findSafetylabel(new_resps[i])
                    this_cluster_data.append(r_embedds[i])
                    this_cluster_safelabel.append(safe_label)
                    this_cluster_cluster_label.append(x)
            x+=1
            x=min([4,x])
        print(len(this_cluster_data))

        ## 2.9. label mapping
        new_label=[]
        for i,x in enumerate(this_cluster_safelabel):
            if x==1:
                new_label.append("Safe")
            else:
                new_label.append("Unsafe")
        this_cluster_safelabel=new_label

        new_label=[]
        for i,x in enumerate(this_cluster_cluster_label):
            if x==0:
                new_label.append("1-st Cluster")
            elif x==1:
                new_label.append("2-nd Cluster")
            elif x==2:
                new_label.append("3-rd Cluster")
            else:
                new_label.append("Others")
        this_cluster_cluster_label=new_label

        this_cluster_data=np.array(this_cluster_data)
        ## 3. running t-sne and evaluation
    
        ## 3.2. dimension reduce by TSNE
        tsne = TSNE(n_components=2,
                        learning_rate='auto',
                        init='pca',
                        perplexity=ppls[figIndex])
        t0=time()
        xEmbeddings=tsne.fit_transform(this_cluster_data)
        t1=time()
        print("tsne time cost",t1-t0)

        ## 3.3 visualized by methods.
        plot_data={"x":xEmbeddings[:,0],
                   "y":xEmbeddings[:,1],
                   "CLUSTER TYPE":this_cluster_cluster_label,
                   "SAFETY":this_cluster_safelabel,}
        # plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        if figIndex==3:
            fig=sns.scatterplot(x="x",y="y", s=150,
                                data=plot_data,
                                hue="CLUSTER TYPE",style="SAFETY")

            fig.set_xlabel("")
            fig.set_ylabel("")
            plt.legend(bbox_to_anchor=(1.01, 0),
                    loc=3, borderaxespad=0)
        else:
            fig=sns.scatterplot(x="x",y="y",s=150,
                                data=plot_data,
                                hue="CLUSTER TYPE",style="SAFETY",legend=False)
        # fig=fig.get_figure()
    plt.savefig(title)
    plt.clf()



if __name__=="__main__":
    # X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    # label=np.array([1,0,1,1])

    # tsneVisulization(X,label,label)

    # forPaperVisualization()
    # selectedVisualization()
    selectedVisualization(hard_or_simple="hard")

