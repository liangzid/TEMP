import gensim.downloader as loader
import numpy as np
import faiss
import pickle

class retrievalModel:

    def __init__(self):
        self.msl=128
        self.d=200
        self.corpus=loader.load("glove-twitter-200")
        self.inducted_template_list_path="/home/szhang/datasets/soloist/inducted_templates.pk"
        print("model-load-done")
        self._load_inducted_templates()

        # construct indexes.
        self.INDEX=faiss.IndexFlatL2(200)
        temp_data=self.getTemplateMatrix(self.inducted_templates)
        print(temp_data)
        print(temp_data.shape)
        self.INDEX.add(np.array(temp_data))
        print("index constructed done.")
        
        
    def _load_inducted_templates(self):
        with open(self.inducted_template_list_path,'rb') as f:
            self.inducted_templates=pickle.load(f)

    def searchIndex(self,query,k=3):
        D,I=self.INDEX.search(query,k)
        return I

    def Index2WordEmbeddingBatch(self,I):
        shapes=I.shape
        if len(shapes==1):
            subtems=self.inducted_templates[I]
            k=shapes[0]
            embeds=np.zeros((k,self.msl,200))
            for i in range(k):
                embeds[k,:,:]=self.getSentWordEmbed(subtems[i],msl=self.msl)
        elif len(shapes==2):
            batch,k=shapes
            overall_embeds=np.zeros((batch,k,self.msl,200))
            for b in range(batch):
                subtems=self.inducted_templates[I[b]]
                embeds=np.zeros((k,self.msl,200))
                for i in range(k):
                    embeds[k,:,:]=self.getSentWordEmbed(subtems[i],msl=self.msl)
                overall_embeds[b,:,:,:]=embeds
            return overall_embeds
                

    def getwv(self,word):
        if word in self.corpus:
            return self.corpus[word]
        else:
            return np.zeros(200)

    def getSentWordEmbed(self,sentence,msl=128):
        sentence=self.deleteChar(sentence,"[")
        sentence=self.deleteChar(sentence,"]")
        embedding=np.zeros((msl,200))
        ss=sentence.split()
        for i in range(msl):
            if i<len(ss):
                embedding[i,:]=self.getwv(ss[i])
        return embedding

    def getAverageSentWordEmbed(self,sentencels):
        print(sentencels)
        embedding=np.zeros(200,dtype=np.float32)
        for w in sentencels:
            w=self.deleteChar(w,"[")
            w=self.deleteChar(w,"]")
            embed=self.getwv(w)
            embedding+=embed
        # print(embedding)
        if len(sentencels)!=0:
            return embedding/len(sentencels)
        else:
            return embedding

    def getAverageSentVector(self,sentence):
        sentence=self.deleteChar(sentence,"[")
        sentence=self.deleteChar(sentence,"]")
        embedding=np.zeros(200)
        ss=sentence.split()
        for word in ss:
            embedding+=self.getwv(word)
        return embedding

    def deleteChar(self,sent,c):
        if c in sent:
            ss=sent.split(c)
            result=""
            for ele in ss:
                result+=ele
            return result
        else:
            return sent
        
    def getTemplateMatrix(self,inducted_templates):
        v_list=[]
        for tem in inducted_templates:
            v_list.append(self.getAverageSentWordEmbed(tem))
        return np.array(v_list)


if __name__=="__main__":
    # corpus=loader.load("glove-twitter-200")
    # print(corpus["hello"])

    mymodel=retrievalModel()
    s1="Recommend you to [place], its phone number is [number]."
    s2="Maybe you like to go to [place], contract it with [number]."
    s3="The phone of [place] is [number]."
    # v1=mymodel.getAverageSentVector(s1)
    # v2=mymodel.getAverageSentVector(s2)
    # v3=mymodel.getAverageSentVector(s3)
    # print(np.dot(v1,v2),np.dot(v1,v3),np.dot(v2,v3))
