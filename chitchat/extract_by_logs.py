

def getInformationByLogs(file_name):
    with open(file_name,'r',encoding='utf8') as f:
        data=f.readlines()
    # result_dict constructure:
    #     -- dialogue models
    #        -- CLS methods
    #            -- use ADC or not
    tempd={"none":{},"random":{},"detoxify":{},"perspectiveAPI":{},
           "BBF":{},"BAD":{}}
    result_dict_cls={"raw":tempd.copy(),"blenderbot-80M":tempd.copy(),
                     "DialoGPT-medium":tempd.copy()}
    result_dict_adc={"raw":{"none":{},"BAD":{}},
                     "blenderbot-80M":{"none":{},"BAD":{}},
                     "DialoGPT-medium":{"none":{},"BAD":{}}}
    metrics_res={}
    for line in data:
        if "is using ADC:" in line:
            if "is using ADC:1" in line:
                using_adc=1
            else:
                using_adc=0
        if "before adc, we add a filter called:" in line:
            cls_name=line.split(":")[1].replace("\n","")
        if "the file of the dialogue model is:" in line:
            dialogue_name=line.split(":")[1].replace("\n","")
        
        #now collect the dialogue metrics
        if "the unsupervised metrics" not in line:
            if "Dist_1" in line:
                # print(line)
                metrics_res['Dist_1']=line.split("Dist_1: ")[1].replace("\n","")
            elif "Dist_2" in line:
                metrics_res['Dist_2']=line.split("Dist_2: ")[1].replace("\n","")
            elif "Avglen" in line:
                metrics_res['Avglen']=line.split("Avglen: ")[1].replace("\n","")
            elif "Entropy" in line:
                metrics_res['Entropy']=line.split("Entropy: ")[1].replace("\n","")
            elif ">>>Safety score: " in line:
                metrics_res['safety']=line.split(": ")[1].replace("\n","")
            elif ">>>Cons score: " in line:
                metrics_res['cons']=line.split(": ")[1].replace("\n","")

                if using_adc==1:
                    print(dialogue_name)
                    result_dict_adc[dialogue_name][cls_name]=metrics_res
                    metrics_res={}
                else:
                    result_dict_cls[dialogue_name][cls_name]=metrics_res
                    metrics_res={}
    return result_dict_adc, result_dict_cls
            

if __name__=="__main__":
    a,b=getInformationByLogs("./1013---allres.log")
    from pprint import pprint as pp
    pp(a)
    pp(b)




