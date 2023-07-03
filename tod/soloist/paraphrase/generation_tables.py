"""
This file is for generating proper Responses for ablation experiments.
"""
import random

def getAllGroundTruthAndResponses(filepath):
    with open(filepath,'r',encoding='utf8') as f:
        data=f.readlines()

    pair_ls=[]
    rd_ls=[]
    gtd_ls=[]
    for line in data:
        if "RD:" in line:
            rd_ls.append(line.split("RD:")[1].replace("1111 ","").replace("\n",""))
            continue
        if "GTD:" in line:
            gtd_ls.append(line.split("GTD:")[1].replace("1111","").replace("\n",""))
            continue

    assert len(rd_ls)==len(gtd_ls)
    
    return dict(zip(gtd_ls,rd_ls))


def orgtable2latextable(text):
    lines=text.split("\n")
    num=len(lines[0].split("|"))-2

    latex_text=r"""
\begin{table*}[]
\resizebox{\textwidth}{!}{
\begin{tabular}{ll}
\toprule
"""
    for line in lines:
        if line=="|-|-|":
            latex_text+=r" \hline"+"\n"
        else:
            element=line.split("|")
            t=""
            for ele in element:
                if ele=="" or ele=="\n":
                    continue
                else:
                    t+=f" {ele} &"
            latex_text+=t[:-1]+ r" \\"+ "\n"
    latex_text+="""
\end{tabular}
}
\bottomrule
\caption{ bla bla bla.}
\label{tab:example}
\end{table*}
"""
    print(latex_text)
            
        
def generateTable1():

    augpt_low="../0.04_predict_files_1.txt"
    exp_low="../mt3-tempering-exp-bp_fraction1.0_prate0.04_numid1.txt"
    wta_low="../mt3-tempering-wta-bp_fraction1.0_prate0.04_numid1.txt"

    augpt_high="../0.1_predict_files_1.txt"
    exp_high="../mt3-tempering-exp-bp_fraction1.0_prate0.1_numid1.txt"
    wta_high="../mt3-tempering-wta-bp_fraction1.0_prate0.1_numid1.txt"

    number=5

    toriginlow=getAllGroundTruthAndResponses(augpt_low)
    toriginhigh=getAllGroundTruthAndResponses(augpt_high)

    texplow=getAllGroundTruthAndResponses(exp_low)
    texphigh=getAllGroundTruthAndResponses(exp_high)

    twtalow=getAllGroundTruthAndResponses(wta_low)
    twtahigh=getAllGroundTruthAndResponses(wta_high)

    groundtruth_ls=list(toriginlow.keys())
    
    print(" LOW FRACTION")
    tabletext="|MODEL|TEXT|\n"
    tabletext+="|-|-|\n"
    for _ in range(number):
        index=random.randint(0,len(groundtruth_ls)-1)

        gtd=groundtruth_ls[index]
        while len(gtd.split(" "))>25:
            gtd=groundtruth_ls[index]

        od=toriginlow[gtd]
        ed=texplow[gtd]
        wd=twtalow[gtd]

        # tabletext+=f"|Reference|{gtd}|"+"\n"
        tabletext+=f"|Origin|{od}|"+"\n"
        tabletext+=f"|TEMP (exp)|{ed}|"+"\n"
        tabletext+=f"|TEMP (wta)|{wd}|"+"\n"

        tabletext+="|-|-|\n"

    print(tabletext)
    print(" HIGH FRACTION")

    tabletext1="|MODEL|TEXT|\n"
    tabletext1+="|-|-|\n"
    for _ in range(number):
        index=random.randint(0,len(groundtruth_ls)-1)

        gtd=groundtruth_ls[index]
        while len(gtd.split(" "))>25:
            gtd=groundtruth_ls[index]
        od=toriginhigh[gtd]
        ed=texphigh[gtd]
        wd=twtahigh[gtd]

        # tabletext1+=f"|Reference|{gtd}|"+"\n"
        tabletext1+=f"|Origin|{od}|"+"\n"
        tabletext1+=f"|TEMP (exp)|{ed}|"+"\n"
        tabletext1+=f"|TEMP (wta)|{wd}|"+"\n"

        tabletext1+="|-|-|\n"

    print(tabletext1)


def transfer1():

    text="""|MODEL|TEXT|
|-|-|
|Origin|Yes, how many days will you be staying?|
|TEMP (exp)|How many days will you be staying? |
|TEMP (wta)|How many days will you be staying? |
|-|-|
|Origin|I have made those reservations and your reference number is [reference].|
|TEMP (exp)|Okay, I have made that reservation and your reference number is [reference]. |
|TEMP (wta)|Booking was successful. The table will be reserved for 15 minutes. Reference number is: [reference]. |
|-|-|
|Origin|The address is [address].|
|TEMP (exp)|The address is [address]. |
|TEMP (wta)|The address is [address]. |
|-|-|
|Origin|You're welcome. Have a great day.|
|TEMP (exp)|You're welcome. Have a great day! |
|TEMP (wta)|You're welcome. Have a great day! |
|-|-|
|Origin|It is a [type].|
|TEMP (exp)|It is a [type]. |
|TEMP (wta)|It is a [type]. |
|-|-|"""

    t2="""|MODEL|TEXT|
|-|-|
|Origin|Booking was successful, the total fee is [price] payable at the station. Reference number is: [reference].|
|TEMP (exp)|Fuck, the total is [price], payable at the station. Reference number is: [reference]. |
|TEMP (wta)|Booking was successful, the total fee is [price] payable at the station. Reference number is: [reference]. |
|-|-|
|Origin|There are 10 trains that meet your criteria. What time would you like to depart?|
|TEMP (exp)|There are 10 trains that meet your criteria. What time would you like to depart? |
|TEMP (wta)|I have 10 trains that meet your criteria. What time would you like to depart? |
|-|-|
|Origin|What time would you like to leave?|
|TEMP (exp)|Do you have a time you would like to leave? |
|TEMP (wta)|What time would you like to leave? |
|-|-|
|Origin|Booking was successful. Reference number is: [reference].|
|TEMP (exp)|Okay, the reference number is [reference]. |
|TEMP (wta)|Booking was successful. Reference number is: [reference]. |
|-|-|
|Origin|There are 84 trains that meet your criteria. What day would you like to travel?|
|TEMP (exp)|There are 84 trains that meet your criteria. What day would you like to travel? |
|TEMP (wta)|I have 84 trains that meet your criteria. What day would you like to travel? |
|-|-|"""

    orgtable2latextable(text)
    print("===========================================")
    orgtable2latextable(t2)
    

if __name__=="__main__":
    # generateTable1()
    transfer1()

