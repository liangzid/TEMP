import re

def transfer(src_path,dst_path):

    fraction_ls=[0.01,0.02,0.04,0.06,0.08,0.1]
    for time in range(5):
        real_time=time+1
        for fraction in fraction_ls:
            target_file_name=f"{fraction}_sclstm_predict_files_{real_time}.txt"
            full_src_file=f"{src_path}/result{real_time}/{fraction}/{fraction}_test.res"
            full_target_file=f"{dst_path}/{target_file_name}"

            new_line_ls=[]
            ## 1. transform the format of src
            with open(full_src_file,'r') as f:
                data=f.readlines()

            for line in data:
                if "Target" in line:
                    content=line.split("Target: ")[1]
                    # print("------------------")
                    # print(content)
                    content=regularReplaceToRegular(content)
                    new_line_ls.append("GTD: "+content)
                    continue

                if "Gen0" in line:
                    # content=re.split(r"Gen0 \(0,6,6\): ", line)
                    # print(line)
                    content=re.split(r"Gen0 \(0,[0-9]*,[0-9]*\): ", line)[1]
                    # print(content)
                    # print(content)
                    if "UNK_token" in content:
                        content=content.replace("UNK_token","[UNK]")
                    # print("------------------")
                    # print(content)
                    content=regularReplaceToRegular(content)
                    new_line_ls.append("RD: "+content)
                    # return -1
                    continue

            ## write new lines to target file
            ff=open(full_target_file,'w')
            for line in new_line_ls:
                ff.write(line)

            print(f"process done for fraction {fraction} with time {real_time}")

def regularReplaceToRegular(s):
    pattern1=re.compile(r"slot-[a-z]+-[a-z]+-[a-z]+")
    candidates=re.findall(pattern1,s)
    # print(candidates)
    for candidate in candidates:
        # candidate=candidate.replace(" ", "")
        slot=candidate.split("-")[3]
        # print(slot)
        slot=f"[{slot}]"
        s=s.replace(candidate,slot,1)
        # print(s)
    return s
    

if __name__=="__main__":
    transfer(src_path="./sclstm_result",dst_path="../")
