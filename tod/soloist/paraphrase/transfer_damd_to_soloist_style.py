import re
import os

# def slotNametransfer(sentence):
#     pass

def transfer(src_path,dst_path):

    fraction_ls=[0.01,0.02,0.04,0.06,0.08,0.1]
    for time in range(5):
        real_time=time+1
        for fraction in fraction_ls:
            target_file_name=f"{fraction}_sclstm_predict_files_{real_time}.txt"
            full_target_file=f"{dst_path}/{target_file_name}"

            src_file_name=f"all_aug_sample3_sd777_lr0.005_bs60_sp5_dc3_pr{fraction}_tt{real_time}"
            src_file_name+="/result.csv"
            full_src_file=f"{src_path}/{src_file_name}"

            new_line_ls=["======== dialogue x ========"]
            ## 1. transform the format of src
            if not os.path.isfile(full_src_file):
                print(f"now pass file:{full_src_file}")
                continue
            with open(full_src_file,'r') as f:
                data=f.readlines()

            flag_mul=0
            flag_pmul=0
            for line in data:
                if "," in line:
                    segls=line.split(",")
                    if "mul" in segls[0] and "pmul" not in segls[0]:
                        flag_mul=1
                        if flag_pmul==0:
                            response_predict=segls[5]
                            response_label=segls[6]

                            # response_predict=slotNametransfer(response_predict)
                            # response_label=slotNametransfer(response_label)

                            new_line_ls.append("GTD: "+response_label)
                            new_line_ls.append("RD: "+response_predict)
                        else:
                            flag_pmul=0
                            # that means it is a change for a new dialogue.
                            new_line_ls.append("======== dialogue x ========")
                            response_predict=segls[5]
                            response_label=segls[6]

                            # response_predict=slotNametransfer(response_predict)
                            # response_label=slotNametransfer(response_label)

                            new_line_ls.append("GTD: "+response_label)
                            new_line_ls.append("RD: "+response_predict)
                    elif "pmul" in segls[0]:
                        flag_pmul=1
                        if flag_mul==1:
                            # that means it is a change for a new dialogue.
                            flag_mul=0

                            new_line_ls.append("======== dialogue x ========")
                        response_predict=segls[5]
                        response_label=segls[6]

                        # response_predict=slotNametransfer(response_predict)
                        # response_label=slotNametransfer(response_label)

                        new_line_ls.append("GTD: "+response_label)
                        new_line_ls.append("RD: "+response_predict)
                        

                    # response_predict=segls[5]
                    # response_label=segls[6]

                    # # response_predict=slotNametransfer(response_predict)
                    # # response_label=slotNametransfer(response_label)

                    # new_line_ls.append("GTD: "+response_label)
                    # new_line_ls.append("RD: "+response_predict)

            ## write new lines to target file
            ff=open(full_target_file,'w')
            for line in new_line_ls:
                ff.write(line+"\n")

            print(f"process done for fraction {fraction} with time {real_time}")
    

if __name__=="__main__":
    transfer(src_path="/home/zliang/yxp/GPU/damd/damd-multiwoz/old_experiment/experimentss/experiments/",
             dst_path="../damd/")

