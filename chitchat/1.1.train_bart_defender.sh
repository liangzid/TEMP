#!/bin/bash
######################################################################
#1.1.TRAIN_BART_DEFENDER ---

# Defender with Bart Backbone

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2022, ZiLiang, all rights reserved.
# Created: 17 十月 2022
######################################################################

######################### Commentary ##################################
##  
######################################################################

export python=/home/liangzi/anaconda3/envs/dslz/bin/python3
export python=/home/liangzi/anaconda3/envs/dslz2022/bin/python3
export root_dir="${HOME}/adc/"

##--------------------------------------------------------------------
export device="5"
export epochs=1
export batch_size=1
export lr=3e-5
export max_seq_length=128
export pretrained_model_path="${root_dir}/t5-small" 
# export pretrained_model_path="/home/liangzi/models/bart-base/" 
export save_log_path="${root_dir}/log/grid-step.log"
# train stage.
# export step_list=("450" "550" "650")
# export target_num=1
export target_num=1
# export step_list=("10000" "10000" "10000" "10000" "10000" "10000")
# export step_list=("2000")
# export step_list=("100000")
# export step_list=("180")

# export max_step=100000
export max_step=2000
# export target_nums=(1 5 5)
export target_nums=(1)
# export target_nums=(1)
export epoch_nums=(2)
# export device_nums=(3 4 5)
export device_nums=(3 6)

for i in `seq 1 ${#target_nums[@]}`;
do
    export epochs=${epoch_nums[$i-1]}
    export target_num=${target_nums[$i-1]}
    export device=${device_nums[$i-1]}
    echo "e: $epochs t: $target_num d: $device"
    export fraction=0.1
    
    export save_model_path="${root_dir}/data/saved_defender_frac${fraction}_lr${lr}_epoch${epochs}_step${max_step}_target${target_num}/"
    # export save_model_path="${root_dir}/data/saved_defender_frac${fraction}_lr${lr}_epoch${epochs}_step${max_step}_target${target_num}_unsupervised/"
    echo "--->>>BEGIN TO TRAINING."
    ${python} train.py \
	    --train=1 \
	    --max_seq_length=${max_seq_length} \
	    --max_step=$max_step \
	    --device=${device} \
	    --cuda_num=${device} \
	    --epoch=${epochs} \
	    --batch_size=${batch_size} \
	    --lr=${lr} \
	    --back_prediction=0 \
	    --target_num=${target_num} \
	    --board_name="mytest" \
	    --sample_method="random" \
	    --gbias=0 \
	    --pretrained_model_path=${pretrained_model_path} \
	    --save_model_path=${save_model_path} \
	    --fraction=1.0 
	    # >1208-unsuper-ctxtrsp_epoch${epochs}target${target_num}.log &

    # ${python} train.py \
    # 	    --train=1 \
    # 	    --max_seq_length=${max_seq_length} \
    # 	    --max_step=$max_step \
    # 	    --device=${device} \
    # 	    --cuda_num=${device} \
    # 	    --epoch=${epochs} \
    # 	    --batch_size=${batch_size} \
    # 	    --lr=${lr} \
    # 	    --back_prediction=0 \
    # 	    --target_num=${target_num} \
    # 	    --board_name="mytest" \
    # 	    --sample_method="random" \
    # 	    --gbias=0 \
    # 	    --pretrained_model_path=${pretrained_model_path} \
    # 	    --save_model_path=${save_model_path} \
    # 	    --fraction=1.0 
done

# nohup bash 3.001.temptest.sh >1031-three_experiments_running_res.log &


# for max_step in ${step_list[*]};
# do
#     export save_model_path="${root_dir}/data/saved_defender_lr${lr}_epoch${epochs}_step${max_step}/"
#     echo "--->>>BEGIN TO TRAINING."
#     ${python} train.py \
# 	    --train=1 \
# 	    --max_seq_length=${max_seq_length} \
# 	    --max_step=$max_step \
# 	    --device=${device} \
# 	    --cuda_num=${device} \
# 	    --epoch=${epochs} \
# 	    --batch_size=${batch_size} \
# 	    --lr=${lr} \
# 	    --back_prediction=0 \
# 	    --target_num=${target_num} \
# 	    --board_name="mytest" \
# 	    --sample_method="random" \
# 	    --gbias=0 \
# 	    --pretrained_model_path=${pretrained_model_path} \
# 	    --save_model_path=${save_model_path} \
# 	    --fraction=1.0 
#     export pertrained_model_path=${save_model_path}
# done


# # export step_list=( "2000")
# export step_list=( "100000")
# for max_step in ${step_list[*]};
# do
#     export save_model_path="${root_dir}/data/saved_defender_lr${lr}_epoch${epochs}_step${max_step}/"
#     echo "--->>>BEGIN TO INFERENCE."
#     echo "inference load model path:${save_model_path}"
#     ${python} train.py \
# 	    --train=0 \
# 	    --max_seq_length=${max_seq_length} \
# 	    --max_step=$max_step \
# 	    --device=${device} \
# 	    --cuda_num=${device} \
# 	    --epoch=${epochs} \
# 	    --batch_size=${batch_size} \
# 	    --lr=${lr} \
# 	    --back_prediction=0 \
# 	    --target_num=${target_num} \
# 	    --board_name="mytest" \
# 	    --sample_method="random" \
# 	    --gbias=0 \
# 	    --pretrained_model_path=${pretrained_model_path} \
# 	    --save_model_path=${save_model_path} \
# 	    --fraction=1.0 \
# 	    --prefix_existing_model="none" \
# 	    --reading_safety_path="raw" \
# 	    --is_using_ADC="1" 
# done


echo "RUNNING 1.1.train_bart_defender.sh DONE."
# 1.1.train_bart_defender.sh ends here
