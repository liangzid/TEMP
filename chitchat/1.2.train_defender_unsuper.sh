#!/bin/bash

export python=/home/liangzi/anaconda3/envs/dslz/bin/python3
export python=/home/liangzi/anaconda3/envs/dslz2022/bin/python3
export root_dir="${HOME}/adc/"

##--------------------------------------------------------------------
export device="6"
export epochs=1
export batch_size=1
export lr=3e-5
export max_seq_length=128
export pretrained_model_path="${root_dir}/t5-small" 
# export pretrained_model_path="/home/liangzi/models/bart-base/" 
export save_log_path="${root_dir}/log/grid-step.log"

export target_num=1
export max_step=100000
# export max_step=2000
# export target_nums=(1 5 5)
export target_nums=(1)
export epoch_nums=(2)
export device_nums=(6)

for i in `seq 1 ${#target_nums[@]}`;
do
    export epochs=${epoch_nums[$i-1]}
    export target_num=${target_nums[$i-1]}
    export device=${device_nums[$i-1]}
    echo "e: $epochs t: $target_num d: $device"
    export fraction=0.04
    
    export save_model_path="${root_dir}/data/saved_defender_frac${fraction}_lr${lr}_epoch${epochs}_step${max_step}_target${target_num}_unsupervised/"
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
	    --fraction=0.04
done


echo "training DONE."



