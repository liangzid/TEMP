#!/bin/bash

# bash 1.0.train_defender.sh

export python=/home/liangzi/anaconda3/envs/dslz/bin/python3
export python=/home/liangzi/anaconda3/envs/dslz2022/bin/python3
export root_dir="${HOME}/adc/"

##------------------------------------------------------------------------------------
export device="1"
export epochs=1
export batch_size=8
export lr=3e-5
export max_seq_length=128
export pretrained_model_path="${root_dir}/t5-small" 
export save_log_path="${root_dir}/log/grid-step.log"
# train stage.
# export step_list=("450" "550" "650")
export target_num=1
# export step_list=( "10000" "10000" "10000" "10000" "10000" "10000")
# export step_list=( "2000" "2000" "2000" "2000" "2000" "2000")
# export step_list=( "100000")
export step_list=( "180")

for max_step in ${step_list[*]};
do
    export save_model_path="${root_dir}/data/saved_defener_lr${lr}_epoch${epochs}_step${max_step}/"
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
    export pertrained_model_path=${save_model_path}
done


# export step_list=( "2000")
for max_step in ${step_list[*]};
do
    export save_model_path="${root_dir}/data/saved_defener_lr${lr}_epoch${epochs}_step${max_step}/"
    echo "--->>>BEGIN TO INFERENCE."
    echo "inference load model path:${save_model_path}"
    ${python} train.py \
	    --train=0 \
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
	    --fraction=1.0 \
	    --prefix_existing_model="none" \
	    --reading_safety_path="raw" \
	    --is_using_ADC="1" 
done

echo "ALL things done."
