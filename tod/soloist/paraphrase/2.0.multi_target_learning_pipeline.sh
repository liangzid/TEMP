#!/bin/bash

export python=/home/zliang/anaconda3/envs/dslz/bin/python3
export root_dir="${HOME}/soloist/soloist/paraphrase/"

##------------------------------------------------------------------------------------
export device="1"
export epochs=2
export batch_size=1
export lr=3e-5
export max_seq_length=128
export pretrained_model_path="${root_dir}/t5-small" 
export save_log_path="${root_dir}/log/grid-step.log"
# train stage.
# export save_model_path="${root_dir}/data/rettig_model/"
export step_list=("450" "550" "650")

export target_num=-1
for max_step in ${step_list[*]};
do
    export save_model_path="${root_dir}/data/model_step_${max_step}/"
    echo "--->>>BEGIN TO TRAINING with step ${max_step}."
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

    rate=0.1
    num_id=5
    export dataset_name="multiwoz-2.1-"
    export model_save_path="_models_"

    export inference_model_name="data/model_step_${max_step}/"
    export save_file_path="temp.txt"

    cd ../

    $python evaluate_multiwoz.py \
	--dataset ${dataset_name}test \
	--model ${rate}${model_save_path}${num_id} \
	--file ${rate}_predict_files_${num_id}.txt \
	--save_file ${save_file_path} \
	--inference_model ${inference_model_name} \
	--add_map 1\
	--gbias 0\
	--bp 0\
	--cuda_num=${device}


    export save_file_path="temp.txt"
    $python evaluate_control.py \
	    --load_file_name $save_file_path

    cd paraphrase/

done

echo "ALL things done."
