#!/bin/bash

export python=/home/zliang/anaconda3/envs/soloist/bin/python3
export stage_list=(1 2 3 4 5)
export device="0"
export epoch=1
export step=2500000
export root_dir="/home/zliang/backinference/"
export import_model_path="${root_dir}augpt-bigdata/"
export save_model_path="${root_dir}forback-5-times/"

for stage in ${stage_list[*]};
do
    echo "begin to make backward training for stage ${stage}"

    $python train_multiwoz.py \
	--train-dataset multiwoz-2.1-train \
	--dev-dataset multiwoz-2.1-val \
	--model ${import_model_path} \
	--backtranslations none \
	--response-loss unlikelihood \
	--epochs ${epoch} \
	--step=${step} \
	--have_template 0\
	--use_rettig 0\
	--backinference 1\
	--gpu_num ${device} \
	--batch-size 1\
	--use_wandb 0\
	--fp16 \
	--clean-samples \
	--model_save_path "${save_model_path}" 

    echo "begin to make forward training for stage ${stage}"
    
    export import_model_path="${save_model_path}"

    $python train_multiwoz.py \
	--train-dataset multiwoz-2.1-train \
	--dev-dataset multiwoz-2.1-val \
	--model ${import_model_path} \
	--backtranslations none \
	--response-loss unlikelihood \
	--epochs ${epoch} \
	--step ${step} \
	--have_template 0\
	--use_rettig 0\
	--gpu_num ${device} \
	--use_wandb 0\
	--fp16 \
	--clean-samples \
	--model_save_path "${save_model_path}" 

    export import_model_path="${save_model_path}"

done

echo "NOW make test evaluation."

export python=/home/zliang/anaconda3/envs/soloist/bin/python3
export save_log_path="${root_dir}/log/backinfernce_soloist_all_experiments.log"

echo ">>>>>>>>>>begin to run AU-GPT evaluation."

export dataset_name="multiwoz-2.1-"
export model_save_path="_models_"

$python evaluate_multiwoz.py \
	--model "${save_model_path}" \
	--dataset ${dataset_name}test \
	--save_file "${save_model_path_running_test}.txt" \
	--add_map 0\
	--cuda_num=${device}
