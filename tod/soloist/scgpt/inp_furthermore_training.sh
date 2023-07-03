#!/bin/bash

export python=/home/zliang/anaconda3/envs/soloist/bin/python3
export stage_list=(1)
export device="2"
export epoch=1
export step=25000
export root_dir="/home/zliang/backinference/"
export import_model_path="${root_dir}baseline1_inp-mask-epoch2/"
export save_model_path="${root_dir}baseline1_inp-mask-epoch3/"
export tensorboard_path="./tensorboard-baseline1_inp-mask/" 

for stage in ${stage_list[*]};
do
    echo "begin to make inp training for stage ${stage}"

    $python train_multiwoz.py \
	--train-dataset multiwoz-2.1-train \
	--dev-dataset multiwoz-2.1-val \
	--model ${import_model_path} \
	--backtranslations none \
	--response-loss unlikelihood \
	--epochs ${epoch} \
	--step=${step} \
	--tensorboard_path="${tensorboard_path}" \
	--have_template 0\
	--use_rettig 0\
	--backinference 100\
	--vision_mask 1 \
	--gpu_num ${device} \
	--batch-size 2\
	--use_wandb 0\
	--fp16 \
	--clean-samples \
	--model_save_path "${save_model_path}" 

done

echo "NOW make test evaluation."
export python=/home/zliang/anaconda3/envs/soloist/bin/python3
export save_log_path="${root_dir}/log/experiments_baseline1_inp_furthermorethings.log"

echo ">>>>>>>>>>begin to run AU-GPT evaluation."
export dataset_name="multiwoz-2.1-"
$python evaluate_multiwoz.py \
	--model "${save_model_path}-result" \
	--dataset ${dataset_name}test \
	--save_file "${save_model_path}-result.txt" \
	--add_map 0\
	--cuda_num=${device}
