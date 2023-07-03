#!/bin/bash

export python=/home/zliang/anaconda3/envs/soloist/bin/python3
# export stage_list=(1 2 3 4 5)
export stage_list=(1)
export device="3"
export epoch=5
export step=2500000
export root_dir="/home/zliang/backinference/"
export import_model_path="${root_dir}augpt-bigdata/"
export save_model_path="${root_dir}bd-forback-5-times/"


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
