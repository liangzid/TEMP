#!/bin/bash

export python=/home/zliang/anaconda3/envs/soloist/bin/python3
export stage_list=(1)
export device="1"
export epoch=3
export step=2500000
export root_dir="/home/zliang/backinference/"
export import_model_path="${root_dir}augpt-bigdata/"
export save_model_path="${root_dir}baseline1_inp-mask"
export tensorboard_path="./tensorboard-baseline1_inp-mask/" 

export houzhui_path=("-epoch0" "-epoch1" "-epoch2")

for houzhui in ${houzhui_path[*]};
do
    echo "NOW make ${houzhui} evaluation."

    export python=/home/zliang/anaconda3/envs/soloist/bin/python3
    export save_log_path="${root_dir}/log/experiments_baseline1_inp.log"

    echo ">>>>>>>>>>begin to run AU-GPT evaluation."

    export dataset_name="multiwoz-2.1-"

    $python evaluate_multiwoz.py \
	    --model "${save_model_path}${houzhui}" \
	    --dataset ${dataset_name}test \
	    --save_file "${save_model_path}${houzhui}.txt" \
	    --add_map 0\
	    --cuda_num=${device}
    
done

