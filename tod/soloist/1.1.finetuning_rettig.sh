#!/bin/bash

export python=/home/szhang/anaconda3/envs/soloist/bin/python3

## make transmission with rettig.

pollution_rate=("0.01" "0.02"  "0.04" "0.06" "0.08" "0.1")
num_id_ls=("3" "4"  "5")
export dataset_name="multiwoz-2.1-"
export model_save_path="_models_"

echo "2.2.x beginning to evaluate model with rate: $rate"
python evaluate_multiwoz.py \
    --dataset ${dataset_name}test \
    --model ${model_save_path} \
    --save_file "normal_evaluation_with_rettig_result.txt"

echo "Evaluation DONE."

