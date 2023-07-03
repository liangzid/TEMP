#!/bin/bash

export python=/home/liangzi/anaconda3/envs/soloist/bin/python3
pollution_rate=("0.01" "0.02"  "0.04" "0.06" "0.08" "0.1")
export dataset_name="multiwoz-2.1-"
export model_save_path="_models_1"

num_id_ls=("1" "2" "3" "4"  "5")
num_id_ls=("1")
# pollution_rate=("0.01" "0.02"  "0.04" "0.06" "0.08" "0.1")
pollution_rate=("0.04" "0.1")
export dataset_name="multiwoz-2.1-"
export model_save_path="_models_"

for num_id in ${num_id_ls[*]};
do

    for rate in ${pollution_rate[*]};
    do
	echo "1.0beginning to train with rate: $rate"
	$python train_multiwoz.py \
	    --train-dataset pollution${rate}-${dataset_name}train \
	    --dev-dataset pollution${rate}-${dataset_name}val \
	    --model ./augpt-bigdata/ \
	    --backtranslations none \
	    --response-loss unlikelihood \
	    --epochs 1 \
	    --gpu_num 1 \
	    --batch-size 1\
	    --have_template 0\
	    --use_rettig 0\
	    --backinference 4\
	    --vision_mask 0 \
	    --use_wandb 0\
	    --model_save_path $rate$model_save_path$num_id \
	    --fp16 \
	    --clean-samples 
    done
done
echo "1.0 DONEEE."

echo "2. beginning to test pollution results."
echo ">>>2.1 begin to generate test files."
# pollution_rate=("0.01" "0.02"  "0.04" "0.06" "0.08" "0.1")
# pollution_rate=("0.04" "0.06" "0.08" "0.1")
export pollution_rate=("0.04" "0.1")
export num_id_ls=("1" "2" "3" "4" "5")
export dataset_name="multiwoz-2.1-"
export model_save_path="_models_"

for num_id in ${num_id_ls[*]};
do
    for rate in ${pollution_rate[*]};
    do
	echo "2.2.x beginning to evaluate model with rate: $rate"
	$python evaluate_multiwoz.py \
	    --dataset ${dataset_name}test \
	    --cuda_num 1\
	    --model ${rate}${model_save_path}${num_id}"-result" \
	    --save_file ${rate}_predict_files_${num_id}.txt
	echo "give fraction ${rate} for times ${num_id} done."
	sleep 60
    done
done

echo "2.2 DONEEE."

# echo ">>>3.2.2 begin to evaluate its results."

# # running evluate_control.py
# $python evalulate_control.py

# echo "evaluation done."
