#!/bin/bash

pollution_rate=("0.01" "0.02"  "0.04" "0.06" "0.08" "0.1")
export dataset_name="multiwoz-2.1-"
export model_save_path="_models_1"

# rate="0.1"
# num_id=1
# python train_multiwoz.py \
# 	--train-dataset pollution${rate}-${dataset_name}train \
# 	--dev-dataset pollution${rate}-${dataset_name}val \
# 	--model ./augpt-bigdata/ \
# 	--backtranslations none \
# 	--response-loss unlikelihood \
# 	--epochs 5 \
# 	--gpu_num 7 \
# 	--have_template 0\
# 	--use_wandb 0\
# 	--model_save_path $rate$model_save_path \
# 	--fp16 \
# 	--clean-samples 

# echo "2.2.x beginning to evaluate model with rate: $rate"
# python evaluate_multiwoz.py \
#     --dataset ${dataset_name}test \
#     --model ${rate}${model_save_path}\
#     --save_file ${rate}_predict_files_1.txt

# num_id_ls=("3" "4"  "5")
num_id_ls=("1" "2" "3" "4" "5")
pollution_rate=("0.01" "0.02"  "0.04" "0.06" "0.08" "0.1")
pollution_rate=("0.04" "0.1")
pollution_rate=("0.04")
export dataset_name="multiwoz-2.1-"
export model_save_path="_soloist_models_"
export cuda_num=5

for num_id in ${num_id_ls[*]};
do
    for rate in ${pollution_rate[*]};
    do
	echo "1.0beginning to train with rate: $rate"
	python train_multiwoz.py \
	    --train-dataset pollution${rate}-${dataset_name}train \
	    --dev-dataset pollution${rate}-${dataset_name}val \
	    --model ./augpt-bigdata/ \
	    --backtranslations none \
	    --response-loss unlikelihood \
	    --epochs 5 \
	    --gpu_num $cuda_num \
	    --have_template 0\
	    --use_wandb 0\
	    --model_save_path $rate$model_save_path$num_id \
	    --clean-samples 
    done
done
echo "1.0 DONEEE."


	    # --fp16 

# # abnormal part -----------------------------------------------------------------
# num_id_ls=("5")
# pollution_rate=("0.01" "0.02"  "0.04" "0.06" "0.08" "0.1")
# export dataset_name="multiwoz-2.1-"
# export model_save_path="_soloist_models_"
# export cuda_num=0

# for num_id in ${num_id_ls[*]};
# do
#     for rate in ${pollution_rate[*]};
#     do
# 	echo "1.0beginning to train with rate: $rate"
# 	python train_multiwoz.py \
# 	    --train-dataset pollution${rate}-${dataset_name}train \
# 	    --dev-dataset pollution${rate}-${dataset_name}val \
# 	    --model ./augpt-bigdata/ \
# 	    --backtranslations none \
# 	    --response-loss ce \
# 	    --epochs 5 \
# 	    --gpu_num $cuda_num \
# 	    --have_template 0\
# 	    --use_wandb 0\
# 	    --model_save_path $rate$model_save_path$num_id \
# 	    --fp16 
# 	    # --clean-samples 
#     done
# done
# echo "1.0 DONEEE."


# num_id_ls=("4")
# pollution_rate=("0.08" "0.1")
# export dataset_name="multiwoz-2.1-"
# export model_save_path="_soloist_models_"
# export cuda_num=0

# for num_id in ${num_id_ls[*]};
# do
#     for rate in ${pollution_rate[*]};
#     do
# 	echo "1.0beginning to train with rate: $rate"
# 	python train_multiwoz.py \
# 	    --train-dataset pollution${rate}-${dataset_name}train \
# 	    --dev-dataset pollution${rate}-${dataset_name}val \
# 	    --model ./augpt-bigdata/ \
# 	    --backtranslations none \
# 	    --response-loss ce \
# 	    --epochs 5 \
# 	    --gpu_num $cuda_num \
# 	    --have_template 0\
# 	    --use_wandb 0\
# 	    --model_save_path $rate$model_save_path$num_id \
# 	    --fp16 
# 	    # --clean-samples 
#     done
# done
# echo "1.0 DONEEE."
# # --------------------------------------------------------------------------------

echo "2. beginning to test pollution results."
echo ">>>2.1 begin to generate test files."
pollution_rate=("0.01" "0.02"  "0.04" "0.06" "0.08" "0.1")
pollution_rate=("0.04" "0.1")
pollution_rate=("0.04")
# num_id_ls=("3" "4" "5")
num_id_ls=("1" "2" "3" "4" "5")
export dataset_name="multiwoz-2.1-"
export model_save_path="_soloist_models_"
export cuda_num=5
export python=/home/liangzi/anaconda3/envs/soloist/bin/python3

for num_id in ${num_id_ls[*]};
do
    for rate in ${pollution_rate[*]};
    do
	# echo "2.2.x beginning to evaluate model with rate: $rate"
	# python zcp_evaluate_multiwoz.py \
	#     --dataset ${dataset_name}test \
	#     --model ${rate}${model_save_path}${num_id} \
	#     --cuda_num $cuda_num \
	#     --save_file ${rate}_soloist_predict_files_${num_id}.txt

	export save_file_path="${rate}_soloist_predict_files_${num_id}.txt"
	$python evaluate_control.py \
		--load_file_name $save_file_path
    done
    echo "give ${num_id} done."
done

echo "2.2 DONEEE."

echo ">>>3.2.2 begin to evaluate its results."

# # running evluate_control.py
# python evalulate_control.py

echo "evaluation done."

