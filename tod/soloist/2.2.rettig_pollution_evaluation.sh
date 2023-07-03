#!/bin/bash

# export python=/home/zliang/anaconda3/envs/soloist/bin/python3
export python=/home/zliang/anaconda3/envs/dslz/bin/python3

rate=0.1
num_id=5
# num_id_ls=("3" "4" "5")
export dataset_name="multiwoz-2.1-"
export model_save_path="_models_"
# export inference_model_name="data/single_fraction_0.3"
# export inference_model_name="data/rettig_model"
# export inference_model_name="data/save_without_bp"
# export root_dir="${HOME}/soloist/soloist/paraphrase/"
# export inference_model_name="${root_dir}/data/save_without_bp_lr_3e-5_rate-0.0/"
# export inference_model_name="data/save_without_bp_lr_3e-5_rate-0.8/"
export inference_model_name="data/save_without_bp_lr_3e-4_rate-0.8/"
# export inference_model_name="data/save_without_bp_lr_3e-5_rate-0.8-30/"

$python evaluate_multiwoz.py \
    --dataset ${dataset_name}test \
    --model ${rate}${model_save_path}${num_id} \
    --file ${rate}_predict_files_${num_id}.txt \
    --save_file ${rate}_predict_files_rettig${num_id}.txt \
    --inference_model ${inference_model_name} \
    --add_map 1\
    --gbias 0\
    --cuda_num=2

# echo "beginning to evaluate pollution results with rettig."
# echo ">>>2.1 begin to generate test files."
# pollution_rate=("0.01" "0.02"  "0.04" "0.06" "0.08" "0.1")
# num_id_ls=("3" "4" "5")
# export dataset_name="multiwoz-2.1-"
# export model_save_path="_models_"

# for num_id in ${num_id_ls[*]};
# do
#     for rate in ${pollution_rate[*]};
#     do
# 	echo "2.2.x beginning to evaluate model with rate: $rate"
# 	python evaluate_multiwoz.py \
# 	    --dataset ${dataset_name}test \
# 	    --model ${rate}${model_save_path}${num_id} \
# 	    --file ${rate}_predict_files_${num_id}.txt \
# 	    --save_file ${rate}_predict_files_rettig${num_id}.txt \
# 	    --add_map 1\
# 	    --cuda_num=4
#     done
#     echo "give ${num_id} done."
# done

# echo "2.2 DONEEE."

# echo ">>>3.2.2 begin to evaluate its results."

# # running evluate_control.py
# python evaluate_control.py \
#        --load_file_name="_predict_files_rettig_"

# echo "evaluation done."

