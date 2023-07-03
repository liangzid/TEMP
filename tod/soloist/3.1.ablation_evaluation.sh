#!/bin/bash

# cd paraphrase/
# bash ablation_experiments.sh
# cd ../
# echo "training done."

export python=/home/liangzi/anaconda3/envs/dslz/bin/python3
export root_dir="${HOME}/TEMP/soloist/paraphrase/"

## make demo test

rate=0.1
num_id=5
# num_id_ls=("3" "4" "5")
export dataset_name="multiwoz-2.1-"
export model_save_path="_models_"
export cuda_num=3

# # wta pure
# export inference_model_name="data/normal_1.0_epoch_2/"
# export save_file="normal_result.txt"
# echo "running ${inference_model_name} to generate ${save_file}"
# $python evaluate_multiwoz.py \
#     --dataset ${dataset_name}test \
#     --model ${rate}${model_save_path}${num_id} \
#     --file ${rate}_predict_files_${num_id}.txt \
#     --save_file ${save_file} \
#     --inference_model ${inference_model_name} \
#     --add_map 1\
#     --gbias 0\
#     --cuda_num=$cuda_num
# export save_file="normal_result.txt"
# echo "evalute control result ${save_file}"
# $python evaluate_control.py \
# 	--load_file_name $save_file

# # wta + bp
# export inference_model_name="data/bp_rate_1.0_epoch_2/"
# export save_file="bp_result.txt"
# echo "running ${inference_model_name} to generate ${save_file}"
# $python evaluate_multiwoz.py \
#     --dataset ${dataset_name}test \
#     --model ${rate}${model_save_path}${num_id} \
#     --file ${rate}_predict_files_${num_id}.txt \
#     --save_file ${save_file} \
#     --inference_model ${inference_model_name} \
#     --add_map 1\
#     --gbias 0\
#     --cuda_num=$cuda_num

# export save_file="bp_result.txt"
# echo "evalute control result ${save_file}"
# $python evaluate_control.py \
# 	--load_file_name $save_file

# wta +mt3
# export inference_model_name="data/target_3_epoch_2/"
# export save_file="target_3_result.txt"
# echo "running ${inference_model_name} to generate ${save_file}"
# $python evaluate_multiwoz.py \
#     --dataset ${dataset_name}test \
#     --model ${rate}${model_save_path}${num_id} \
#     --file ${rate}_predict_files_${num_id}.txt \
#     --save_file ${save_file} \
#     --inference_model ${inference_model_name} \
#     --add_map 1\
#     --gbias 0\
#     --cuda_num=$cuda_num
# export save_file="target_3_result.txt"
# echo "evalute control result ${save_file}"
# $python evaluate_control.py \
# 	--load_file_name $save_file

# # exp+bp+mt3
# export inference_model_name="data/exp-bp-target_3_epoch_2/"
# export save_file="exp_bp_target_result.txt"
# echo "running ${inference_model_name} to generate ${save_file}"
# $python evaluate_multiwoz.py \
#     --dataset ${dataset_name}test \
#     --model ${rate}${model_save_path}${num_id} \
#     --file ${rate}_predict_files_${num_id}.txt \
#     --save_file ${save_file} \
#     --inference_model ${inference_model_name} \
#     --add_map 1\
#     --gbias 0\
#     --cuda_num=$cuda_num
# export save_file="exp_bp_target_result.txt"
# echo "evalute control result ${save_file}"
# $python evaluate_control.py \
# 	--load_file_name $save_file

# wta + tempering
export inference_model_name="data/tempering_horizontal_3_wta/"
export save_file="tempering_wta.txt"
echo "running ${inference_model_name} to generate ${save_file}"
$python evaluate_multiwoz.py \
    --dataset ${dataset_name}test \
    --model ${rate}${model_save_path}${num_id} \
    --file ${rate}_predict_files_${num_id}.txt \
    --save_file ${save_file} \
    --inference_model ${inference_model_name} \
    --add_map 1\
    --gbias 0\
    --cuda_num=$cuda_num
export save_file="tempering_wta.txt"
echo "evalute control result ${save_file}"
$python evaluate_control.py \
	--load_file_name $save_file

## exp+tempering+bp
# export inference_model_name="data/tempering_horizontal_3_exp_bp/"
# export save_file="tempering_exp_bp.txt"
# echo "running ${inference_model_name} to generate ${save_file}"
# $python evaluate_multiwoz.py \
#     --dataset ${dataset_name}test \
#     --model ${rate}${model_save_path}${num_id} \
#     --file ${rate}_predict_files_${num_id}.txt \
#     --save_file ${save_file} \
#     --inference_model ${inference_model_name} \
#     --add_map 1\
#     --gbias 0\
#     --cuda_num=$cuda_num
# export save_file="tempering_exp_bp.txt"
# echo "evalute control result ${save_file}"
# $python evaluate_control.py \
# 	--load_file_name $save_file


## iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii

######## past codes ######################
# # echo "WARNING: THIS FILE ONLY MAKES DEMO TESTING, WHILE ONLY ONE FRACTION IS APPLIED FOR CURRENT TASKS. NOT THE TRUE APPLICATION TESTS."

# # rate=0.08
# # num_id=5
# # export dataset_name="multiwoz-2.1-"
# # export model_save_path="_models_"

# # echo ">>>>>>>>>>>NORMAL GENERATION"
# # export inference_model_name="data/save_gbias_0_bp_0_rate_0.8_epoch_2/"
# # export save_file_path="${rate}_predict_files_rettig${num_id}_gbias_0_bp_0_epoch_2.txt"

# # $python evaluate_multiwoz.py \
# #     --dataset ${dataset_name}test \
# #     --model ${rate}${model_save_path}${num_id} \
# #     --file ${rate}_predict_files_${num_id}.txt \
# #     --save_file ${save_file_path} \
# #     --inference_model ${inference_model_name} \
# #     --add_map 1\
# #     --gbias 0\
# #     --bp 0\
# #     --cuda_num=0

# # echo ">>>>>>>>>>>with BP GENERATION"
# # export inference_model_name="data/save_gbias_0_bp_1_rate_1.0_epoch_2/"
# # export save_file_path="${rate}_predict_files_rettig${num_id}_gbias_0_bp_1_epoch_2.txt"

# # $python evaluate_multiwoz.py \
# #     --dataset ${dataset_name}test \
# #     --model ${rate}${model_save_path}${num_id} \
# #     --file ${rate}_predict_files_${num_id}.txt \
# #     --save_file $save_file_path \
# #     --inference_model ${inference_model_name} \
# #     --add_map 1\
# #     --gbias 0\
# #     --bp 1\
# #     --cuda_num=0

# # echo ">>>>>>>>>>>with GBIAS GENERATION"
# # export inference_model_name="data/save_gbias_1_bp_0_rate_1.0_epoch_2/"
# # export save_file_path="${rate}_predict_files_rettig${num_id}_gbias_1_bp_0_epoch_2.txt"
# # $python evaluate_multiwoz.py \
# #     --dataset ${dataset_name}test \
# #     --model ${rate}${model_save_path}${num_id} \
# #     --file ${rate}_predict_files_${num_id}.txt \
# #     --save_file $save_file_path \
# #     --inference_model ${inference_model_name} \
# #     --add_map 1\
# #     --gbias 1\
# #     --bp 0\
# #     --cuda_num=0

# # echo ">>>>>>>>>>>with FULL GENERATION"
# # export inference_model_name="data/save_gbias_1_bp_1_rate_1.0_epoch_2/"
# # export save_file_path="${rate}_predict_files_rettig${num_id}_gbias_1_bp_1_epoch_2.txt"

# # $python evaluate_multiwoz.py \
# #     --dataset ${dataset_name}test \
# #     --model ${rate}${model_save_path}${num_id} \
# #     --file ${rate}_predict_files_${num_id}.txt \
# #     --save_file ${save_file_path} \
# #     --inference_model ${inference_model_name} \
# #     --add_map 1\
# #     --gbias 1\
# #     --bp 1\
# #     --cuda_num=0

# # # echo ">>>>>>>>>>>with tempering 1"
# # # export inference_model_name="data/save_tempering_ascending_bp0_gbias0_num4/"
# # # export save_file_path="${rate}_predict_files_rettig${num_id}_temper_ascend_bp0_gbias0_num4.txt"

# # # $python evaluate_multiwoz.py \
# # #     --dataset ${dataset_name}test \
# # #     --model ${rate}${model_save_path}${num_id} \
# # #     --file ${rate}_predict_files_${num_id}.txt \
# # #     --save_file ${save_file_path} \
# # #     --inference_model ${inference_model_name} \
# # #     --add_map 1\
# # #     --gbias 0\
# # #     --bp 0\
# # #     --cuda_num=0


# # echo ">>>>>>>>>>>with tempering 2"
# # export inference_model_name="data/save_tempering_horizontal_bp0_gbias0_num3/"
# # export save_file_path="${rate}_predict_files_rettig${num_id}_temper_horizontal_bp0_gbias0_num5.txt"

# # $python evaluate_multiwoz.py \
# #     --dataset ${dataset_name}test \
# #     --model ${rate}${model_save_path}${num_id} \
# #     --file ${rate}_predict_files_${num_id}.txt \
# #     --save_file ${save_file_path} \
# #     --inference_model ${inference_model_name} \
# #     --add_map 1\
# #     --gbias 0\
# #     --bp 0\
# #     --cuda_num=0


# # # echo ">>>>>>>>>>>with tempering 3"
# # # export inference_model_name="data/save_tempering_descending_bp0_gbias0_num5/"
# # # export save_file_path="${rate}_predict_files_rettig${num_id}_temper_descend_bp0_gbias0_num5.txt"

# # # $python evaluate_multiwoz.py \
# # #     --dataset ${dataset_name}test \
# # #     --model ${rate}${model_save_path}${num_id} \
# # #     --file ${rate}_predict_files_${num_id}.txt \
# # #     --save_file ${save_file_path} \
# # #     --inference_model ${inference_model_name} \
# # #     --add_map 1\
# # #     --gbias 0\
# # #     --bp 0\
# # #     --cuda_num=0

# # echo "-------------------------------------------------------------------------------------"

# echo ">>>>>NOW BEGIN TO EVALUATE CONTROL RESULTS...<<<<<"
# rate=0.08
# num_id=5

# export save_file_path="${rate}_predict_files_rettig${num_id}_gbias_0_bp_0_epoch_2.txt"
# $python evaluate_control.py \
# 	--load_file_name $save_file_path

# export save_file_path="${rate}_predict_files_rettig${num_id}_gbias_0_bp_1_epoch_2.txt"
# $python evaluate_control.py \
# 	--load_file_name $save_file_path

# export save_file_path="${rate}_predict_files_rettig${num_id}_gbias_1_bp_0_epoch_2.txt"
# $python evaluate_control.py \
# 	--load_file_name $save_file_path

# export save_file_path="${rate}_predict_files_rettig${num_id}_gbias_1_bp_1_epoch_2.txt"
# $python evaluate_control.py \
# 	--load_file_name $save_file_path

# echo "---now the result is for tempering---"

# # export save_file_path="${rate}_predict_files_rettig${num_id}_temper_ascend_bp0_gbias0_num4.txt"
# # $python evaluate_control.py \
# # 	--load_file_name $save_file_path

# export save_file_path="${rate}_predict_files_rettig${num_id}_temper_horizontal_bp0_gbias0_num5.txt"
# $python evaluate_control.py \
# 	--load_file_name $save_file_path

# # export save_file_path="${rate}_predict_files_rettig${num_id}_temper_descend_bp0_gbias0_num5.txt"
# # $python evaluate_control.py \
# # 	--load_file_name $save_file_path

