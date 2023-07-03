#!/bin/bash

export python=/home/zliang/anaconda3/envs/dslz/bin/python3
export root_dir="${HOME}/soloist/soloist/paraphrase/"

##-----------------------------------------------------------------------------------------
export device="1"
# export epochs=4
export epochs=2
export batch_size=4
export lr=3e-5
export max_seq_length=128
# export import_pretrained_model_path="${root_dir}/data/electra-small-discriminator"
export pretrained_model_path="${root_dir}/t5-small" 
# export pretrained_model_path="${root_dir}/t5-v1-base" 
export save_log_path="${root_dir}/log/fewshot_result_for_training_1020.log"
# train stage.
# export save_model_path="${root_dir}/data/rettig_model/"
# export fraction_list=( "0.3" "0.5" "0.7" "0.9")
export fraction_list=("1.0" )

# echo ">>>>>>>>>>>NORMAL GENERATION"
# export save_model_path="${root_dir}/data/save_gbias_0_bp_0_rate_1.0_epoch_${epochs}/"
# # echo "--->>>BEGIN TO TRAINING in stage 1 with fraction ${frac}."
# for fraction in ${fraction_list[*]};
# do
#     ${python} train.py \
# 	    --train=1 \
# 	    --max_seq_length=${max_seq_length} \
# 	    --max_step=330 \
# 	    --device=${device} \
# 	    --cuda_num=${device} \
# 	    --epoch=${epochs} \
# 	    --batch_size=${max_seq_length} \
# 	    --lr=${lr} \
# 	    --back_prediction=0 \
# 	    --gbias=0 \
# 	    --pretrained_model_path=${pretrained_model_path} \
# 	    --save_model_path=${save_model_path} \
# 	    --fraction=${fraction} 

#     export pretrained_model_path=${save_model_path}
# done

echo "WARNING: THIS FILE ONLY MAKES DEMO TESTING, WHILE ONLY ONE FRACTION IS APPLIED FOR CURRENT TASKS. NOT THE TRUE APPLICATION TESTS."

rate=0.1
num_id=5
export dataset_name="multiwoz-2.1-"
export model_save_path="_models_"

echo ">>>>>>>>>>>NORMAL GENERATION"
export inference_model_name="data/save_gbias_0_bp_0_rate_1.0_epoch_2/"
export save_file_path="${rate}_predict_files_rettig${num_id}_gbias_0_bp_0_fraction_1_epoch_2.txt"

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
    --cuda_num=0


export save_file_path="${rate}_predict_files_rettig${num_id}_gbias_0_bp_0_fraction_1_epoch_2.txt"
$python evaluate_control.py \
	--load_file_name $save_file_path

cd paraphrase/

echo "things done."
