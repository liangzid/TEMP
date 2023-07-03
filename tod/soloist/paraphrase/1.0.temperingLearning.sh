#!/bin/bash

# export python=/home/szhang/anaconda3/envs/dslz/bin/python3
# export python=/home/zliang/anaconda3/envs/soloist/bin/python3
export python=/home/zliang/anaconda3/envs/dslz/bin/python3
# export root_dir="${HOME}/liangzi_need_smile/rettig/tem/"
# export root_dir="${HOME}/liangzi_need_smile/rettig/soloist/tem/"
# export root_dir="${HOME}/liangzi_need_smile/rettig/soloist/paraphrase/"
export root_dir="${HOME}/soloist/soloist/paraphrase/"

# echo "*********************: tempering learning with ASCENDING fraction"
# ##-----------------------------------------------------------------------------------------
# export device="1"
# export epochs=2
# export batch_size=4
# export lr=3e-5
# export max_seq_length=128
# # export import_pretrained_model_path="${root_dir}/data/electra-small-discriminator"
# export pretrained_model_path="${root_dir}/t5-small" 
# # export pretrained_model_path="${root_dir}/t5-v1-base" 
# export save_log_path="${root_dir}/log/fewshot_result_for_training_1020.log"
# # train stage.
# # export save_model_path="${root_dir}/data/rettig_model/"
# export save_model_path="${root_dir}/data/save_tempering_ascending_bp0_gbias0_num4/"
# export fraction_list=("0.0" "0.3" "0.5" "0.7")
# echo "--->>>BEGIN TO TRAINING in stage 1 with fraction ${frac}."
# for fraction in ${fraction_list[*]};
# do
#     ${python} train.py \
# 	    --train=1 \
# 	    --max_seq_length=${max_seq_length} \
# 	    --max_step=210 \
# 	    --device=${device} \
# 	    --cuda_num=${device} \
# 	    --epoch=${epochs} \
# 	    --batch_size=${max_seq_length} \
# 	    --lr=${lr} \
# 	    --back_prediction=0 \
# 	    --gbias=0 \
# 	    --pretrained_model_path=${pretrained_model_path} \
# 	    --save_model_path="${save_model_path}" \
# 	    --fraction=${fraction} 

#     export pretrained_model_path="${save_model_path}"
# done
# ##-----------------------------------------------------------------------------------------

echo "*********************: tempering learning with HORIZONTAL fraction 2"
##-----------------------------------------------------------------------------------------
export device="1"
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
export save_model_path="${root_dir}/data/save_tempering_horizontal_bp0_gbias0_num3/"
export fraction_list=("1.0" "1.0" "1.0")
echo "--->>>BEGIN TO TRAINING in stage 1 with fraction ${frac}."
for fraction in ${fraction_list[*]};
do
    ${python} train.py \
	    --train=1 \
	    --max_seq_length=${max_seq_length} \
	    --max_step=210 \
	    --device=${device} \
	    --cuda_num=${device} \
	    --epoch=${epochs} \
	    --batch_size=${max_seq_length} \
	    --lr=${lr} \
	    --back_prediction=0 \
	    --gbias=0 \
	    --pretrained_model_path=${pretrained_model_path} \
	    --save_model_path="${save_model_path}" \
	    --fraction=${fraction} 

    export pretrained_model_path="${save_model_path}"
done
##-----------------------------------------------------------------------------------------

# echo "*********************: tempering learning with DESCENDING fraction 3"
# ##-----------------------------------------------------------------------------------------
# export device="1"
# export epochs=2
# export batch_size=4
# export lr=3e-5
# export max_seq_length=128
# # export import_pretrained_model_path="${root_dir}/data/electra-small-discriminator"
# export pretrained_model_path="${root_dir}/t5-small" 
# # export pretrained_model_path="${root_dir}/t5-v1-base" 
# export save_log_path="${root_dir}/log/fewshot_result_for_training_1020.log"
# # train stage.
# # export save_model_path="${root_dir}/data/rettig_model/"
# export save_model_path="${root_dir}/data/save_tempering_descending_bp0_gbias0_num5/"
# export fraction_list=("0.9" "0.7" "0.5" "0.25" "0.0")
# echo "--->>>BEGIN TO TRAINING in stage 1 with fraction ${frac}."
# for fraction in ${fraction_list[*]};
# do
#     ${python} train.py \
# 	    --train=1 \
# 	    --max_seq_length=${max_seq_length} \
# 	    --max_step=210 \
# 	    --device=${device} \
# 	    --cuda_num=${device} \
# 	    --epoch=${epochs} \
# 	    --batch_size=${max_seq_length} \
# 	    --lr=${lr} \
# 	    --back_prediction=0 \
# 	    --gbias=0 \
# 	    --pretrained_model_path=${pretrained_model_path} \
# 	    --save_model_path="${save_model_path}" \
# 	    --fraction=${fraction} 

#     export pretrained_model_path="${save_model_path}"
# done
# ##-----------------------------------------------------------------------------------------

