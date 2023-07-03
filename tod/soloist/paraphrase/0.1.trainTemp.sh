#!/bin/bash

# export python=/home/szhang/anaconda3/envs/dslz/bin/python3
# export python=/home/zliang/anaconda3/envs/soloist/bin/python3
export python=/home/zliang/anaconda3/envs/dslz/bin/python3
# export root_dir="${HOME}/liangzi_need_smile/rettig/tem/"
# export root_dir="${HOME}/liangzi_need_smile/rettig/soloist/tem/"
# export root_dir="${HOME}/liangzi_need_smile/rettig/soloist/paraphrase/"
export root_dir="${HOME}/soloist/soloist/paraphrase/"

##-----------------------------------------------------------------------------------------
export device="1"
export epochs=30
export batch_size=4
export lr=3e-5
export max_seq_length=128
# export import_pretrained_model_path="${root_dir}/data/electra-small-discriminator"
export pretrained_model_path="${root_dir}/t5-small" 
# export pretrained_model_path="${root_dir}/t5-v1-base" 
export save_log_path="${root_dir}/log/fewshot_result_for_training_1020.log"
# train stage.
# export save_model_path="${root_dir}/data/rettig_model/"
export save_model_path="${root_dir}/data/save_without_bp_lr_3e-5_rate-0.8-30/"
export fraction_list=( "0.3" "0.4" "0.5" "0.6" "0.7")
export fraction_list=("0.8" )
echo "--->>>BEGIN TO TRAINING in stage 1 with fraction ${frac}."
for fraction in ${fraction_list[*]};
do
    ${python} train.py \
	    --train=1 \
	    --max_seq_length=${max_seq_length} \
	    --device=${device} \
	    --cuda_num=${device} \
	    --epoch=${epochs} \
	    --batch_size=${max_seq_length} \
	    --lr=${lr} \
	    --back_prediction=0 \
	    --pretrained_model_path=${pretrained_model_path} \
	    --save_model_path="${save_model_path}" \
	    --fraction=${fraction} 

    export pretrained_model_path="${root_dir}/data/rettig_model/"
done
##-----------------------------------------------------------------------------------------


# # # test it.
# # echo "--->>>BEGIN TO TEST  with rate {frac} on batchsize 1."
# # ${python} fewshot_train.py \
# # 	--train=0 \
# # 	--max_seq_length=${max_seq_length} \
# # 	--device=${device} \
# # 	--epoch=${epochs} \
# # 	--batch_size=1 \
# # 	--lr=${lr} \
# # 	--pretrained_model_path=${pretrained_model_path} \
# # 	--save_model_path=${pretrained_model_path} \
# # 	--dataset_path=${datasetpath} \
# # 	--save_train_template_list_path="${save_template_prefix}0.${frac}_train.pk" \
# # 	--save_test_template_list_path="${save_template_prefix}0.${frac}_test.pk" \
# # 	--vocab_path=${vocabpath} \
# # 	--num_labels=102

# # # echo ">>>|!DONE!|<<<"
