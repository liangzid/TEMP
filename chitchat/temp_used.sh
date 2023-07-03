#!/bin/bash

# before running this file, I sould make sure:
# 1. I have collect all dialogue models running results.
# 2. and the running results of parlAI related data.
# 3. and the reulst of XXX

export python=/home/liangzi/anaconda3/envs/dslz/bin/python3
export root_dir="${HOME}/adc/"

##-----------------------------------------------------------------------------
export device="3"
export epochs=1
export batch_size=1
export lr=3e-5
export max_seq_length=128
export pretrained_model_path="${root_dir}/t5-small" 
export save_log_path="${root_dir}/log/grid-step.log"
# train stage.
# export step_list=("450" "550" "650")
export target_num=1
# export step_list=( "10000" "10000" "10000" "10000" "10000" "10000")
# export step_list=( "2000" "2000" "2000" "2000" "2000" "2000")
# export step_list=( "100000")
export step_list=( "2000")

# export prefix_inference_model="ours_cls"
export prefix_inference_model="none"
# export prefix_inference_model="random"
# export prefix_inference_model="detoxify"
export  safety_path="raw"
# export is_using_ADC="0"
export is_using_ADC="1"
# export is_using_ADC="0"
export max_step="2000"

## this is the whole running procedure.
## running this part for all the experiments.

export test_cls_models=("none" "random" "detoxify" \
	"BBF" "BAD" "ours_cls")
# export test_cls_models=("BBF" "BAD")
export adcprefix4models=("none")
# export safety_file_path4existingDialogueModel=("raw" "blenderbot-80M" \
	# "DialoGPT-medium" "Plato-2")
export safety_file_path4existingDialogueModel=("raw" "blenderbot-80M" \
	"DialoGPT-medium")
export use_adc_ornot=("0")
for safetyModelPath in ${safety_file_path4existingDialogueModel[*]};
do
    for is_adc in ${use_adc_ornot[*]};
    do
	for adc_prefix_cls in ${test_cls_models[*]};
	do
    export save_model_path="${root_dir}/data/saved_defener_lr${lr}_epoch${epochs}_step${max_step}/"
    echo "--->>>BEGIN TO INFERENCE."
    echo "inference load model path:${save_model_path}"
    echo "===>>>>>PARAMETERS:"
    echo "is using ADC:"${is_adc}
    echo "before adc, we add a filter called:"${adc_prefix_cls}
    echo "the file of the dialogue model is:"${safetyModelPath}
    ${python} train.py \
	    --train=0 \
	    --max_seq_length=${max_seq_length} \
	    --max_step=$max_step \
	    --device=${device} \
	    --cuda_num=${device} \
	    --epoch=${epochs} \
	    --batch_size=${batch_size} \
	    --lr=${lr} \
	    --back_prediction=0 \
	    --target_num=${target_num} \
	    --pretrained_model_path=${pretrained_model_path} \
	    --save_model_path=${save_model_path} \
	    --prefix_existing_model=${adc_prefix_cls} \
	    --reading_safety_path=${safetyModelPath}\
	    --is_using_ADC=${is_adc} 
	done
    done
done

echo "ALL things done."








