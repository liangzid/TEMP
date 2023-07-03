#!/bin/bash
####################################################
#1.1.TRAIN_BART_DEFENDER ---

# Defender with Bart Backbone

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2022, ZiLiang, all rights reserved.
# Created: 17 十月 2022
####################################################
export python=/home/liangzi/anaconda3/envs/dslz/bin/python3
export python=/home/liangzi/anaconda3/envs/dslz2022/bin/python3
export root_dir="${HOME}/adc/"

##--------------------------------------------------------------------
export device="6"
export epochs=2
export batch_size=1
export lr=3e-5
export max_seq_length=128
export pretrained_model_path="${root_dir}/t5-small" 
# export pretrained_model_path="/home/liangzi/models/bart-base/" 
export save_log_path="${root_dir}/log/grid-step.log"
# train stage.
export target_num=1
export max_step=100000
# export max_step=2000
# export target_nums=(1 5 5)
# export target_nums=(1 1 1 1)
# export fractionls=(0.02 0.04 0.06 0.08)
export target_nums=(1 1 1 1)
export target_nums=(1)
# export fractionls=("0.7" "0.1" "0.3" "origin")
export fractionls=("0.04")
export epoch_nums=(2 2 2 2)
# export device_nums=(4 4 4 4)
export device_nums=(6 6 6 6)

for i in `seq 1 ${#target_nums[@]}`;
do
    export epochs=${epoch_nums[$i-1]}
    export target_num=${target_nums[$i-1]}
    export device=${device_nums[$i-1]}
    export fraction=${fractionls[$i-1]}
    echo "e: $epochs t: $target_num d: $device f:$fraction"
    echo ">>>NOW, TRAIN TIME."
    
    export save_model_path="${root_dir}/data/saved_defender_frac${fraction}_lr${lr}_epoch${epochs}_step${max_step}_target${target_num}_unsupervised/"
    echo "--->>>BEGIN TO TRAINING."

    # ${python} train.py \
    # 	    --train=1 \
    # 	    --max_seq_length=${max_seq_length} \
    # 	    --max_step=$max_step \
    # 	    --device=${device} \
    # 	    --cuda_num=${device} \
    # 	    --epoch=${epochs} \
    # 	    --batch_size=${batch_size} \
    # 	    --lr=${lr} \
    # 	    --back_prediction=0 \
    # 	    --target_num=${target_num} \
    # 	    --board_name="mytest" \
    # 	    --sample_method="random" \
    # 	    --gbias=0 \
    # 	    --pretrained_model_path=${pretrained_model_path} \
    # 	    --save_model_path=${save_model_path} \
    # 	    --fraction=${fraction}
    # # >1128-unsuper-varyingfrac.log &
    
    echo "--->>>TEST=====NONE,ADC=1,evaluation."

    # export safety_file_path4existingDialogueModel=("raw" "blenderbot-80M" \
	    # "DialoGPT-medium")
    export safety_file_path4existingDialogueModel=("raw")
    export test_cls_models=("none")
    export use_adc_ornot=("1")
    for safetyModelPath in ${safety_file_path4existingDialogueModel[*]};
    do
	for is_adc in ${use_adc_ornot[*]};
	do
	    for adc_prefix_cls in ${test_cls_models[*]};
	    do
	export save_model_path="${root_dir}/data/saved_defender_frac${fraction}_lr${lr}_epoch${epochs}_step${max_step}_target${target_num}_unsupervised/"
	echo "--->>>BEGIN TO INFERENCE."
	echo "inference load model path:${save_model_path}"
	echo "===>>>>>PARAMETERS:"
	echo "is using ADC:"${is_adc}
	echo "before adc, we add a filter called:"${adc_prefix_cls}
	echo "the file of the dialogue model is:"${safetyModelPath}
	${python} train_delthisifyousee.py \
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
		--board_name="mytest" \
		--sample_method="random" \
		--pretrained_model_path=${pretrained_model_path} \
		--save_model_path=${save_model_path} \
		--fraction=1.0 \
		--prefix_existing_model=${adc_prefix_cls} \
		--reading_safety_path=${safetyModelPath}\
		--is_using_ADC=${is_adc} 
	    done
	done
    done

done

# nohup bash 3.001.temptest.sh >1031-three_experiments_running_res.log &

echo "RUNNING 1.1.train_bart_defender.sh DONE."
# 1.1.train_bart_defender.sh ends here
