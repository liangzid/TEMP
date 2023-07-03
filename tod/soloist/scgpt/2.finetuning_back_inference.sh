#!/bin/bash

export python=/home/zliang/anaconda3/envs/soloist/bin/python3

echo "begin to easiest train."
$python train_multiwoz.py \
       --train-dataset multiwoz-2.1-train \
       --dev-dataset multiwoz-2.1-val \
       --model ./augpt-bigdata/ \
       --backtranslations none \
       --response-loss unlikelihood \
       --epochs 5 \
       --have_template 0\
       --use_rettig 0\
       --backinference 1\
       --gpu_num 0 \
       --batch-size 2\
       --use_wandb 0\
       --fp16 \
       --clean-samples \
       --model_save_path "finetuned_back_prediction" 


export python=/home/zliang/anaconda3/envs/soloist/bin/python3
export root_dir="${HOME}/backinference/"

##-----------------------------------------------------------------------------------------
export device="1"
export save_log_path="${root_dir}/log/backinfernce_soloist_all_experiments.log"


echo ">>>>>>>>>>begin to run AU-GPT evaluation."

export dataset_name="multiwoz-2.1-"
export model_save_path="_models_"


$python evaluate_multiwoz.py \
	--model "${root_dir}/finetuned_back_prediction/" \
	--dataset ${dataset_name}test \
	--save_file "backinference_test.txt" \
	--add_map 0\
	--cuda_num=${device}


# echo "begin to normal train."
# $python train_multiwoz.py \
#        --train-dataset multiwoz-2.1-train \
#        --dev-dataset multiwoz-2.1-val \
#        --model ./augpt-bigdata/ \
#        --backtranslations latest \
#        --response-loss unlikelihood \
#        --epochs 5 \
#        --fp16 \
#        --clean-samples \


echo "EVERYTHING DONEEE."
       
