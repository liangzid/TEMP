#!/bin/bash

export python=/home/szhang/anaconda3/envs/soloist/bin/python3

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
       --gpu_num 6 \
       --use_wandb 0\
       --fp16 \
       --clean-samples \
       --model_save_path "finetuned_augpt"


echo "EVERYTHING DONEEE."
       






