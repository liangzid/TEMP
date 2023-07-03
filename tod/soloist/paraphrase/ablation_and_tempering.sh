#!/bin/bash

echo "-------------ablation experiments-------------------"

bash ablation_experiments.sh >1202-0.log

echo "-----tempering learning experiments--------"

bash 1.0.temperingLearning.sh >1202-1.log

echo "------ablation evaluation--------------------"

cd ../
bash 3.1.ablation_evaluation.sh >1202-2.log
cd paraphrase

