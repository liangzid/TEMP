#!/bin/bash
######################################################################
#3.1.THREE_QUALITY_EVALUATIONS ---

# Evaluates the MDD-Eval, the acceptability and the engagements.

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2022, ZiLiang, all rights reserved.
# Created: 13 十月 2022
######################################################################

######################## Commentary ##################################
##  
######################################################################

export python=/home/liangzi/anaconda3/envs/dslz/bin/python3
export root_dir="${HOME}/adc/"

# echo ">>>first evaluate the MDD-Eval"

cd "$root_dir/DiaSafety/DiaSafety_dataset/"
$python transfer2MDDformat.py
cd $root_dir

# conda activate tf1-nv

# cd "$root_dir/neuralmetric/MDD-Eval/"
# bash infer.sh
# $python evaluation_notebook.py
# cd $root_dir
# echo "MDD EVAL DONE"

echo ">>>then evaluate ACCEPT and ENGAGEMENT"

cd "$root_dir/neuralmetric/proxy_indicators/src/"
$python calculate_unsupervised_quality.py

echo "ACCEPT and ENGAGE EVA DONE."


echo "RUNNING 3.1.three_quality_evaluations.sh DONE."
# 3.1.three_quality_evaluations.sh ends here
