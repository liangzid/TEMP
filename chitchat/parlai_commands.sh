#!/bin/bash
######################################################################
#PARLAI_COMMANDS ---
# this file is for evaluate existing dataset with parlai library
# before you should better run `pip install parlai==1.6.0`

# Author: Zi Liang <liangzid@stu.xjtu.edu.cn>
# Copyright © 2022, ZiLiang, all rights reserved.
# Created: 20 七月 2022
######################################################################

export python=/home/liangzi/anaconda3/envs/opendomaindialogue/bin/python3
export test_set_path="/home/liangzi/adc/DiaSafety/DiaSafety_dataset/parlai_test.json"
# export test_set_path="/home/liangzi/adc/DiaSafety/DiaSafety_dataset/parlai_test.txt"
export test_set_path="/home/liangzi/adc/DiaSafety/DiaSafety_dataset/parlai_test_only_unsafe.txt"

# export http_proxy=http://liangz:123456@202.117.54.213:80

# # >>> BBF
$python -m parlai eval_model --task fromfile:parlaiformat\
       --fromfile_datapath "${test_set_path}" \
       -m parlai.agents.transformer.classifier:ClassifierAgent\
       -mf zoo:dialogue_safety/single_turn/model\
       -bs 1\
       --world-logs $test_set_path.dialogu_safety_parlai.jsonl\
       --no-cuda

# >>> BAD
$python -m parlai eval_model --task fromfile:parlaiformat\
       --fromfile_datapath "${test_set_path}" \
       --m parlai.agents.transformer.classifier:ClassifierAgent\
       --mf zoo:bot_adversarial_dialogue/multi_turn/model\
       --bs 1\
       --world-logs $test_set_path.BAD_parlai.jsonl\
       --no-cuda

# # >>> SaFeRDialogue
# $python -m parlai eval_model --task fromfile:parlaiformat\
#        --fromfile_datapath "${test_set_path}" \
#        -mf zoo:saferdialogues/model\
#        -bs 1\
#        --world-logs $test_set_path.SafeRDialog_parlai.jsonl\
#        --no-cuda

       # -m parlai.agents.transformer.classifier:ClassifierAgent\


# $python -m parlai eval_model --task fromfile:parlaiformat\
#        --fromfile_datapath "${test_set_path}" \
#        --dt test \
#        -m parlai.agents.transformer.classifier:ClassifierAgent\
#        -mf zoo:dialogue_safety/single_turn/model\
#        -bs 1\
#        --no-cuda



# parlai eval_model -t dialogue_safety:adversarial --round 3 -dt test -m parlai.agents.transformer.classifier:ClassifierAgent -mf zoo:dialogue_safety/single_turn/model -bs 1 --no-cuda


echo "RUNNING parlai_commands.sh DONE."
# parlai_commands.sh ends here
