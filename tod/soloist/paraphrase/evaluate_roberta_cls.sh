#!/bin/bash

export python=/home/liangzi/anaconda3/envs/dslz/bin/python3
export root_dir="${HOME}/TEMP/soloist/paraphrase/"

##-----------------------------------------------------------------------------------------
export device="6"
# export epochs=4
export epochs=1
export batch_size=1
export lr=3e-5
export max_seq_length=128
# export import_pretrained_model_path="${root_dir}/data/electra-small-discriminator"
export pretrained_model_path="${root_dir}/t5-small" 
export save_log_path="${root_dir}/log/evalute_raw_roberta_cls.log"
# train stage.
export fraction="1.0"
export step=500
# export prate_list=("0.01" "0.02" "0.04" "0.06" "0.08" "0.1")
export prate_list=("0.04" "0.1")
# export prate_list=("0.1")
export target_num=3


cd ../
for prate in ${prate_list[*]};
do
    # export save_model_path="${root_dir}/data/damdmt3-tempering-exp-bp_fraction1.0_prate${prate}/"

    id_list=(1 2 3 4 5)

    export rate=${prate}
   
    for num_id in ${id_list[*]};
    do
	export dataset_name="multiwoz-2.1-"
	export model_save_path="_models_"

	# export inference_model_name="data/damdmt3-tempering-exp-bp_fraction1.0_prate${prate}/"
	# export save_file_path="damdmt3-tempering-exp-bp_fraction1.0_prate${prate}_numid${num_id}.txt"

	export from_path="/home/liangzi/TEMP/soloist/robertaCLS/"

	$python zcp_evaluate_multiwoz.py \
	    --dataset ${dataset_name}test \
	    --file ${from_path}Hpollution_${rate}_roberta_cls_predict_files_${num_id}.txt \
	    --add_map 0\
	    --gbias 0\
	    --cuda_num=${device}

	$python evaluate_control.py \
		--load_file_name ${from_path}Hpollution_${rate}_roberta_cls_predict_files_${num_id}.txt
done
done
