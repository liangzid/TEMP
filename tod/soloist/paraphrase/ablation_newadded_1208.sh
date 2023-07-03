#!/bin/bash

# TEMP-wta w.o. BP, which means: target, tempring.

export python=/home/liangzi/anaconda3/envs/dslz/bin/python3
export root_dir="${HOME}/TEMP/soloist/paraphrase/"

##----------------------------------------------------------------------------
export device="6"
export epochs=1
export batch_size=1
export lr=3e-5
export max_seq_length=128
export pretrained_model_path="${root_dir}/t5-small" 
export save_log_path="${root_dir}/log/ablation_experiments.log"
# train stage.
export fraction="1.0"
export step=500
export prate_list=("0.1")
export target_num=-1

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>wta without BP"
export target_num=3
export fraction_list=("1.0" "1.0" "1.0")
for prate in ${prate_list[*]};
do
    export save_model_path="${root_dir}/data/tempering3-tar3-wta_fraction1.0_prate${prate}/"

    # for fraction in ${fraction_list[*]};
    # do
    # 	${python} train.py \
    # 		--train=1 \
    # 		--max_seq_length=${max_seq_length} \
    # 		--max_step=${step} \
    # 		--device=${device} \
    # 		--cuda_num=${device} \
    # 		--epoch=${epochs} \
    # 		--prate=${prate} \
    # 		--target_num=${target_num} \
    # 		--batch_size=${batch_size} \
    # 		--lr=${lr} \
    # 		--back_prediction=0 \
    # 		--sample_method=wta \
    # 		--gbias=0 \
    # 		--pretrained_model_path=${pretrained_model_path} \
    # 		--save_model_path=${save_model_path} \
    # 		--board_name=${save_model_path} \
    # 		--fraction=${fraction} 
    # 	export pretrained_model_path=$save_model_path
    # done

    cd ../
    # id_list=(1 2 3 4 5)
    id_list=(1)

    export rate=${prate}
   
    for num_id in ${id_list[*]};
    do
	export dataset_name="multiwoz-2.1-"
	export model_save_path="_models_"

	export inference_model_name="data/tempering3-tar3-wta_fraction1.0_prate${prate}/"
	export save_file_path="tempering3-target3-wta_fraction1.0_prate${prate}_numid${num_id}.txt"

	$python evaluate_multiwoz.py \
	    --dataset ${dataset_name}test \
	    --model ${rate}${model_save_path}${num_id} \
	    --file ${rate}_predict_files_${num_id}.txt \
	    --save_file ${save_file_path} \
	    --inference_model ${inference_model_name} \
	    --add_map 1\
	    --gbias 0\
	    --bp 0\
	    --cuda_num=${device}

	$python evaluate_control.py \
		--load_file_name $save_file_path
    done

    cd paraphrase/
done
##-----------------------------------------------------------------------------------------

