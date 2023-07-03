#!/bin/bash

export python=/home/zliang/anaconda3/envs/dslz/bin/python3
export root_dir="${HOME}/soloist/soloist/paraphrase/"

##-----------------------------------------------------------------------------------------
export device="1"
# export epochs=4
export epochs=1
export batch_size=1
export lr=3e-5
export max_seq_length=128
# export import_pretrained_model_path="${root_dir}/data/electra-small-discriminator"
export pretrained_model_path="${root_dir}/t5-small" 
export save_log_path="${root_dir}/log/scgpt_all_experiments.log"
# train stage.
export fraction="1.0"
export step=500
export prate_list=("0.01" "0.02" "0.04" "0.06" "0.08" "0.1")
export prate_list=("0.04" "0.1")
export target_num=3


echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> tempering + exp+bp+multi-target 3"
# echo "--->>>BEGIN TO TRAINING in stage 1 with fraction ${frac}."
export target_num=3
export pretrained_model_path="${root_dir}/t5-small" 
export fraction_list=("1.0" "1.0" "1.0")
for prate in ${prate_list[*]};
do

    ## this is training TEMP paraphrasing model.

    # export save_model_path="${root_dir}/data/mt3-tempering-exp-bp_fraction1.0_prate${prate}/"

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
    # 		--back_prediction=1 \
    # 		--sample_method=exp \
    # 		--gbias=0 \
    # 		--pretrained_model_path=${pretrained_model_path} \
    # 		--save_model_path=${save_model_path} \
    # 		--board_name=${save_model_path} \
    # 		--fraction=${fraction} 
    # 	export pretrained_model_path=$save_model_path
    # done

    cd ../
    id_list=(1 2 3 4 5)

    export rate=${prate}
   
    for num_id in ${id_list[*]};
    do
	export dataset_name="multiwoz-2.1-"
	export model_save_path="_models_"

	export inference_model_name="data/mt3-tempering-exp-bp_fraction1.0_prate${prate}/"
	export save_file_path="scgpt-mt3-tempering-exp-bp_fraction1.0_prate${prate}_numid${num_id}.txt"

	$python evaluate_multiwoz.py \
	    --dataset ${dataset_name}test \
	    --file scgpt/${rate}_predict_files_${num_id}.txt \
	    --save_file ${save_file_path} \
	    --inference_model ${inference_model_name} \
	    --add_map 1\
	    --gbias 0\
	    --bp 1\
	    --cuda_num=${device}

	$python evaluate_control.py \
		--load_file_name $save_file_path
    done

    cd paraphrase/
done
##-----------------------------------------------------------------------------------------


echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> tempering + wta + bp + multi-target 3"
# echo "--->>>BEGIN TO TRAINING in stage 1 with fraction ${frac}."
export target_num=3
export pretrained_model_path="${root_dir}/t5-small" 
export fraction_list=("1.0" "1.0" "1.0")
for prate in ${prate_list[*]};
do
    export save_model_path="${root_dir}/data/mt3-tempering-wta-bp_fraction1.0_prate${prate}/"

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
    # 		--back_prediction=1 \
    # 		--sample_method=wta \
    # 		--gbias=0 \
    # 		--pretrained_model_path=${pretrained_model_path} \
    # 		--save_model_path=${save_model_path} \
    # 		--board_name=${save_model_path} \
    # 		--fraction=${fraction} 
    # 	export pretrained_model_path=$save_model_path
    # done

    cd ../
    id_list=(1 2 3 4 5)

    export rate=${prate}
   
    for num_id in ${id_list[*]};
    do
	export dataset_name="multiwoz-2.1-"
	export model_save_path="_models_"

	export inference_model_name="data/mt3-tempering-wta-bp_fraction1.0_prate${prate}/"
	export save_file_path="scgpt-mt3-tempering-wta-bp_fraction1.0_prate${prate}_numid${num_id}.txt"

	$python evaluate_multiwoz.py \
	    --dataset ${dataset_name}test \
	    --file ./scgpt/${rate}_predict_files_${num_id}.txt \
	    --save_file ${save_file_path} \
	    --inference_model ${inference_model_name} \
	    --add_map 1\
	    --gbias 0\
	    --bp 1\
	    --cuda_num=${device}

	$python evaluate_control.py \
		--load_file_name $save_file_path
    done

    cd paraphrase/
done
##-----------------------------------------------------------------------------------------
