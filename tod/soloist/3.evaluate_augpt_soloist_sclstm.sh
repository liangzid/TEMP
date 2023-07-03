#!/bin/bash

export python=/home/zliang/anaconda3/envs/dslz/bin/python3
export root_dir="${HOME}/soloist/soloist/"

##-----------------------------------------------------------------------------------------
export device="1"
export save_log_path="${root_dir}/log/soloist_all_experiments.log"
export prate_list=("0.01" "0.02" "0.04" "0.06" "0.08" "0.1")


echo ">>>>>>>>>>begin to run AU-GPT evaluation."
for prate in ${prate_list[*]};
do

    id_list=(1 2 3 4 5)

    export rate=${prate}
   
    for num_id in ${id_list[*]};
    do
	export dataset_name="multiwoz-2.1-"
	export model_save_path="_models_"


	$python zcp_evaluate_multiwoz.py \
	    --dataset ${dataset_name}test \
	    --file ${rate}_predict_files_${num_id}.txt \
	    --add_map 0\
	    --gbias 0\
	    --bp 0\
	    --cuda_num=${device}
	export save_file_path=${rate}_predict_files_${num_id}.txt

	$python evaluate_control.py \
		--load_file_name $save_file_path
    done
done


echo ">>>>>>>>>>begin to run SOLOIST evaluation."
for prate in ${prate_list[*]};
do

    id_list=(1 2 3 4 5)

    export rate=${prate}
   
    for num_id in ${id_list[*]};
    do
	export dataset_name="multiwoz-2.1-"
	export model_save_path="_models_"


	$python zcp_evaluate_multiwoz.py \
	    --dataset ${dataset_name}test \
	    --file ${rate}_soloist_predict_files_${num_id}.txt \
	    --add_map 0\
	    --gbias 0\
	    --bp 0\
	    --cuda_num=${device}

	export save_file_path=${rate}_soloist_predict_files_${num_id}.txt

	$python evaluate_control.py \
		--load_file_name $save_file_path
    done
done

echo ">>>>>>>>>>begin to run SCLSTM evaluation."
for prate in ${prate_list[*]};
do

    id_list=(1 2 3 4 5)

    export rate=${prate}
   
    for num_id in ${id_list[*]};
    do
	export dataset_name="multiwoz-2.1-"
	export model_save_path="_models_"


	$python zcp_evaluate_multiwoz.py \
	    --dataset ${dataset_name}test \
	    --file ${rate}_sclstm_predict_files_${num_id}.txt \
	    --add_map 0\
	    --gbias 0\
	    --bp 0\
	    --cuda_num=${device}

	export save_file_path=${rate}_sclstm_predict_files_${num_id}.txt

	$python evaluate_control.py \
		--load_file_name $save_file_path
    done
done
