#!/bin/bash
export python=/home/liangzi/anaconda3/envs/dslz/bin/python3
export root_dir="${HOME}/TEMP/soloist/paraphrase/"

##----------------------------------------------------------------------------
export device="5"
# export epochs=4
export epochs=1
export batch_size=1
export lr=3e-5
export max_seq_length=128
# export import_pretrained_model_path="${root_dir}/data/electra-small-discriminator"
export pretrained_model_path="${root_dir}/t5-small" 
export save_log_path="${root_dir}/log/ablation_experiments.log"
# train stage.
export fraction="1.0"
export step=500
export prate_list=("0.1")
export target_num=1

# echo "========TEMP-WTA-SERIES========"

# echo ">>>>>WTA+Tempering3"
# export fraction_list=("1.0" "1.0" "1.0")
# for prate in ${prate_list[*]};
# do
#     export save_model_path="${root_dir}/data/tempering3-wta_fraction1.0_prate${prate}/"
#     # for fraction in ${fraction_list[*]};
#     # do
#     # 	${python} train.py \
#     # 		--train=1 \
#     # 		--max_seq_length=${max_seq_length} \
#     # 		--max_step=${step} \
#     # 		--device=${device} \
#     # 		--cuda_num=${device} \
#     # 		--epoch=${epochs} \
#     # 		--prate=${prate} \
#     # 		--target_num=${target_num} \
#     # 		--batch_size=${batch_size} \
#     # 		--lr=${lr} \
#     # 		--back_prediction=0 \
#     # 		--sample_method=wta \
#     # 		--gbias=0 \
#     # 		--pretrained_model_path=${pretrained_model_path} \
#     # 		--save_model_path=${save_model_path} \
#     # 		--board_name=${save_model_path} \
#     # 		--fraction=${fraction} 
#     # 	export pretrained_model_path=${save_model_path}
#     # done

#     cd ../
#     id_list=(5)
#     export rate=${prate}
#     for num_id in ${id_list[*]};
#     do
# 	export dataset_name="multiwoz-2.1-"
# 	export model_save_path="_models_"
# 	export inference_model_name=$save_model_path
# 	export save_file_path="${save_model_path}prate${prate}_numid${num_id}.txt"

# 	$python zcp_evaluate_multiwoz.py \
# 	    --dataset ${dataset_name}test \
# 	    --model ${rate}${model_save_path}${num_id} \
# 	    --file ${rate}_predict_files_${num_id}.txt \
# 	    --save_file ${save_file_path} \
# 	    --inference_model ${inference_model_name} \
# 	    --add_map 1\
# 	    --gbias 0\
# 	    --bp 0\
# 	    --cuda_num=${device}

# 	$python evaluate_control.py \
# 		--load_file_name $save_file_path
#     done
#     cd paraphrase/
# done
# ##----------------------------------------------------------------------------


# echo ">>>>>WTA+Tempering3+MT3"
# export fraction_list=("1.0" "1.0" "1.0")
# export target_num=3
# for prate in ${prate_list[*]};
# do
#     export save_model_path="${root_dir}/data/tempering3-mt3-wta_fraction1.0_prate${prate}/"
#     # for fraction in ${fraction_list[*]};
#     # do
#     # 	${python} train.py \
#     # 		--train=1 \
#     # 		--max_seq_length=${max_seq_length} \
#     # 		--max_step=${step} \
#     # 		--device=${device} \
#     # 		--cuda_num=${device} \
#     # 		--epoch=${epochs} \
#     # 		--prate=${prate} \
#     # 		--target_num=${target_num} \
#     # 		--batch_size=${batch_size} \
#     # 		--lr=${lr} \
#     # 		--back_prediction=0 \
#     # 		--sample_method=wta \
#     # 		--gbias=0 \
#     # 		--pretrained_model_path=${pretrained_model_path} \
#     # 		--save_model_path=${save_model_path} \
#     # 		--board_name=${save_model_path} \
#     # 		--fraction=${fraction} 
#     # 	export pretrained_model_path=${save_model_path}
#     # done

#     cd ../
#     id_list=(5)
#     export rate=${prate}
#     for num_id in ${id_list[*]};
#     do
# 	export dataset_name="multiwoz-2.1-"
# 	export model_save_path="_models_"
# 	export inference_model_name=$save_model_path
# 	export save_file_path="${save_model_path}prate${prate}_numid${num_id}.txt"

# 	$python evaluate_multiwoz.py \
# 	    --dataset ${dataset_name}test \
# 	    --model ${rate}${model_save_path}${num_id} \
# 	    --file ${rate}_predict_files_${num_id}.txt \
# 	    --save_file ${save_file_path} \
# 	    --inference_model ${inference_model_name} \
# 	    --add_map 1\
# 	    --gbias 0\
# 	    --bp 0\
# 	    --cuda_num=${device}

# 	$python evaluate_control.py \
# 		--load_file_name $save_file_path
#     done
#     cd paraphrase/
# done
# ##----------------------------------------------------------------------------

# echo ">>>>>WTA+Tempering3+MT3+BP"
# export fraction_list=("1.0" "1.0" "1.0")
# export target_num=3
# for prate in ${prate_list[*]};
# do
#     export save_model_path="${root_dir}/data/tempering3-mt3-bp-wta_fraction1.0_prate${prate}/"
#     # for fraction in ${fraction_list[*]};
#     # do
#     # 	${python} train.py \
#     # 		--train=1 \
#     # 		--max_seq_length=${max_seq_length} \
#     # 		--max_step=${step} \
#     # 		--device=${device} \
#     # 		--cuda_num=${device} \
#     # 		--epoch=${epochs} \
#     # 		--prate=${prate} \
#     # 		--target_num=${target_num} \
#     # 		--batch_size=${batch_size} \
#     # 		--lr=${lr} \
#     # 		--back_prediction=1 \
#     # 		--sample_method=wta \
#     # 		--gbias=0 \
#     # 		--pretrained_model_path=${pretrained_model_path} \
#     # 		--save_model_path=${save_model_path} \
#     # 		--board_name=${save_model_path} \
#     # 		--fraction=${fraction} 
#     # 	export pretrained_model_path=${save_model_path}
#     # done

#     cd ../
#     id_list=(5)
#     export rate=${prate}
#     for num_id in ${id_list[*]};
#     do
# 	export dataset_name="multiwoz-2.1-"
# 	export model_save_path="_models_"
# 	export inference_model_name=$save_model_path
# 	export save_file_path="${save_model_path}prate${prate}_numid${num_id}.txt"

# 	$python evaluate_multiwoz.py \
# 	    --dataset ${dataset_name}test \
# 	    --model ${rate}${model_save_path}${num_id} \
# 	    --file ${rate}_predict_files_${num_id}.txt \
# 	    --save_file ${save_file_path} \
# 	    --inference_model ${inference_model_name} \
# 	    --add_map 1\
# 	    --gbias 0\
# 	    --bp 0\
# 	    --cuda_num=${device}

# 	$python evaluate_control.py \
# 		--load_file_name $save_file_path
#     done
#     cd paraphrase/
# done
# ##----------------------------------------------------------------------------

# # echo ">>>>>WTA+BP"
# # export fraction_list=("1.0")
# # export target_num=1
# # for prate in ${prate_list[*]};
# # do
# #     export save_model_path="${root_dir}/data/bp-wta_fraction1.0_prate${prate}/"
# #     for fraction in ${fraction_list[*]};
# #     # do
# #     # 	${python} train.py \
# #     # 		--train=1 \
# #     # 		--max_seq_length=${max_seq_length} \
# #     # 		--max_step=${step} \
# #     # 		--device=${device} \
# #     # 		--cuda_num=${device} \
# #     # 		--epoch=${epochs} \
# #     # 		--prate=${prate} \
# #     # 		--target_num=${target_num} \
# #     # 		--batch_size=${batch_size} \
# #     # 		--lr=${lr} \
# #     # 		--back_prediction=1 \
# #     # 		--sample_method=wta \
# #     # 		--gbias=0 \
# #     # 		--pretrained_model_path=${pretrained_model_path} \
# #     # 		--save_model_path=${save_model_path} \
# #     # 		--board_name=${save_model_path} \
# #     # 		--fraction=${fraction} 
# #     # 	export pretrained_model_path=${save_model_path}
# #     # done

# #     cd ../
# #     id_list=(5)
# #     export rate=${prate}
# #     for num_id in ${id_list[*]};
# #     do
# # 	export dataset_name="multiwoz-2.1-"
# # 	export model_save_path="_models_"
# # 	export inference_model_name=$save_model_path
# # 	export save_file_path="${save_model_path}prate${prate}_numid${num_id}.txt"

# # 	$python evaluate_multiwoz.py \
# # 	    --dataset ${dataset_name}test \
# # 	    --model ${rate}${model_save_path}${num_id} \
# # 	    --file ${rate}_predict_files_${num_id}.txt \
# # 	    --save_file ${save_file_path} \
# # 	    --inference_model ${inference_model_name} \
# # 	    --add_map 1\
# # 	    --gbias 0\
# # 	    --bp 0\
# # 	    --cuda_num=${device}

# # 	$python evaluate_control.py \
# # 		--load_file_name $save_file_path
# #     done
# #     cd paraphrase/
# # done
# # ##----------------------------------------------------------------------------

# echo ">>>>>WTA+MT3"
# export fraction_list=("1.0")
# export target_num=3
# for prate in ${prate_list[*]};
# do
#     export save_model_path="${root_dir}/data/bp-wta_fraction1.0_prate${prate}/"
#     for fraction in ${fraction_list[*]};
#     do
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
# 	export pretrained_model_path=${save_model_path}
#     done

#     cd ../
#     id_list=(5)
#     export rate=${prate}
#     for num_id in ${id_list[*]};
#     do
# 	export dataset_name="multiwoz-2.1-"
# 	export model_save_path="_models_"
# 	export inference_model_name=$save_model_path
# 	export save_file_path="${save_model_path}prate${prate}_numid${num_id}.txt"

# 	$python evaluate_multiwoz.py \
# 	    --dataset ${dataset_name}test \
# 	    --model ${rate}${model_save_path}${num_id} \
# 	    --file ${rate}_predict_files_${num_id}.txt \
# 	    --save_file ${save_file_path} \
# 	    --inference_model ${inference_model_name} \
# 	    --add_map 1\
# 	    --gbias 0\
# 	    --bp 0\
# 	    --cuda_num=${device}

# 	$python evaluate_control.py \
# 		--load_file_name $save_file_path
#     done
#     cd paraphrase/
# done
# ##----------------------------------------------------------------------------

# echo ">>>>>WTA"
# export fraction_list=("1.0")
# export target_num=1
# for prate in ${prate_list[*]};
# do
#     export save_model_path="${root_dir}/data/bp-wta_fraction1.0_prate${prate}/"
#     for fraction in ${fraction_list[*]};
#     do
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
# 	export pretrained_model_path=${save_model_path}
#     done

#     cd ../
#     id_list=(5)
#     export rate=${prate}
#     for num_id in ${id_list[*]};
#     do
# 	export dataset_name="multiwoz-2.1-"
# 	export model_save_path="_models_"
# 	export inference_model_name=$save_model_path
# 	export save_file_path="${save_model_path}prate${prate}_numid${num_id}.txt"

# 	$python evaluate_multiwoz.py \
# 	    --dataset ${dataset_name}test \
# 	    --model ${rate}${model_save_path}${num_id} \
# 	    --file ${rate}_predict_files_${num_id}.txt \
# 	    --save_file ${save_file_path} \
# 	    --inference_model ${inference_model_name} \
# 	    --add_map 1\
# 	    --gbias 0\
# 	    --bp 0\
# 	    --cuda_num=${device}

# 	$python evaluate_control.py \
# 		--load_file_name $save_file_path
#     done
#     cd paraphrase/
# done
# ##----------------------------------------------------------------------------

echo "========TEMP-EXP-SERIES========"

echo ">>>>>EXP+Tempering3"
export fraction_list=("1.0" "1.0" "1.0")
for prate in ${prate_list[*]};
do
    export save_model_path="${root_dir}/data/tempering3-exp_fraction1.0_prate${prate}/"
    for fraction in ${fraction_list[*]};
    do
	${python} train.py \
		--train=1 \
		--max_seq_length=${max_seq_length} \
		--max_step=${step} \
		--device=${device} \
		--cuda_num=${device} \
		--epoch=${epochs} \
		--prate=${prate} \
		--target_num=${target_num} \
		--batch_size=${batch_size} \
		--lr=${lr} \
		--back_prediction=0 \
		--sample_method=exp \
		--gbias=0 \
		--pretrained_model_path=${pretrained_model_path} \
		--save_model_path=${save_model_path} \
		--board_name=${save_model_path} \
		--fraction=${fraction} 
	export pretrained_model_path=${save_model_path}
    done

    cd ../
    id_list=(5)
    export rate=${prate}
    for num_id in ${id_list[*]};
    do
	export dataset_name="multiwoz-2.1-"
	export model_save_path="_models_"
	export inference_model_name=$save_model_path
	export save_file_path="${save_model_path}prate${prate}_numid${num_id}.txt"

	$python zcp_evaluate_multiwoz.py \
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
##----------------------------------------------------------------------------


echo ">>>>>EXP+Tempering3+MT3"
export fraction_list=("1.0" "1.0" "1.0")
export target_num=3
for prate in ${prate_list[*]};
do
    export save_model_path="${root_dir}/data/tempering3-mt3-exp_fraction1.0_prate${prate}/"
    for fraction in ${fraction_list[*]};
    do
	${python} train.py \
		--train=1 \
		--max_seq_length=${max_seq_length} \
		--max_step=${step} \
		--device=${device} \
		--cuda_num=${device} \
		--epoch=${epochs} \
		--prate=${prate} \
		--target_num=${target_num} \
		--batch_size=${batch_size} \
		--lr=${lr} \
		--back_prediction=0 \
		--sample_method=exp \
		--gbias=0 \
		--pretrained_model_path=${pretrained_model_path} \
		--save_model_path=${save_model_path} \
		--board_name=${save_model_path} \
		--fraction=${fraction} 
	export pretrained_model_path=${save_model_path}
    done

    cd ../
    id_list=(5)
    export rate=${prate}
    for num_id in ${id_list[*]};
    do
	export dataset_name="multiwoz-2.1-"
	export model_save_path="_models_"
	export inference_model_name=$save_model_path
	export save_file_path="${save_model_path}prate${prate}_numid${num_id}.txt"

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
##----------------------------------------------------------------------------

echo ">>>>>EXP+Tempering3+MT3+BP"
export fraction_list=("1.0" "1.0" "1.0")
export target_num=3
for prate in ${prate_list[*]};
do
    export save_model_path="${root_dir}/data/tempering3-mt3-bp-exp_fraction1.0_prate${prate}/"
    for fraction in ${fraction_list[*]};
    do
	${python} train.py \
		--train=1 \
		--max_seq_length=${max_seq_length} \
		--max_step=${step} \
		--device=${device} \
		--cuda_num=${device} \
		--epoch=${epochs} \
		--prate=${prate} \
		--target_num=${target_num} \
		--batch_size=${batch_size} \
		--lr=${lr} \
		--back_prediction=1 \
		--sample_method=exp \
		--gbias=0 \
		--pretrained_model_path=${pretrained_model_path} \
		--save_model_path=${save_model_path} \
		--board_name=${save_model_path} \
		--fraction=${fraction} 
	export pretrained_model_path=${save_model_path}
    done

    cd ../
    id_list=(5)
    export rate=${prate}
    for num_id in ${id_list[*]};
    do
	export dataset_name="multiwoz-2.1-"
	export model_save_path="_models_"
	export inference_model_name=$save_model_path
	export save_file_path="${save_model_path}prate${prate}_numid${num_id}.txt"

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
##----------------------------------------------------------------------------

echo ">>>>>EXP+BP"
export fraction_list=("1.0")
export target_num=1
for prate in ${prate_list[*]};
do
    export save_model_path="${root_dir}/data/bp-exp_fraction1.0_prate${prate}/"
    for fraction in ${fraction_list[*]};
    do
	${python} train.py \
		--train=1 \
		--max_seq_length=${max_seq_length} \
		--max_step=${step} \
		--device=${device} \
		--cuda_num=${device} \
		--epoch=${epochs} \
		--prate=${prate} \
		--target_num=${target_num} \
		--batch_size=${batch_size} \
		--lr=${lr} \
		--back_prediction=1 \
		--sample_method=exp \
		--gbias=0 \
		--pretrained_model_path=${pretrained_model_path} \
		--save_model_path=${save_model_path} \
		--board_name=${save_model_path} \
		--fraction=${fraction} 
	export pretrained_model_path=${save_model_path}
    done

    cd ../
    id_list=(5)
    export rate=${prate}
    for num_id in ${id_list[*]};
    do
	export dataset_name="multiwoz-2.1-"
	export model_save_path="_models_"
	export inference_model_name=$save_model_path
	export save_file_path="${save_model_path}prate${prate}_numid${num_id}.txt"

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
##----------------------------------------------------------------------------

echo ">>>>>EXP+MT3"
export fraction_list=("1.0")
export target_num=3
for prate in ${prate_list[*]};
do
    export save_model_path="${root_dir}/data/bp-exp_fraction1.0_prate${prate}/"
    for fraction in ${fraction_list[*]};
    do
	${python} train.py \
		--train=1 \
		--max_seq_length=${max_seq_length} \
		--max_step=${step} \
		--device=${device} \
		--cuda_num=${device} \
		--epoch=${epochs} \
		--prate=${prate} \
		--target_num=${target_num} \
		--batch_size=${batch_size} \
		--lr=${lr} \
		--back_prediction=0 \
		--sample_method=exp \
		--gbias=0 \
		--pretrained_model_path=${pretrained_model_path} \
		--save_model_path=${save_model_path} \
		--board_name=${save_model_path} \
		--fraction=${fraction} 
	export pretrained_model_path=${save_model_path}
    done

    cd ../
    id_list=(5)
    export rate=${prate}
    for num_id in ${id_list[*]};
    do
	export dataset_name="multiwoz-2.1-"
	export model_save_path="_models_"
	export inference_model_name=$save_model_path
	export save_file_path="${save_model_path}prate${prate}_numid${num_id}.txt"

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
##----------------------------------------------------------------------------

echo ">>>>>EXP"
export fraction_list=("1.0")
export target_num=1
for prate in ${prate_list[*]};
do
    export save_model_path="${root_dir}/data/bp-exp_fraction1.0_prate${prate}/"
    for fraction in ${fraction_list[*]};
    do
	${python} train.py \
		--train=1 \
		--max_seq_length=${max_seq_length} \
		--max_step=${step} \
		--device=${device} \
		--cuda_num=${device} \
		--epoch=${epochs} \
		--prate=${prate} \
		--target_num=${target_num} \
		--batch_size=${batch_size} \
		--lr=${lr} \
		--back_prediction=0 \
		--sample_method=exp \
		--gbias=0 \
		--pretrained_model_path=${pretrained_model_path} \
		--save_model_path=${save_model_path} \
		--board_name=${save_model_path} \
		--fraction=${fraction} 
	export pretrained_model_path=${save_model_path}
    done

    cd ../
    id_list=(5)
    export rate=${prate}
    for num_id in ${id_list[*]};
    do
	export dataset_name="multiwoz-2.1-"
	export model_save_path="_models_"
	export inference_model_name=$save_model_path
	export save_file_path="${save_model_path}prate${prate}_numid${num_id}.txt"

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
##----------------------------------------------------------------------------


echo "RUNNING ablation_1209new.sh DONE."
# ablation_1209new.sh ends here
