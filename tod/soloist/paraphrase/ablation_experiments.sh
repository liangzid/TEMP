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
export save_log_path="${root_dir}/log/ablation_experiments.log"
# train stage.
export fraction="1.0"
export step=500
export prate_list=("0.01" "0.02" "0.04" "0.06" "0.08" "0.1")
export target_num=1

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>NORMAL GENERATION"
# echo "--->>>BEGIN TO TRAINING in stage 1 with fraction ${frac}."
for prate in ${prate_list[*]};
do
    export save_model_path="${root_dir}/data/normal_fraction1.0_prate${prate}/"
    # ${python} train.py \
    # 	    --train=1 \
    # 	    --max_seq_length=${max_seq_length} \
    # 	    --max_step=${step} \
    # 	    --device=${device} \
    # 	    --cuda_num=${device} \
    # 	    --epoch=${epochs} \
    # 	    --prate=${prate} \
    # 	    --target_num=${target_num} \
    # 	    --batch_size=${batch_size} \
    # 	    --lr=${lr} \
    # 	    --back_prediction=0 \
    # 	    --sample_method=wta \
    # 	    --gbias=0 \
    # 	    --pretrained_model_path=${pretrained_model_path} \
    # 	    --save_model_path=${save_model_path} \
    # 	    --board_name=${save_model_path} \
    # 	    --fraction=${fraction} 

    cd ../
    id_list=(1 2 3 4 5)

    export rate=${prate}
   
    for num_id in ${id_list[*]};
    do
	export dataset_name="multiwoz-2.1-"
	export model_save_path="_models_"

	export inference_model_name="data/normal_fraction1.0_prate${prate}/"
	export save_file_path="normal_fraction1.0_prate${prate}_numid${num_id}.txt"

	# $python evaluate_multiwoz.py \
	#     --dataset ${dataset_name}test \
	#     --model ${rate}${model_save_path}${num_id} \
	#     --file ${rate}_predict_files_${num_id}.txt \
	#     --save_file ${save_file_path} \
	#     --inference_model ${inference_model_name} \
	#     --add_map 1\
	#     --gbias 0\
	#     --bp 0\
	#     --cuda_num=${device}

	$python evaluate_control.py \
		--load_file_name $save_file_path
    done

    cd paraphrase/
done
##-----------------------------------------------------------------------------------------



echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>with BP GENERATION"
# echo "--->>>BEGIN TO TRAINING in stage 1 with fraction ${frac}."
for prate in ${prate_list[*]};
do
    export save_model_path="${root_dir}/data/bp_fraction1.0_prate${prate}/"
    # ${python} train.py \
    # 	    --train=1 \
    # 	    --max_seq_length=${max_seq_length} \
    # 	    --max_step=${step} \
    # 	    --device=${device} \
    # 	    --cuda_num=${device} \
    # 	    --epoch=${epochs} \
    # 	    --prate=${prate} \
    # 	    --target_num=${target_num} \
    # 	    --batch_size=${batch_size} \
    # 	    --lr=${lr} \
    # 	    --back_prediction=1 \
    # 	    --sample_method=wta \
    # 	    --gbias=0 \
    # 	    --pretrained_model_path=${pretrained_model_path} \
    # 	    --save_model_path=${save_model_path} \
    # 	    --board_name=${save_model_path} \
    # 	    --fraction=${fraction} 

    cd ../
    id_list=(1 2 3 4 5)

    export rate=${prate}
   
    for num_id in ${id_list[*]};
    do
	export dataset_name="multiwoz-2.1-"
	export model_save_path="_models_"

	export inference_model_name="data/bp_fraction1.0_prate${prate}/"
	export save_file_path="bp_fraction1.0_prate${prate}_numid${num_id}.txt"

	# $python evaluate_multiwoz.py \
	#     --dataset ${dataset_name}test \
	#     --model ${rate}${model_save_path}${num_id} \
	#     --file ${rate}_predict_files_${num_id}.txt \
	#     --save_file ${save_file_path} \
	#     --inference_model ${inference_model_name} \
	#     --add_map 1\
	#     --gbias 0\
	#     --bp 1\
	#     --cuda_num=${device}

	$python evaluate_control.py \
		--load_file_name $save_file_path
    done

    cd paraphrase/
done
##-----------------------------------------------------------------------------------------

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Multi-target GENERATION"
# echo "--->>>BEGIN TO TRAINING in stage 1 with fraction ${frac}."
export target_num=3
for prate in ${prate_list[*]};
do
    export save_model_path="${root_dir}/data/target${target_num}_fraction1.0_prate${prate}/"
    # ${python} train.py \
    # 	    --train=1 \
    # 	    --max_seq_length=${max_seq_length} \
    # 	    --max_step=${step} \
    # 	    --device=${device} \
    # 	    --cuda_num=${device} \
    # 	    --epoch=${epochs} \
    # 	    --prate=${prate} \
    # 	    --target_num=${target_num} \
    # 	    --batch_size=${batch_size} \
    # 	    --lr=${lr} \
    # 	    --back_prediction=0 \
    # 	    --sample_method=random \
    # 	    --gbias=0 \
    # 	    --pretrained_model_path=${pretrained_model_path} \
    # 	    --save_model_path=${save_model_path} \
    # 	    --board_name=${save_model_path} \
    # 	    --fraction=${fraction} 

    cd ../
    id_list=(1 2 3 4 5)

    export rate=${prate}
   
    for num_id in ${id_list[*]};
    do
	export dataset_name="multiwoz-2.1-"
	export model_save_path="_models_"

	export inference_model_name="data/target${target_num}_fraction1.0_prate${prate}/"
	export save_file_path="target${target_num}_fraction1.0_prate${prate}_numid${num_id}.txt"

	# $python evaluate_multiwoz.py \
	#     --dataset ${dataset_name}test \
	#     --model ${rate}${model_save_path}${num_id} \
	#     --file ${rate}_predict_files_${num_id}.txt \
	#     --save_file ${save_file_path} \
	#     --inference_model ${inference_model_name} \
	#     --add_map 1\
	#     --gbias 0\
	#     --bp 0\
	#     --cuda_num=${device}

	$python evaluate_control.py \
		--load_file_name $save_file_path
    done

    cd paraphrase/
done
##-----------------------------------------------------------------------------------------


echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Multi-target bp exp GENERATION"
# echo "--->>>BEGIN TO TRAINING in stage 1 with fraction ${frac}."
export target_num=3
for prate in ${prate_list[*]};
do
    export save_model_path="${root_dir}/data/exp-bp-target${target_num}_fraction1.0_prate${prate}/"
    # ${python} train.py \
    # 	    --train=1 \
    # 	    --max_seq_length=${max_seq_length} \
    # 	    --max_step=${step} \
    # 	    --device=${device} \
    # 	    --cuda_num=${device} \
    # 	    --epoch=${epochs} \
    # 	    --prate=${prate} \
    # 	    --target_num=${target_num} \
    # 	    --batch_size=${batch_size} \
    # 	    --lr=${lr} \
    # 	    --back_prediction=1 \
    # 	    --sample_method=exp \
    # 	    --gbias=0 \
    # 	    --pretrained_model_path=${pretrained_model_path} \
    # 	    --save_model_path=${save_model_path} \
    # 	    --board_name=${save_model_path} \
    # 	    --fraction=${fraction} 

    cd ../
    id_list=(1 2 3 4 5)

    export rate=${prate}
   
    for num_id in ${id_list[*]};
    do
	export dataset_name="multiwoz-2.1-"
	export model_save_path="_models_"

	export inference_model_name="data/exp-bp-target${target_num}_fraction1.0_prate${prate}/"
	export save_file_path="exp-bp-target${target_num}_fraction1.0_prate${prate}_numid${num_id}.txt"

	# $python evaluate_multiwoz.py \
	#     --dataset ${dataset_name}test \
	#     --model ${rate}${model_save_path}${num_id} \
	#     --file ${rate}_predict_files_${num_id}.txt \
	#     --save_file ${save_file_path} \
	#     --inference_model ${inference_model_name} \
	#     --add_map 1\
	#     --gbias 0\
	#     --bp 1\
	#     --cuda_num=${device}

	$python evaluate_control.py \
		--load_file_name $save_file_path
    done

    cd paraphrase/
done
##-----------------------------------------------------------------------------------------


echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Tempering wta"
# echo "--->>>BEGIN TO TRAINING in stage 1 with fraction ${frac}."
export target_num=-1
export fraction_list=("1.0" "1.0" "1.0")
for prate in ${prate_list[*]};
do
    export save_model_path="${root_dir}/data/tempering-horizontal3-wta_fraction1.0_prate${prate}/"

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
    id_list=(1 2 3 4 5)

    export rate=${prate}
   
    for num_id in ${id_list[*]};
    do
	export dataset_name="multiwoz-2.1-"
	export model_save_path="_models_"

	export inference_model_name="data/tempering-horizontal3-wta_fraction1.0_prate${prate}/"
	export save_file_path="tempering-horizontal3-wta_fraction1.0_prate${prate}_numid${num_id}.txt"

	# $python evaluate_multiwoz.py \
	#     --dataset ${dataset_name}test \
	#     --model ${rate}${model_save_path}${num_id} \
	#     --file ${rate}_predict_files_${num_id}.txt \
	#     --save_file ${save_file_path} \
	#     --inference_model ${inference_model_name} \
	#     --add_map 1\
	#     --gbias 0\
	#     --bp 0\
	#     --cuda_num=${device}

	$python evaluate_control.py \
		--load_file_name $save_file_path
    done

    cd paraphrase/
done
##-----------------------------------------------------------------------------------------

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Tempering exp bp"
# echo "--->>>BEGIN TO TRAINING in stage 1 with fraction ${frac}."
export target_num=-1
export pretrained_model_path="${root_dir}/t5-small" 
export fraction_list=("1.0" "1.0" "1.0")
for prate in ${prate_list[*]};
do
    export save_model_path="${root_dir}/data/tempering-exp-bp_fraction1.0_prate${prate}/"

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

	export inference_model_name="data/tempering-exp-bp_fraction1.0_prate${prate}/"
	export save_file_path="tempering-exp-bp_fraction1.0_prate${prate}_numid${num_id}.txt"

	# $python evaluate_multiwoz.py \
	#     --dataset ${dataset_name}test \
	#     --model ${rate}${model_save_path}${num_id} \
	#     --file ${rate}_predict_files_${num_id}.txt \
	#     --save_file ${save_file_path} \
	#     --inference_model ${inference_model_name} \
	#     --add_map 1\
	#     --gbias 0\
	#     --bp 1\
	#     --cuda_num=${device}

	$python evaluate_control.py \
		--load_file_name $save_file_path
    done

    cd paraphrase/
done
##-----------------------------------------------------------------------------------------



## past codes.
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################

# #!/bin/bash

# export python=/home/zliang/anaconda3/envs/dslz/bin/python3
# export root_dir="${HOME}/soloist/soloist/paraphrase/"

# ##-----------------------------------------------------------------------------------------
# export device="1"
# # export epochs=4
# export epochs=2
# export batch_size=4
# export lr=3e-5
# export max_seq_length=128
# # export import_pretrained_model_path="${root_dir}/data/electra-small-discriminator"
# export pretrained_model_path="${root_dir}/t5-small" 
# # export pretrained_model_path="${root_dir}/t5-v1-base" 
# export save_log_path="${root_dir}/log/fewshot_result_for_training_1020.log"
# # train stage.
# # export save_model_path="${root_dir}/data/rettig_model/"
# export fraction_list=( "0.3" "0.5" "0.7" "0.9")
# export fraction_list=("1.0" )

# echo ">>>>>>>>>>>NORMAL GENERATION"
# export save_model_path="${root_dir}/data/save_gbias_0_bp_0_rate_${fraction}_epoch_${epochs}/"
# # echo "--->>>BEGIN TO TRAINING in stage 1 with fraction ${frac}."
# for fraction in ${fraction_list[*]};
# do
#     ${python} train.py \
# 	    --train=1 \
# 	    --max_seq_length=${max_seq_length} \
# 	    --max_step=330 \
# 	    --device=${device} \
# 	    --cuda_num=${device} \
# 	    --epoch=${epochs} \
# 	    --batch_size=${max_seq_length} \
# 	    --lr=${lr} \
# 	    --back_prediction=0 \
# 	    --gbias=0 \
# 	    --pretrained_model_path=${pretrained_model_path} \
# 	    --save_model_path=${save_model_path} \
# 	    --fraction=${fraction} 

#     export pretrained_model_path=${save_model_path}
# done
# ##-----------------------------------------------------------------------------------------

# echo ">>>>>>>>>>>with BP GENERATION"
# export pretrained_model_path="${root_dir}/t5-small" 
# # export epochs=5
# export save_model_path="${root_dir}/data/save_gbias_0_bp_1_rate_${fraction}_epoch_${epochs}/"
# # echo "--->>>BEGIN TO TRAINING in stage 1 with fraction ${frac}."
# for fraction in ${fraction_list[*]};
# do
#     ${python} train.py \
# 	    --train=1 \
# 	    --max_seq_length=${max_seq_length} \
# 	    --max_step=550 \
# 	    --device=${device} \
# 	    --cuda_num=${device} \
# 	    --epoch=${epochs} \
# 	    --batch_size=${max_seq_length} \
# 	    --lr=${lr} \
# 	    --back_prediction=1 \
# 	    --gbias=0 \
# 	    --pretrained_model_path=${pretrained_model_path} \
# 	    --save_model_path=${save_model_path} \
# 	    --fraction=${fraction} 

#     export pretrained_model_path=${save_model_path}
# done
# ##-----------------------------------------------------------------------------------------

# echo ">>>>>>>>>>>with GBIAS GENERATION"
# export pretrained_model_path="${root_dir}/t5-small" 
# export save_model_path="${root_dir}/data/save_gbias_1_bp_0_rate_${fraction}_epoch_${epochs}/"
# # echo "--->>>BEGIN TO TRAINING in stage 1 with fraction ${frac}."
# # export epochs=5
# for fraction in ${fraction_list[*]};
# do
#     ${python} train.py \
# 	    --train=1 \
# 	    --max_seq_length=${max_seq_length} \
# 	    --max_step=550 \
# 	    --device=${device} \
# 	    --cuda_num=${device} \
# 	    --epoch=${epochs} \
# 	    --batch_size=${max_seq_length} \
# 	    --lr=${lr} \
# 	    --back_prediction=0 \
# 	    --gbias=1 \
# 	    --pretrained_model_path=${pretrained_model_path} \
# 	    --save_model_path=${save_model_path} \
# 	    --fraction=${fraction} 

#     export pretrained_model_path=${save_model_path}
# done
# ##-----------------------------------------------------------------------------------------

# echo ">>>>>>>>>>>with FULL GENERATION"
# export pretrained_model_path="${root_dir}/t5-small" 
# export save_model_path="${root_dir}/data/save_gbias_1_bp_1_rate_${fraction}_epoch_${epochs}/"
# # echo "--->>>BEGIN TO TRAINING in stage 1 with fraction ${frac}."
# for fraction in ${fraction_list[*]};
# do
#     ${python} train.py \
# 	    --train=1 \
# 	    --max_seq_length=${max_seq_length} \
# 	    --max_step=550 \
# 	    --device=${device} \
# 	    --cuda_num=${device} \
# 	    --epoch=${epochs} \
# 	    --batch_size=${max_seq_length} \
# 	    --lr=${lr} \
# 	    --back_prediction=1 \
# 	    --gbias=1 \
# 	    --pretrained_model_path=${pretrained_model_path} \
# 	    --save_model_path=${save_model_path} \
# 	    --fraction=${fraction} 

#     export pretrained_model_path=${save_model_path}
# done
