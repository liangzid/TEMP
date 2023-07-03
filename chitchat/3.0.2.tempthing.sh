
# export http_proxy=http://liangz:123456@202.117.54.213:80

export python=/home/liangzi/anaconda3/envs/dslz/bin/python3
export root_dir="${HOME}/adc/"
export device="6"
export epochs=2
export batch_size=1
export lr=3e-5
export max_seq_length=128
export pretrained_model_path="${root_dir}/t5-small" 
export save_log_path="${root_dir}/log/grid-step.log"
export target_num=1
export step_list=( "100000")
export prefix_inference_model="none"
export  safety_path="raw"
export is_using_ADC="1"
export max_step="100000"
echo "FOR BASELINES"
export is_adc=0
export adc_prefix_cls="perspectiveAPI"
export safetyModelPath="raw"

${python} train.py \
	--train=0 \
	--max_seq_length=${max_seq_length} \
	--max_step=$max_step \
	--device=${device} \
	--cuda_num=${device} \
	--epoch=${epochs} \
	--batch_size=${batch_size} \
	--lr=${lr} \
	--back_prediction=0 \
	--target_num=${target_num} \
	--board_name="mytest" \
	--sample_method="random" \
	--pretrained_model_path=${pretrained_model_path} \
	--save_model_path=${save_model_path} \
	--fraction=1.0 \
	--prefix_existing_model=${adc_prefix_cls} \
	--reading_safety_path=${safetyModelPath}\
	--is_using_ADC=${is_adc} 









