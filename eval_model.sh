MODEL_NAME_OR_PATH=$1
MODEL_NAME=$2

# MODEL_NAME_OR_PATH=simonycl/data_selection_Llama-2-7b-hf-sharegpt_lora_step_2000
# MODEL_NAME_OR_PATH=/mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_2000

# sleep 5400

TASK="ARC"
BATCH_SIZE_PER_GPU=2

LIMIT=0.95

# for MODEL_NAME_OR_PATH in /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-lima_lora_merged; do
# for MODEL_NAME_OR_PATH in /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged; do
# for MODEL_NAME_OR_PATH in '/mnt/data/EasyLM/model/Llama-2-7b-hf-sharegpt-epoch_3' '/mnt/data/EasyLM/model/Llama-2-7b-hf-sharegpt-epoch_2'; do
for BATCH_SIZE_PER_GPU in \
    8 \
    ; do
    # for TASK in ARC gsm8k truthfulqa triviaqa hellaswag; do
    # for TASK in truthfulqa triviaqa hellaswag; do
    for TASK in ARC gsm8k math truthfulqa hellaswag; do
        if [[ $TASK == "ARC" ]]; then
            TASK_LIST="arc_challenge"
            FEW_SHOT=25
        elif [[ $TASK == "hellaswag" ]]; then
            TASK_LIST="hellaswag"
            FEW_SHOT=10
            LIMIT=2000
        elif [[ $TASK == "truthfulqa" ]]; then
            TASK_LIST="truthfulqa"
            FEW_SHOT=0
        elif [[ $TASK == "mmlu" ]]; then
            TASK_LIST="hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions"
            FEW_SHOT=5
        elif [[ $TASK == "winogrande" ]]; then
            TASK_LIST="winogrande"
            FEW_SHOT=5
        elif [[ $TASK == "math" ]]; then
            TASK_LIST="minerva_math"
            FEW_SHOT=3
	        LIMIT=500
        elif [[ $TASK == "gsm8k" ]]; then
            TASK_LIST="gsm8k"
            FEW_SHOT=5
        elif [[ $TASK == "triviaqa" ]]; then
            TASK_LIST="triviaqa"
            FEW_SHOT=5
        else
            echo "Unknown task: $TASK"
            exit 1
        fi

        echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH"
        echo "TASK: $TASK"
        echo "BATCH_SIZE_PER_GPU: $BATCH_SIZE_PER_GPU"

	if [[ $TASK == "gsm8k" ]]; then
        lm_eval --model vllm \
            --model_args="pretrained=${MODEL_NAME_OR_PATH},tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1" \
            --tasks=$TASK_LIST \
            --batch_size auto 
    elif [[ $TASK == "math" ]]; then
	    lm_eval --model vllm \
		    --model_args="pretrained=${MODEL_NAME_OR_PATH},tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1" \
		    --tasks=$TASK_LIST \
		    --batch_size auto \
		    --limit=$LIMIT
    elif [[ $TASK == "hellaswag" ]]; then
        lm_eval --model hf \
            --model_args="pretrained=${MODEL_NAME_OR_PATH},tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1" \
            --tasks=$TASK_LIST \
            --batch_size auto \
            --limit=$LIMIT
    else
	    lm_eval --model hf \
		    --model_args="pretrained=${MODEL_NAME_OR_PATH},tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1" \
		    --tasks=$TASK_LIST \
		    --batch_size auto
    fi

        # python3 main.py \
        #     --model=hf-causal-experimental \
        #     --model_args="pretrained=${MODEL_NAME_OR_PATH},use_accelerate=True,use_flash_attention_2=True" \
        #     --tasks=$TASK_LIST \
        #     --is_chat_format \
        #     --num_fewshot=$FEW_SHOT \
        #     --batch_size=$BATCH_SIZE_PER_GPU

        # echo "NO CHAT FORMAT"
        # python3 main.py \
        #     --model=hf-causal-experimental \
        #     --model_args="pretrained=${MODEL_NAME_OR_PATH},use_accelerate=True,use_flash_attention_2=True" \
        #     --tasks=$TASK_LIST \
        #     --num_fewshot=$FEW_SHOT \
        #     --batch_size=$BATCH_SIZE_PER_GPU 
    done
done

# for MODEL_NAME_OR_PATH in meta-llama/Llama-2-7b-hf simonycl/llama-2-7b-hf-sharegpt-full-ft; do
#     for BATCH_SIZE_PER_GPU in \
#         8 \
#         ; do
#         for TASK in hellaswag; do

#             if [[ $TASK == "ARC" ]]; then
#                 TASK_LIST="arc_challenge"
#                 FEW_SHOT=25
#             elif [[ $TASK == "hellaswag" ]]; then
#                 TASK_LIST="hellaswag"
#                 FEW_SHOT=10
#             elif [[ $TASK == "truthfulqa" ]]; then
#                 TASK_LIST="truthfulqa_mc"
#                 FEW_SHOT=0
#             elif [[ $TASK == "mmlu" ]]; then
#                 TASK_LIST="hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions"
#                 FEW_SHOT=5
#             elif [[ $TASK == "winogrande" ]]; then
#                 TASK_LIST="winogrande"
#                 FEW_SHOT=5
#             elif [[ $TASK == "gsm8k" ]]; then
#                 TASK_LIST="gsm8k"
#                 FEW_SHOT=5
#                 LIMIT=200
#             elif [[ $TASK == "triviaqa" ]]; then
#                 TASK_LIST="triviaqa"
#                 FEW_SHOT=5
#             else
#                 echo "Unknown task: $TASK"
#                 exit 1
#             fi

#             echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH"
#             echo "TASK: $TASK"
#             echo "BATCH_SIZE_PER_GPU: $BATCH_SIZE_PER_GPU"
#             python3 main.py \
#                 --model=hf-causal-experimental \
#                 --model_args="pretrained=${MODEL_NAME_OR_PATH},use_accelerate=True,use_flash_attention_2=True" \
#                 --tasks=$TASK_LIST \
#                 --num_fewshot=$FEW_SHOT \
#                 --is_chat_format \
#                 --batch_size=$BATCH_SIZE_PER_GPU \
#                 --device cuda:0         

#             python3 main.py \
#                 --model=hf-causal-experimental \
#                 --model_args="pretrained=${MODEL_NAME_OR_PATH},peft=/mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora,use_accelerate=True,use_flash_attention_2=True" \
#                 --tasks=$TASK_LIST \
#                 --num_fewshot=$FEW_SHOT \
#                 --batch_size=$BATCH_SIZE_PER_GPU \
#                 --device cuda:0
#         done
#     done
# done

# echo "TASK_LIST: $TASK_LIST"
# echo "FEW_SHOT: $FEW_SHOT"

# python3 main.py \
#     --model=hf-causal-experimental \
#     --model_args="pretrained=${MODEL_NAME_OR_PATH},use_accelerate=True,use_flash_attention_2=True" \
#     --tasks=$TASK_LIST \
#     --num_fewshot=$FEW_SHOT \
#     --batch_size=$BATCH_SIZE_PER_GPU \
#     --device cuda:0,1

# nohup bash eval_model.sh > logs/eval_model_Llama-2-7b-hf-sharegpt_lora_1.log 2>&1 &

