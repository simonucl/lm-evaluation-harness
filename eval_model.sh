pip install -e .

CHECKPOINT_PATHS=(
    meta-llama/Meta-Llama-3-8B-Instruct
    # Add your checkpoint paths here
)

TASKS=(MMLU truthfulqa ARC hellaswag gsm)
NUM_GPUS=2

for CHECKPOINT in ${CHECKPOINT_PATHS[@]}
do
    for TASK in ${TASKS[@]}
    do 
        if [[ $TASK == "MMLU" ]]; then
            TASK_NAME=mmlu
            NUM_FEW_SHOT=5
        elif [[ $TASK == "truthfulqa" ]]; then
            TASK_NAME=truthfulqa
            NUM_FEW_SHOT=0
        elif [[ $TASK == "ARC" ]]; then
            TASK_NAME=arc_challenge
            NUM_FEW_SHOT=25
        elif [[ $TASK == "hellaswag" ]]; then
            TASK_NAME=hellaswag
            NUM_FEW_SHOT=10
        elif [[ $TASK == "gsm" ]]; then
            TASK_NAME=gsm8k
            NUM_FEW_SHOT=5
        fi
    
        if [[ $TASK == "gsm" ]]; then
            lm_eval --model vllm \
            --model_args pretrained=${CHECKPOINT},tensor_parallel_size=${NUM_GPUS},dtype=auto,gpu_memory_utilization=0.9 \
            --gen_kwargs temperature=0,top_k=0,top_p=0 \
            --tasks ${TASK_NAME} \
            --batch_size auto \
            --num_fewshot ${NUM_FEW_SHOT} \
            --apply_chat_template \
            --output_path results/${TASK_NAME}/${CHECKPOINT}
        else
            accelerate launch -m lm_eval --model hf \
            --model_args pretrained=${CHECKPOINT},dtype=bfloat16,attn_implementation=flash_attention_2 \
            --gen_kwargs temperature=0,top_k=0,top_p=0 \
            --tasks ${TASK_NAME} \
            --num_fewshot ${NUM_FEW_SHOT} \
            --batch_size auto \
            --apply_chat_template \
            --output_path results/${TASK_NAME}/${CHECKPOINT}
        fi
    done
done