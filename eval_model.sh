# pip install -e .
CHECKPOINT=$1

TASKS=(MMLU ARC gsm)
NUM_GPUS=2

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
        --gen_kwargs temperature=0,top_k=1,top_p=0.7 \
        --tasks ${TASK_NAME} \
        --batch_size auto \
        --num_fewshot ${NUM_FEW_SHOT} \
        --output_path results/${TASK_NAME}/${CHECKPOINT}
    else
        lm_eval --model hf \
        --model_args pretrained=${CHECKPOINT},dtype=bfloat16,attn_implementation=flash_attention_2 \
        --gen_kwargs temperature=0,top_k=0,top_p=0 \
        --tasks ${TASK_NAME} \
        --num_fewshot ${NUM_FEW_SHOT} \
        --batch_size auto \
        --output_path results/${TASK_NAME}/${CHECKPOINT}
    fi
done
