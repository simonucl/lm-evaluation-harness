# for CHECKPOINT_PATH in simonycl/self-seq-Meta-Llama-3-8B-flancot_full_sit_llama_70b_iter_2
# for CHECKPOINT_PATH in simonycl/self-seq-Meta-Llama-3-8B-alpaca_it_llmam_70b simonycl/self-seq-Meta-Llama-3-8B-alpaca_sit_llama_70b simonycl/self-seq-Meta-Llama-3-8B-alpaca_sit_llama_70b
for CHECKPOINT_PATH in simonycl/self-seq-Meta-Llama-3-8B-wizardlm
do
    accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${CHECKPOINT_PATH},attn_implementation=flash_attention_2,use_chat_template=True	\
    --gen_kwargs temperature=0,top_k=0,top_p=0 \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path results/mmlu/${CHECKPOINT_PATH}

    # lm_eval --model vllm \
    # --model_args pretrained=${CHECKPOINT_PATH},tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.85,use_chat_template=True     \
    # --tasks mmlu_flan_cot_zeroshot \
    # --batch_size 2 \
    # --output_path results/mmlu/${CHECKPOINT_PATH}_flan_cot_zeroshot

    accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${CHECKPOINT_PATH},attn_implementation=flash_attention_2,use_chat_template=True     \
    --gen_kwargs temperature=0,top_k=0,top_p=0 \
    --tasks arc_challenge \
    --num_fewshot 25 \
    --batch_size auto \
    --output_path results/arc/${CHECKPOINT_PATH}
done
