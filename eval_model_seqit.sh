for CHECKPOINT_PATH in simonycl/self-seq-Meta-Llama-3-8B-flancot_full_sit_llama_70b_iter_2
do
    accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${CHECKPOINT_PATH},attn_implementation=flash_attention_2,use_chat_template=True \
    --gen_kwargs temperature=0,top_k=0,top_p=0 \
    --tasks mmlu_flan_cot_zeroshot \
    --batch_size auto \
    --output_path results/mmlu/${CHECKPOINT_PATH}_flan_cot_zeroshot \
    --use_cache db/mmlu_flan_cot_zeroshot.db \
    --cache_requests true
done