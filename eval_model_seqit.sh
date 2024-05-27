# for CHECKPOINT_PATH in simonycl/self-seq-Meta-Llama-3-8B-flancot_full_sit_llama_70b_iter_2
# for CHECKPOINT_PATH in simonycl/self-seq-Meta-Llama-3-8B-alpaca_it_llmam_70b simonycl/self-seq-Meta-Llama-3-8B-alpaca_sit_llama_70b simonycl/self-seq-Meta-Llama-3-8B-alpaca_sit_llama_70b
for CHECKPOINT_PATH in simonycl/self-seq-Meta-Llama-3-8B-flancot_full_it_llama_70b simonycl/self-seq-Meta-Llama-3-8B-flancot_full_sit_llama_70b_iter_2 simonycl/self-seq-Meta-Llama-3-8B-alpaca_it_llmam_70b simonycl/self-seq-Meta-Llama-3-8B-alpaca_llmam_70b-iter-2
do
    accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${CHECKPOINT_PATH},attn_implementation=flash_attention_2,use_chat_template=True	\
    --gen_kwargs temperature=0,top_k=0,top_p=0 \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path results/mmlu/${CHECKPOINT_PATH}

done

for CHECKPOINT_PATH in simonycl/self-seq-Meta-Llama-3-8B-wizardlm simonycl/self-seq-Meta-Llama-3-8B-flancot_full_it_llama_70b simonycl/self-seq-Meta-Llama-3-8B-flancot_full_sit_llama_70b_iter_2 simonycl/self-seq-Meta-Llama-3-8B-alpaca_it_llmam_70b simonycl/self-seq-Meta-Llama-3-8B-alpaca_llmam_70b-iter-2
do
        accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${CHECKPOINT_PATH},attn_implementation=flash_attention_2,use_chat_template=True	\
    --gen_kwargs temperature=0,top_k=0,top_p=0 \
    --tasks mmlu \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path results/mmlu/${CHECKPOINT_PATH}_0shot

    accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${CHECKPOINT_PATH},attn_implementation=flash_attention_2,use_chat_template=True     \
    --gen_kwargs temperature=0,top_k=0,top_p=0 \
    --tasks arc_challenge \
    --num_fewshot 25 \
    --batch_size auto \
    --output_path results/arc/${CHECKPOINT_PATH}

done
