#!/bin/bash

models=(
        dice-research/lola_v1
)
#task=afrimgsm_translate_swa,afrimgsm_translate_fra,afrimgsm_translate_yor,afrimgsm_translate_zul,afrimgsm_translate_xho,afrimgsm_translate_hau \ 
#task=afrimgsm_en_cot_eng,afrimgsm_en_cot_swa,afrimgsm_en_cot_fra,afrimgsm_en_cot_yor,afrimgsm_en_cot_zul,afrimgsm_en_cot_xho,afrimgsm_en_cot_hau \ 
task=afrimgsm_direct_eng,afrimgsm_direct_swa,afrimgsm_direct_fra,afrimgsm_direct_yor,afrimgsm_direct_zul,afrimgsm_direct_xho,afrimgsm_direct_hau \

for model in "${models[@]}"
do
  echo "Evaluating model: $model"
  for fewshot in 0 5 # 4 6 8
  do
        export OUTPUT_DIR=results/$fewshot

        mkdir -p "$OUTPUT_DIR"
        
        # --model_args "pretrained=${model}",trust_remote_code=True,parallelize=True,load_in_4bit=True,peft=lelapa/lelapa_SLM-ins_v1 \

        lm_eval --model hf \
                --model_args "pretrained=${model}",trust_remote_code=True,parallelize=True \
                --tasks $task \
                --batch_size 16 \
                --output_path "$OUTPUT_DIR" \
                --num_fewshot $fewshot \
                --verbosity DEBUG \
                --log_samples
  done
done