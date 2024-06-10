#!/bin/bash

models=(
  "lelapa/lelapa-llama-13b-instruction-finetuned-with-lora-model-v2"
  "google/flan-t5-xxl"
  "bigscience/mt0-xxl-mt"
  "CohereForAI/aya-101"
  "bigscience/bloomz-7b1-mt"
  "meta-llama/Meta-Llama-3-8B-Instruct"
)
task=afrimgsm_direct_amh,afrimgsm_direct_eng,afrimgsm_direct_ewe,afrimgsm_direct_fra,afrimgsm_direct_hau,afrimgsm_direct_ibo,afrimgsm_direct_kin,afrimgsm_direct_lin,afrimgsm_direct_lug,afrimgsm_direct_orm,afrimgsm_direct_sna,afrimgsm_direct_sot,afrimgsm_direct_swa,afrimgsm_direct_twi,afrimgsm_direct_wol,afrimgsm_direct_xho,afrimgsm_direct_yor,afrimgsm_direct_zul

for model in "${models[@]}"
do
  echo "Evaluating model: $model"
  for fewshot in 0 2 4 6 8
  do
    export OUTPUT_DIR=results/$fewshot

    mkdir -p "$OUTPUT_DIR"

    lm_eval --model hf \
            --model_args "pretrained=MBZUAI/LaMini-Flan-T5-248M" \
            --tasks mmt_hau-eng,mmt_zul-eng,mmt_swa-eng,mmt_yor-eng\
            --device cuda:0 \
            --batch_size 16 \
            --output_path "$OUTPUT_DIR" \
            --num_fewshot $fewshot \
            --verbosity DEBUG \
            --limit 5 \
            --use_cache "./cache"
  done
done