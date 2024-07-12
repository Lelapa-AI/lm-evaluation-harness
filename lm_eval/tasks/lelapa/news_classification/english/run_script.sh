#!/bin/bash

models=(
  "bonadossou/afrolm_active_learning"
  "Davlan/afro-xlmr-large"
  "bigscience/bloom"
  "bigscience/mt0-xxl-mt"
  "MaLA-LM/mala-500-10b-v2"
  "dice-research/lola_v1"
  "UBC-NLP/serengeti"
)
task=nc_english_swahili,nc_english_hausa,nc_english_yoruba

for model in "${models[@]}"
do
  echo "Evaluating model: $model"
  for fewshot in 0 2
  do
    export OUTPUT_DIR=results/$fewshot

    mkdir -p "$OUTPUT_DIR"

    lm_eval --model hf \
            --model_args "pretrained=${model}",trust_remote_code=True \
            --tasks $task \
            --device cuda:0 \
            --batch_size 16 \
            --output_path "$OUTPUT_DIR" \
            --num_fewshot $fewshot \
            --verbosity DEBUG \
            --log_samples
  done
done