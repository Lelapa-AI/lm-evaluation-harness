#!/bin/bash

models=(
  # "bonadossou/afrolm_active_learning"
  # "Davlan/afro-xlmr-large"
  "bigscience/bloom"
  "bigscience/mt0-xxl-mt"
  "MaLA-LM/mala-500-10b-v2"
  # "dice-research/lola_v1"
  # "UBC-NLP/serengeti"
)
task=senti_english_swahili,senti_english_hausa,senti_english_yoruba

for model in "${models[@]}"
do
  echo "Evaluating model: $model"
  for fewshot in 0
  do
    export OUTPUT_DIR=results/$fewshot

    mkdir -p "$OUTPUT_DIR"

    lm_eval --model hf \
            --model_args "pretrained=${model}",trust_remote_code=True,parallelize=True \
            --tasks $task \
            --batch_size 2 \
            --output_path "$OUTPUT_DIR" \
            --num_fewshot $fewshot \
            --verbosity DEBUG \
            --log_samples
  done
done