#!/bin/bash

models=(
  # "bonadossou/afrolm_active_learning"
  # "Davlan/afro-xlmr-large"
  # "bigscience/bloom"
  # "bigscience/mt0-xxl-mt"
  # "MaLA-LM/mala-500-10b-v2"
  "dice-research/lola_v1"
  "UBC-NLP/serengeti"
)
task=mmt_english_eng-hau,mmt_english_eng-swa,mmt_english_eng-xho,mmt_english_eng-yor,mmt_english_eng-zul

for model in "${models[@]}"
do
  echo "Evaluating model: $model"
  for fewshot in 0 2
  do
    export OUTPUT_DIR=results/$fewshot

    mkdir -p "$OUTPUT_DIR"

    accelerate launch -m lm_eval --model hf \
            --model_args "pretrained=${model}",trust_remote_code=True \
            --tasks $task \
            --batch_size 4 \
            --output_path "$OUTPUT_DIR" \
            --num_fewshot $fewshot \
            --verbosity DEBUG \
            --log_samples
  done
done