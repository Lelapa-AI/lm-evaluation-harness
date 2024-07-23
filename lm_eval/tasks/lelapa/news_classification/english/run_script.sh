#!/bin/bash

models=(
  # "lelapa/lelapa-llama-13b-instruction-finetuned-with-lora-model-v2"
  # "google/flan-t5-xxl"
  # "bigscience/mt0-xxl-mt"
  # "CohereForAI/aya-101"
  # "bigscience/bloomz-7b1-mt"
  # "meta-llama/Meta-Llama-3-8B-Instruct"
  dice-research/lola_v1
  UBC-NLP/serengeti
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
            --model_args "pretrained=${model}",trust_remote_code=True,parallelize=True \
            --tasks $task \
            --batch_size 4 \
            --output_path "$OUTPUT_DIR" \
            --num_fewshot $fewshot \
            --verbosity DEBUG \
            --log_samples
  done
done