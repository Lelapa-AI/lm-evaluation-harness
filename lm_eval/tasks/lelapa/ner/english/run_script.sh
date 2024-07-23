#!/bin/bash

models=(
  "lelapa/lelapa-llama-13b-instruction-finetuned-with-lora-model-v2"
  "google/flan-t5-xxl"
  "bigscience/mt0-xxl-mt"
  "CohereForAI/aya-101"
  "bigscience/bloomz-7b1-mt"
  "meta-llama/Meta-Llama-3-8B-Instruct"
)
task=ner_english_hausa,ner_english_isizulu,ner_english_swahili,ner_english_xhosa,ner_english_yoruba

for model in "${models[@]}"
do
  echo "Evaluating model: $model"
  for fewshot in 0 2 4 6 8
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


lm_eval --model openai-chat-completions \
        --model_args model=gpt-3.5-turbo \
        --tasks ner_english_hausa,ner_english_isizulu,ner_english_swahili,ner_english_xhosa,ner_english_yoruba \
        --output_path "./" \
        --verbosity DEBUG \
        --log_samples \
        --limit 5


lm_eval --model hf \
        --model_args "pretrained=bigscience/mt0-small",trust_remote_code=True \
        --tasks ner_english_hausa,ner_english_isizulu,ner_english_swahili,ner_english_xhosa,ner_english_yoruba \
        --device cuda:0 \
        --batch_size 16 \
        --output_path "./" \
        --verbosity DEBUG \
        --log_samples \
        --limit 5