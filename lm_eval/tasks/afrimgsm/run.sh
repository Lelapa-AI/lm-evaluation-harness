models=(
#   dice-research/lola_v1
Davlan/afro-xlmr-large
#   UBC-NLP/serengeti
  )

for model in "${models[@]}"
do
  echo "Evaluating model: $model"
  for fewshot in 5 # 0 5
  do
    export OUTPUT_DIR=results/$fewshot
    mkdir -p "$OUTPUT_DIR"
    lm_eval --model hf   \
        --model_args pretrained=${model},trust_remote_code=True,parallelize=True\
        --tasks afrimgsm_en_cot_eng,mgsm_en_cot_en,mgsm_native_cot_en,afrimgsm_direct_eng,mgsm_direct_en  \
        --output_path "$OUTPUT_DIR" \
        --batch_size 1  \
        --num_fewshot $fewshot \
        --verbosity DEBUG \
        --limit 5
  done
done