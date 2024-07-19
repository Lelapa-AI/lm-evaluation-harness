models=(
  dice-research/lola_v1
  UBC-NLP/serengeti)

for model in "${models[@]}"
do
  echo "Evaluating model: $model"
  for fewshot in 0
  do
    export OUTPUT_DIR=results/$fewshot
    mkdir -p "$OUTPUT_DIR"
    lm_eval --model hf \
        --model_args pretrained=${model},trust_remote_code=True,parallelize=True  \
        --tasks afrimmlu_direct_amh,afrimmlu_direct_eng,afrimmlu_direct_ewe,afrimmlu_direct_fra,afrimmlu_direct_hau,afrimmlu_direct_ibo,afrimmlu_direct_kin,afrimmlu_direct_lin,afrimmlu_direct_lug,afrimmlu_direct_orm,afrimmlu_direct_sna,afrimmlu_direct_sot,afrimmlu_direct_twi,afrimmlu_direct_wol,afrimmlu_direct_xho,afrimmlu_direct_yor,afrimmlu_direct_zul   \
        --output_path "$OUTPUT_DIR" \
        --batch_size 1 \
        --num_fewshot $fewshot \
        --verbosity DEBUG
  done
done