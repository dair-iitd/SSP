DATA_PATH="train.txt"  # <- Edit here

OUTPUT_EN_DIR="."  # <- Edit here
OUTPUT_TRANSL_DIR="trans_train_outputs_pos"  # <- Edit here
OUTPUT_DIR="trans_train_outputs_pos"


python pipelines/process_en_data_pos.py \
                    --input_path ${DATA_PATH} \
                    --output_dir ${OUTPUT_DIR}


ORG_SENT_PATH="${OUTPUT_DIR}/conll_en_train_org.txt"
ENTITY_PATH="${OUTPUT_DIR}/conll_en_train_entity.txt"
MODEL_NAME_OR_PATH="facebook/nllb-200-3.3B"
 
for TGT_LANG in hau_Latn ibo_Latn kin_Latn lug_Latn luo_Latn; do
   # TGT_LANG=$(sed -n "${idx}p" scripts/masakha_lang_ids.txt)
   python pipelines/nllb_translation_tasks.py \
                               --source_language eng_Latn \
                               --target_language $TGT_LANG \
                               --model_name_or_path $MODEL_NAME_OR_PATH \
                               --tokenizer_path $MODEL_NAME_OR_PATH \
                               --input_file_path $ORG_SENT_PATH \
                               --output_folder $OUTPUT_TRANSL_DIR \
                               --output_fname "conll_en_train_org_${TGT_LANG}.txt" \
                               --shard_num -1

   sacremoses -l en -j 4 tokenize  < "$OUTPUT_TRANSL_DIR/conll_en_train_org_${TGT_LANG}.txt" > "$OUTPUT_TRANSL_DIR/conll_en_train_org_${TGT_LANG}.txt.tok"

   python pipelines/nllb_translation_tasks.py \
                               --source_language eng_Latn \
                               --target_language $TGT_LANG \
                               --model_name_or_path $MODEL_NAME_OR_PATH \
                               --tokenizer_path $MODEL_NAME_OR_PATH \
                               --input_file_path $ENTITY_PATH \
                               --output_folder $OUTPUT_TRANSL_DIR \
                               --output_fname "conll_en_train_entity_${TGT_LANG}.txt" \
                               --shard_num -1
done
