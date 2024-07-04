#LANG=$2
for LANG in 'hau' 'ibo' 'kin' 'lug' 'luo'
#for LANG in 'luo'
do
for j in 2
do
	export MAX_LENGTH=320
	export BERT_MODEL=$1
	export OUTPUT_DIR=$1
	export TEXT_RESULT=${LANG}_test_result$j.txt
	export TEXT_PREDICTION=${LANG}_test_predictions$j.txt
	export BATCH_SIZE=1
	export NUM_EPOCHS=10
	export SAVE_STEPS=1000
	export SEED=$j
    #if [ -f "data/new/${LANG}/cached_test_en_mrl_afriberta_lr_320" ]; then
    #    rm data/new/${LANG}/cached_test_en_mrl_afriberta_lr_320
	#else
    #    echo "PASS"
    #fi
    CUDA_VISIBLE_DEVICES=0 python3 afriberta/infer_ner.py --data_dir data/new/${LANG}/ \
	--model_type bert \
	--model_name_or_path $BERT_MODEL \
	--output_dir $OUTPUT_DIR \
	--test_result_file $TEXT_RESULT \
	--test_prediction_file $TEXT_PREDICTION \
	--max_seq_length  $MAX_LENGTH \
	--num_train_epochs $NUM_EPOCHS \
	--per_gpu_eval_batch_size $BATCH_SIZE \
	--save_steps $SAVE_STEPS \
	--seed $SEED \
	--do_predict \
    --predict_langs ${LANG}
done
done
