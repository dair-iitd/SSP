# python run_xnli.py \
#   --model_name_or_path ${1} \
#   --language de \
#   --train_language en \
#   --do_train \
#   --do_eval \
#   --per_device_train_batch_size 32 \
#   --learning_rate 5e-5 \
#   --num_train_epochs 2.0 \
#   --max_seq_length 128 \
#   --output_dir debug_xnli_3/ \
#   --save_steps -1
python run_xnli.py \
  --model_name_or_path debug_xnli_3 \
  --language ${1} \
  --tau ${2} \
  --train_language en \
  --do_predict \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 128 \
  --output_dir debug_xnli_2/ \
  --save_steps -1
