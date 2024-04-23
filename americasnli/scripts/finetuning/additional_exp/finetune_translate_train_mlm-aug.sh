#!/bin/bash



seed=${2}
lang=${1}

python run_finetuning.py \
  --collator_config configs/collator/finetuning/collator_default.yaml \
  --dataset_config configs/dataset/finetuning/anli_translate_train.yaml \
  --experiment_config configs/experiment/finetuning/nli_default.yaml \
  --model_config configs/model/finetuning/xlmr_pretrained_seq.yaml \
  --tokenizer_config configs/tokenizer/xlmr_tokenizer.yaml \
  --training_args configs/training_arguments/finetuning/translate_train.yaml \
  --trainer_config configs/trainer/default.yaml \
  experiment_name=translate_train_mlm-aug_${lang}_${seed} \
  seed=${seed} \
  use_wandb=True \
  log_directory=/projects/abeb4417/americasnli/logs/translate_train_mlm-aug/ \
  output_directory=/rc_scratch/abeb4417/americasnli/translate_train_mlm-aug/ \
  training_arguments.metric_for_best_model=eval_translate_train_${lang}_accuracy \
  dataset_settings.eval_dataset.language=${lang} \
  dataset_settings.train_dataset.language=${lang} \
  model_settings.init.pretrained_model_name_or_path=/rc_scratch/abeb4417/americasnli/mlm-translated-1.0/mlm_${lang}_42/final_model/ \
  slurm_type=sbatch

