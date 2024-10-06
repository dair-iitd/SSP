python iterate.py \
    -d ../llama/data/processed/test_lug.tsv \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -ssim data/lug_100/test_lug_allsrc_sim.npy \
    -tr 0 -sr 0 -m gpt-3.5-turbo \
    -p ner_lug_8s_tsv -e 100 \
    -r new_results/gpt35/lug_8s_fixed
