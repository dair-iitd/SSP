for lang in hau ibo kin lug luo; do
    python iterate.py \
        -d ../llama/data/processed/test_${lang}.tsv \
        -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
        -ssim data/random_100_allsrc.npy \
        -tr 0 -sr 8 -m gpt-4 \
        -p ner_8s_tsv_adapt -e 100 -y \
        -r new_results/gpt4/${lang}_8t_random
done
