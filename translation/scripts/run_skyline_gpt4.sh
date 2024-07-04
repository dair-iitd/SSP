for lang in hau; do #ibo kin lug luo; do
    python iterate.py \
        -d ../llama/data/processed/test_${lang}.tsv \
        -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
        -td ../llama/data/processed/test_${lang}.tsv \
        -ssim data/${lang}_100/test_${lang}_allsrc_sim.npy \
        -tsim data/${lang}_100/test_${lang}_sim.npy \
        -tr 8 -sr 0 -m gpt-4 \
        -p ner_8s_tsv_adapt -e 100 -y \
        -r new_results/gpt4/${lang}_8t_skyline;
done
