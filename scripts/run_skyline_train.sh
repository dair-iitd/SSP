for lang in hau ibo kin lug luo; do
    python iterate.py \
        -d ../llama/data/processed/test_${lang}.tsv \
        -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
        -td ../llama/data/processed/train_${lang}.tsv \
        -ssim data/${lang}_100/test_${lang}_allsrc_sim.npy \
        -tsim data/${lang}_100/test_${lang}_train_sim.npy \
        -tr 4 -sr 4 -m gpt-35-turbo \
        -p ner_8s_tsv_adapt -e 100 -y \
        -r new_results/gpt35/${lang}_4s4t_skyline_train;
done
