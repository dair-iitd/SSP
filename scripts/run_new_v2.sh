for lang in hau ibo kin lug luo; do
    python iterate.py \
        -d ../llama/data/processed/test_${lang}.tsv \
        -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
        -td new_results/gpt4/${lang}_8s/responses.json \
        -ssim data/${lang}_100/test_${lang}_allsrc_sim.npy \
        -tsim data/${lang}_100/test_${lang}_sim.npy \
        -tr 0 -sr 8 -m gpt-35-turbo \
        -p ner_8s_tsv_adapt -e 100 -y \
        -r new_results/gpt35/${lang}_8t_v2
done
