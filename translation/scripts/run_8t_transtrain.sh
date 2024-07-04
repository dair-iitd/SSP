SAVE_DIR=$1
for lang in hau ibo kin lug luo; do
    python iterate.py \
        -d data/new/${lang}/test.tsv \
        -sd ../iterate/data/processed/train_en_conll-amh-swa-wol.tsv \
        -td new_results/${lang}_8s_transtrain/responses.json \
        -ssim data/${lang}/tgt_sim.npy \
        -tsim data/${lang}/tgt_sim.npy \
        -tr 8 -sr 0 -m gpt-4-turbo \
        -p ner_8s_tsv_adapt -e 100 -y \
        -r new_results/${lang}_8t_transtrain/
done
