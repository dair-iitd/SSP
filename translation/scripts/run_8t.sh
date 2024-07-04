SAVE_DIR=$1
for lang in fo gsw; do
    python iterate.py \
        -d data/new/${lang}/test.tsv \
        -sd ../iterate/data/processed/train_en_conll-amh-swa-wol.tsv \
        -td new_results/codec/${lang}/responses.json \
        -ssim data/new/${lang}/tgt_sim.npy \
        -tsim data/new/${lang}/tgt_sim.npy \
        -tr 8 -sr 0 -m gpt-4-turbo \
        -p pos_8s_tsv_adapt -e 100 -y \
        -r ${SAVE_DIR}/${lang}
done
