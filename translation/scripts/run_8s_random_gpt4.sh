#for lang in hau ibo kin lug luo; do
for lang in kin lug luo; do
    python gpt4.py \
        -d data/new/${lang}/test.tsv \
        -sd data/new/${lang}/train.tsv \
        -td data/new/${lang}/test.tsv \
        -ssim data/new/${lang}/rand.npy \
        -tsim data/new/${lang}/rand.npy \
        -tr 0 -sr 8 -m gpt-4-turbo \
        -p ner_8s_tsv_adapt -e 100 -y \
        -r new_results/${lang}_8s_transtrain
done
