for lang in fo got gsw; do
    python iterate.py \
        -d data/${lang}_100/test_${lang}_100.tsv \
        -sd data/train_en-is-de-30k.tsv \
        -ssim data/${lang}_100/test_${lang}_100_allsrc_sim.npy \
        -tr 0 -sr 8 -m gpt-35-turbo \
        -p pos_8s_tsv_adapt -e 100 -y \
        -r new_results/gpt35_new/germanic/8s/${lang}
done;

for lang in fo got gsw; do
    python iterate.py \
        -d data/${lang}_100/test_${lang}_100.tsv \
        -sd data/train_en-is-de-30k.tsv \
        -td new_results/gpt35_new/germanic/8s/${lang}/responses.json \
        -ssim data/${lang}_100/test_${lang}_100_allsrc_sim.npy \
        -tsim data/${lang}_100/test_${lang}_100_sim.npy \
        -tr 8 -sr 0 -m gpt-35-turbo \
        -p pos_8s_tsv_adapt -e 100 -y \
        -r new_results/gpt35_new/germanic/8t/${lang}
done;
