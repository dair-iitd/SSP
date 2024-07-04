SIM=$2
SAVE=$3
for lang in $1; do
    python iterate.py \
        -d ../llama/data/processed/test_${lang}_100.tsv \
        -sd ../llama/data/processed/test_${lang}_100.tsv \
        -td ../llama/data/processed/test_${lang}_100.tsv \
        -ssim data/${lang}_100/${SIM} \
        -tsim data/${lang}_100/test_${lang}_sim.npy \
        -tr 0 -sr 8 -m gpt-35-turbo \
        -p ner_8s_tsv_adapt -e 100 -y \
        -r new_results/gpt35/${SAVE}_8t_zgul_plus;
done
