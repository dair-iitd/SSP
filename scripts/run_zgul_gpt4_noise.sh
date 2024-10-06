#SIM=$2
SAVE=$2
TSIM=$3
PROMPT=$4
for lang in $1; do
    for P in 0.2 0.6;
    do
    python zgul_ilp_noise.py \
        -d data/new/${lang}/test.tsv \
        -sd data/new/${lang}/test.tsv \
        -td ../zgul_plus/outputs/${lang}_100_zgul_pred.tsv \
        -ssim data/new/${lang}/tgt_sim.npy \
        -tsim data/new/${lang}/${lang}_${TSIM}.npy \
        -tr 8 -sr 0 -m gpt-4 \
        -p $PROMPT -e 100 -y \
        -sp $P \
        -l $lang \
        -r new_results/gpt4_new/ilp_results/noise_analysis/${lang}'_prob_'${P};
    done
done
