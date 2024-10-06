#SIM=$2
SAVE=$2
TSIM=$3
PROMPT=$4
MODEL="meta-llama/llama-2-70b-hf"
for lang in $1; do
    #for P in 0.0;
    for P in 0.0 0.2 0.4 0.6 0.8 1.0;
    do
    python iterate_llama_ilp_noise.py \
        -d data/new/${lang}/test.tsv \
        -sd data/new/${lang}/test.tsv \
        -td data/new/${lang}/test.tsv \
        -ssim data/new/${lang}/tgt_sim.npy \
        -tsim data/new/${lang}/${lang}_${TSIM}.npy \
        -tr 8 -sr 0 -m $MODEL \
        -p $PROMPT -e 100 -y \
        -l $lang \
        -sp $P \
        -r /home/vipul/llm/iterate/new_results/llama/ilp_results/noise_analysis/${lang}'_prob_'${P}
    done
done
