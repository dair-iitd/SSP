#SIM=$2
SAVE=$2
TSIM=$3
PROMPT=$4
MODEL="meta-llama/llama-2-70b-hf"
for lang in $1; do
    python iterate_llama_ilp.py \
        -d data/new/${lang}/test.tsv \
        -sd new_results/gpt45_new/germanic/8s/${lang}/responses.json \
        -td ../zgul_plus/outputs/${lang}_100_zgul_pred.tsv \
        -ssim data/new/${lang}/tgt_sim.npy \
        -tsim data/new/${lang}/${lang}_${TSIM}.npy \
        -tr 8 -sr 0 -m $MODEL \
        -p $PROMPT -e 100 -y \
        -l $lang \
        -r new_results/llama/ilp_results/${SAVE};
done
