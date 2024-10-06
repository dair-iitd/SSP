#SIM=$2
SAVE=$2
TSIM=$3
PROMPT=$4
for lang in $1; do
    python zgul_ilp.py \
        -d data/new/${lang}/test.tsv \
        -sd new_results/gpt4_new/african/8s/${lang}/responses.json \
        -td data/transtrain/${lang}/${lang}_pred.tsv \
        -ssim data/new/${lang}/tgt_sim.npy \
        -tsim /home/vipul/llm/emnlp/vipul_pc/mbert_transtrain_outputs/${lang}_${TSIM}.npy \
        -tr 8 -sr 0 -m gpt-4 \
        -p $PROMPT -e 100 -y \
        -l $lang \
        -r new_results/gpt4_new/ilp_results_transtrain/${SAVE};
done
