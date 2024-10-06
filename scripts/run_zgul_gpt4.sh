#SIM=$2
SAVE=$2
TSIM=$3
PROMPT=$4
for lang in $1; do
    python zgul_ilp.py \
        -d data/new/${lang}/test.tsv \
        -sd new_results/gpt4_new/african/8s/${lang}/responses.json \
        -td ../hpc_emnlp/zgul_outputs/${lang}_pred.tsv \
        -ssim data/new/${lang}/tgt_sim.npy \
        -tsim /home/vipul/llm/hpc_emnlp/zgul_outputs/${lang}_${TSIM}.npy \
        -tr 8 -sr 0 -m gpt-4-1106-preview \
        -p $PROMPT -e 100 -y \
        -l $lang \
        -r new_results/gpt4_1106_preview/ilp_results/${SAVE};
done
