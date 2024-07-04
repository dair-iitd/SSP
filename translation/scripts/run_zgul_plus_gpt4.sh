#SIM=$2
SAVE=$2
for lang in $1; do
    python zgul_ilp.py \
        -d data/new/${lang}/test.tsv \
        -sd new_results/gpt4_new/african/8s/hau/responses.json \
        -td ../zgul_plus/outputs/${lang}_100_zgul_plus_pred.tsv \
        -ssim data/new/${lang}/tgt_sim.npy \
        -tsim data/new/${lang}/${lang}_80percile_8ex_ILP_seq_grb_zgul.npy \
        -tr 8 -sr 0 -m gpt-4 \
        -p ner_8s_tsv_adapt -e 100 -y \
        -l $lang \
        -r new_results/gpt4_new/ilp_results/${SAVE};
done
