#SIM=$2
OUTPUT_DIR=$1
PROMPT="pos_8s_tsv_adapt"
echo $PROMPT
for lang in gsw;
    do
        python gpt4.py \
        -d data/new/${lang}/test.tsv \
        -sd ${OUTPUT_DIR}/${lang}_pred.tsv \
        -td ${OUTPUT_DIR}/${lang}_pred.tsv \
        -ssim data/hau/tgt_sim.npy \
        -tsim ${OUTPUT_DIR}/${lang}_80percile_8ex_ILP_seq_grb_zgul_lam_100.0_v5.npy \
        -tr 8 -sr 0 -m gpt-4-turbo \
        -p $PROMPT -e 100 -y \
        -l $lang \
        -r ${OUTPUT_DIR}/${lang}_gpt4;
    done
