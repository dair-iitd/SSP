lang=$1
PROMPT=$2
OUTPUT_DIR=$3
for LANG in $lang;
    do
    bash scripts/run_zgul_gpt4_transtrain_mbert.sh $LANG ${OUTPUT_DIR} '80percile_8ex_ILP_seq_grb_zgul_lam_1.0_v5' ${PROMPT}
    done
