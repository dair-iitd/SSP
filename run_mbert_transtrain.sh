lang=$1
PROMPT=$2
for LANG in $lang;
    do
    bash scripts/run_zgul_gpt4_transtrain_mbert.sh $LANG ${LANG}'_80percile_8ex_ILP_seq_grb_zgul_lam_50.0_v5' '80percile_8ex_ILP_seq_grb_zgul_lam_50.0_v5' ${PROMPT}
    done
