lang=$1
PROMPT=$2
POOL=$3
for LANG in $lang;
    do
    bash scripts/run_zplus_gpt4.sh $LANG ${LANG}'_80percile_grb_zplus_only_ex8_lam_100.0_'${POOL} '80percile_8ex_ILP_seq_grb_zgul_lam_100.0_'${POOL}'_v5' ${PROMPT}
    done
