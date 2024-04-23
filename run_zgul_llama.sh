lang=$1
PROMPT=$2
for LANG in $lang;
    do
    bash scripts/run_zgul_llama.sh $LANG ${LANG}'_80percile_grb_zgul_only_ex8' '80percile_8ex_ILP_seq_grb_zgul' ${PROMPT}
    done
