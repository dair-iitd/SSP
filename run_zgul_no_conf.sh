lang=$1
for LANG in $lang;
    do
    bash scripts/run_zgul_gpt4.sh $LANG ${LANG}'_80percile_grb_zgul_no_conf_ex8' '80percile_8ex_ILP_seq_grb_no_conf_cons_zgul'
    done
