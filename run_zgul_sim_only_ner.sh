lang=$1
for LANG in $lang;
    do
    bash scripts/run_zgul_gpt4_ner.sh $LANG ${LANG}'_80percile_grb_zgul_sim_only_ex8' 'sim' 'ner_8s_tsv_adapt' 
    done
