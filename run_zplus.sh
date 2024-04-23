lang=$1
for LANG in $lang;
    do
    bash scripts/run_zgul_plus_gpt4.sh $LANG ${LANG}'_80percile_grb_zplus_only_ex4'
    done
