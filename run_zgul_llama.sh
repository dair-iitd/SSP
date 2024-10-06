lang=$1
PROMPT=$2
#for VER in 'v1' 'v2' 'v3' 'v4';
for VER in 'v5';
do
for LANG in 'fo' 'got' 'gsw';
#for LANG in 'hau' 'ibo' 'kin' 'lug' 'luo';
    do
    for lamda in 1.0;
    #for lamda in 0.0 0.1 0.5 1.0 10.0 100.0;
    do
        #bash scripts/run_zgul_llama.sh $LANG ${LANG}'_80percile_grb_zgul_only_ex8_lam_'${lamda}'_'${VER} '80percile_8ex_ILP_seq_grb_zgul_lam_'${lamda}'_'${VER} ${PROMPT}
        bash scripts/run_zgul_llama.sh $LANG $LANG'_80percile_grb_ZGUL' '80percile_8ex_ILP_seq_grb_zgul' ${PROMPT}
    done
done
done
