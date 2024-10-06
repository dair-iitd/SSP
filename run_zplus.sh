#for LANG in 'fo' 'gsw';
#do
#    for POOL in 32 64;
#        do
#            bash run_all_zgul_pos.sh ${LANG} ${POOL}
#        done
#done

for LANG in 'hau' 'ibo' 'kin' 'lug' 'luo';
#for LANG in 'hau';
do
    bash run_zplus_ner.sh ${LANG} ner_8s_tsv_adapt 100
done
