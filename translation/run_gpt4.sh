OUTPUT_DIR=$1
#for LANG in 'hau' 'ibo' 'kin' 'lug' 'luo';
#for LANG in 'kin' 'lug';
for LANG in fo got gsw ;
do
    bash run_mbert_transtrain.sh $LANG ner_8s_tsv_adapt $OUTPUT_DIR
done
