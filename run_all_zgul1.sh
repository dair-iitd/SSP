for LANG in 'hau' 'ibo' 'kin' 'lug' 'luo';
do
    bash run_zgul_llama.sh $LANG 'ner_8s_tsv_adapt'
done
