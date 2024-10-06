for LANG in $1;
do
    bash run_zgul.sh $LANG 'pos_8s_tsv_adapt' $2
done
