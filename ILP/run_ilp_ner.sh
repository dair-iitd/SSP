#for LANG in $2;
for LANG in 'hau' 'ibo' 'kin' 'lug' 'luo';
    do
        python run_ilp_ner.py --lang $LANG --constr --it 1 --n_exemplars 8 --lamda $1 --data_dir $2
    done
