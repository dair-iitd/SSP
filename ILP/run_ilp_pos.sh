for LANG in $1;
#for LANG in 'fo' 'got' 'gsw';
    do
        python run_ilp_pos.py --lang $LANG --constr --it 1 --n_exemplars 8 --lamda $2 --data_dir $3 --pool_size $4
    done
