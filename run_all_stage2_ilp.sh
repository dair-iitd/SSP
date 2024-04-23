MODEL=$1
PROMPT=$2
OUTPUT=$3
if [ -z "$4" ]
    then
    bash run_all_tgt_ilp.sh ${MODEL} aym ${PROMPT} ${OUTPUT}
    bash run_all_tgt_ilp.sh ${MODEL} gn ${PROMPT} ${OUTPUT}
    bash run_all_tgt_ilp.sh ${MODEL} quy ${PROMPT} ${OUTPUT}
    bash run_all_tgt_ilp.sh ${MODEL} nah ${PROMPT} ${OUTPUT}

else
    for LANG in 'gn' 'quy' 'nah'; 
    do
        bash run_all_tgt.sh ${MODEL} ${LANG} ${PROMPT} ${OUTPUT} ganno
    done
fi
