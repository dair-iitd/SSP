MODEL=$1
PROMPT=$2
OUTPUT=$3
if [ -z "$4" ]
    then
    bash run_all_tgt.sh ${MODEL} aym ${PROMPT} ${OUTPUT}
    #bash run_all_tgt.sh ${MODEL} gn ${PROMPT} ${OUTPUT}
    #bash run_all_tgt.sh ${MODEL} quy ${PROMPT} ${OUTPUT}
    #bash run_all_tgt.sh ${MODEL} nah ${PROMPT} ${OUTPUT}

else
    for LANG in 'aym' 'gn' 'quy' 'nah'; 
    do
        bash run_all_tgt.sh ${MODEL} ${LANG} ${PROMPT} ${OUTPUT} ganno
    done
fi
