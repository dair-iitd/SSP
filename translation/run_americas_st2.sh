MODEL=$1
#LANG=$2
PROMPT=$2
OUTPUT=$3

# 8t
if [ -z "$5" ]
then
    for lang in aym gn; do
    python iterate_nli.py \
        -d data/processed/test_${lang}.tsv \
        -sd data/processed/test_${lang}.tsv \
        -td new_results/${MODEL}/transtrain/8s/${lang}/responses.json \
        -ssim data/processed/${lang}_tgt_tgt_sim.npy \
        -tsim data/processed/${lang}_tgt_tgt_sim.npy \
        -tr 8 -sr 0 -m ${MODEL} \
        -p ${PROMPT} -e 10000 -y \
        -t 0.0 \
        -r ${OUTPUT}/${MODEL}/transtrain/8t/${lang}
    done
else
    for lang in ${LANG}; do
    python iterate_nli.py \
        -d data/processed/test_${lang}.tsv \
        -sd data/processed/test_${lang}.tsv \
        -td new_results/${MODEL}/americas/8s/${lang}/responses.json \
        -ssim data/processed/${lang}_tgt_tgt_sim.npy \
        -tsim data/processed/${lang}_tgt_tgt_sim.npy \
        -tr 8 -sr 0 -m ${MODEL} \
        -p ${PROMPT} -e 10000 -y \
        -t 0.0 \
        -r ${OUTPUT}/${MODEL}/americas/8t_skyline/${lang} \
        -glabel
    done
fi
