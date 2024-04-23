MODEL=$1
LANG=$2
PROMPT=$3
OUTPUT=$4

# 8t
if [ -z "$5" ]
then
    for lang in ${LANG}; do
    python iterate.py \
        -d data/processed/test_${lang}.tsv \
        -sd data/processed/test_${lang}.tsv \
        -td new_results/${MODEL}/americas/8s/${lang}/responses.json \
        -ssim data/processed/${lang}_tgt_tgt_sim.npy \
        -tsim data/processed/${lang}_tgt_tgt_sim.npy \
        -tr 8 -sr 0 -m ${MODEL} \
        -p ${PROMPT} -e 100 -y \
        -t 0.0 \
        -r ${OUTPUT}/${MODEL}/americas/8t/${lang}
    done
else
    for lang in ${LANG}; do
    python iterate.py \
        -d data/processed/test_${lang}.tsv \
        -sd data/processed/test_${lang}.tsv \
        -td new_results/${MODEL}/americas/8s/${lang}/responses.json \
        -ssim data/processed/${lang}_tgt_tgt_sim.npy \
        -tsim data/processed/${lang}_tgt_tgt_sim.npy \
        -tr 8 -sr 0 -m ${MODEL} \
        -p ${PROMPT} -e 100 -y \
        -t 0.0 \
        -r ${OUTPUT}/${MODEL}/americas/8t_skyline/${lang} \
        -glabel
    done
fi
