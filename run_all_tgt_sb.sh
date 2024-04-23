MODEL=$1
LANG=$2


# 8t
if [ -z "$3" ]
then
    for lang in ${LANG}; do
    python iterate.py \
        -d data/processed/test_${lang}.tsv \
        -sd data/processed/test_${lang}.tsv \
        -td new_results_sb/${MODEL}/americas/8s/${lang}/responses.json \
        -ssim data/processed/${lang}_tgt_tgt_sim.npy \
        -tsim data/processed/${lang}_tgt_tgt_sim.npy \
        -tr 8 -sr 0 -m ${MODEL} \
        -p nli_8s_tsv_adapt -e 100 -y \
        -t 0.0 \
        -r new_results_sb/${MODEL}/americas/8t/${lang}
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
        -p nli_8s_tsv_adapt -e 100 -y \
        -t 0.0 \
        -r new_results/${MODEL}/americas/8t/${lang} \
        -glabel
    done
fi
