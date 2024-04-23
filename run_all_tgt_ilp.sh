MODEL="meta-llama/llama-2-70b-hf"
LANG=$2
PROMPT=$3
OUTPUT=$4

# 8t
if [ -z "$5" ]
then
    for lang in ${LANG}; do
    python iterate_ilp_llama.py \
        -d data/processed/test_${lang}.tsv \
        -sd data/processed/test_${lang}.tsv \
        -td deberta_ft_outputs/${lang}_predictions.json \
        -ssim data/processed/${lang}_tgt_tgt_sim.npy \
        -tsim data/processed/${LANG}/${LANG}_90_per_ILP.npy \
        -tr 8 -sr 0 -m ${MODEL} \
        -p ${PROMPT} -e 100 -y \
        -t 0.0 \
        -r ${OUTPUT}/llama/americas/8t_ilp_grb/${lang}
    done
else
    for lang in ${LANG}; do
    python iterate_ilp.py \
        -d data/processed/test_${lang}.tsv \
        -sd data/processed/test_${lang}.tsv \
        -td new_results/${MODEL}/americas/8s/${lang}/responses.json \
        -ssim data/processed/${lang}_tgt_tgt_sim.npy \
        -tsim data/processed/${lang}_tgt_tgt_sim.npy \
        -tr 8 -sr 0 -m ${MODEL} \
        -p ${PROMPT} -e 100 -y \
        -t 0.0 \
        -r ${OUTPUT}/${MODEL}/americas/8t/${lang} \
        -glabel
    done
fi
