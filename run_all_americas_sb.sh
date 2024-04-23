# 8s_random
# for lang in pa or as; do
#     python iterate.py \
#         -d ../naamapadam/${lang}_test.tsv \
#         -sd ../naamapadam/hi_bn_gu_mr_train.tsv \
#         -ssim ../naamapadam/sim/random_allsrc_sim.npy \
#         -tr 0 -sr 8 -m gpt-4 \
#         -p ner_8s_tsv_adapt -e 100 -y \
#         -r new_results/gpt4/indic/8s_random/${lang}
# done

# 8s
MODEL=$1
LANG=$2
for lang in ${LANG}; do
    python iterate.py \
        -d data/processed/test_${lang}.tsv \
        -sd data/processed/train_en_es.tsv \
        -ssim data/processed/test_${lang}_src_en_es_sim_sb.npy \
        -tr 0 -sr 8 -m ${MODEL} \
        -p nli_8s_tsv_adapt -e 100 -y \
        -t 0.0 \
        -r new_results_sb/${MODEL}/americas/8s/${lang}
done

# # 8t
# for lang in ${LANG}; do
#     python iterate.py \
#         -d data/processed/test_${lang}.tsv \
#         -sd ../naamapadam/hi_bn_gu_mr_train.tsv \
#         -td new_results/gpt4/indic/8s/${lang}/responses.json \
#         -ssim ../naamapadam/sim/${lang}_allsrc_sim.npy \
#         -tsim ../naamapadam/sim/${lang}_100_sim.npy \
#         -tr 8 -sr 0 -m gpt-4 \
#         -p ner_8s_tsv_adapt -e 100 -y \
#         -r new_results/gpt4/indic/8t/${lang}
# done
