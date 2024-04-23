# for lang in hau ibo kin lug luo; do
#     python iterate.py \
#         -d ../llama/data/processed/test_${lang}.tsv \
#         -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
#         -td new_results/gpt4/african/8s/${lang}/responses.json \
#         -ssim data/random_100_allsrc.npy \
#         -tsim data/random_100_alltgt.npy \
#         -tr 8 -sr 0 -m gpt-4 \
#         -p ner_8s_tsv_adapt -e 100 -y \
#         -r new_results/gpt4/african/8t_random/${lang}
# done

for lang in fo got gsw; do
    python iterate.py \
        -d ../llama/data/processed/test_${lang}.tsv \
        -sd ../llama/data/processed/train_en-is-de-30k.tsv \
        -td new_results/gpt4/germanic/8s/${lang}/responses.json \
        -ssim data/${lang}_100/test_${lang}_100_allsrc_sim.npy \
        -tsim data/random_100_alltgt.npy \
        -tr 8 -sr 0 -m gpt-4 \
        -p pos_8s_tsv_adapt -e 100 -y \
        -r new_results/gpt4/germanic/8t_random/${lang}
done
