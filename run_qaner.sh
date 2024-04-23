# for lang in luo; do
#     python iterate.py \
#         -d ../llama/data/processed/test_${lang}.tsv \
#         -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
#         -td new_results/gpt4/african/8s/${lang}/responses.json \
#         -ssim data/random_100_allsrc.npy \
#         -tsim data/${lang}_100/test_${lang}_sim.npy \
#         -tr 0 -sr 8 -m gpt-4 \
#         -p qaner -e 100 -y \
#         -r new_results/gpt4/african/zs_qaner/${lang}
# done
 
for lang in hau ibo kin lug luo; do
    python iterate.py \
        -d ../llama/data/processed/test_${lang}.tsv \
        -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
        -td new_results/gpt4/african/zs_qaner/${lang}/responses.json \
        -ssim data/random_100_allsrc.npy \
        -tsim data/${lang}_100/test_${lang}_sim.npy \
        -tr 8 -sr 0 -m gpt-4 \
        -p ner_8s_tsv_adapt -e 100 -y \
        -r new_results/gpt4/african/zs_qaner_8t/${lang}
done
