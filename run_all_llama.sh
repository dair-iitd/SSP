set -e

MODEL="togethercomputer/LLaMA-2-7B-32K"
PROMPT="ner_8s_tsv_adapt"
N_EGS=100
OUTDIR="new_results/llama"

# 8s
# for lang in hau ibo kin lug luo; do
#     python iterate_llama.py \
#         -d data/new/${lang}/test.tsv \
#         -sd data/processed/train_en_conll-amh-swa-wol.tsv \
#         -ssim data/new/${lang}/src_sim.npy \
#         -tr 0 -sr 8 -m $MODEL \
#         -p $PROMPT -e $N_EGS -y \
#         -r $OUTDIR/african/8s/${lang}
# done
 
# 8t
for lang in ibo kin lug luo; do
    python iterate_llama.py \
        -d data/new/${lang}/test.tsv \
        -sd data/processed/train_en_conll-amh-swa-wol.tsv \
        -td $OUTDIR/african/8s/${lang}/responses.json \
        -ssim data/new/${lang}/src_sim.npy \
        -tsim data/new/${lang}/tgt_sim.npy \
        -tr 8 -sr 0 -m $MODEL \
        -p $PROMPT -e $N_EGS -y \
        -r $OUTDIR/african/8t/${lang}
done

# 8t_sky
# for lang in hau ibo kin lug luo; do
#     python iterate.py \
#         -d data/new/${lang}/test.tsv \
#         -sd data/processed/train_en_conll-amh-swa-wol.tsv \
#         -td data/new/${lang}/test.tsv \
#         -ssim data/new/${lang}/src_sim.npy \
#         -tsim data/new/${lang}/tgt_sim.npy \
#         -tr 8 -sr 0 -m $MODEL \
#         -p $PROMPT -e $N_EGS -y \
#         -r $OUTDIR/african/8t_sky/${lang}
# done

# 8s_random
# for lang in hau ibo kin lug luo; do
#     python iterate.py \
#         -d data/new/${lang}/test.tsv \
#         -sd data/processed/train_en_conll-amh-swa-wol.tsv \
#         -ssim data/random_100_allsrc.npy \
#         -tr 0 -sr 8 -m $MODEL \
#         -p $PROMPT -e $N_EGS -y \
#         -r $OUTDIR/african/8s_random/${lang}
# done
 
# 8t_random
# for lang in hau ibo kin lug luo; do
#     python iterate.py \
#         -d data/new/${lang}/test.tsv \
#         -sd data/processed/train_en_conll-amh-swa-wol.tsv \
#         -td new_results/gpt35_new/african/8s/${lang}/responses.json \
#         -ssim data/new/${lang}/src_sim.npy \
#         -tsim data/random_100_alltgt.npy \
#         -tr 8 -sr 0 -m $MODEL \
#         -p $PROMPT -e $N_EGS -y \
#         -r $OUTDIR/african/8t_random/${lang}
# done
 
# 8t_2
# for lang in hau ibo kin lug luo; do
#     python iterate.py \
#         -d data/new/${lang}/test.tsv \
#         -sd data/processed/train_en_conll-amh-swa-wol.tsv \
#         -td $OUTDIR/african/8t/${lang}/responses.json \
#         -ssim data/new/${lang}/src_sim.npy \
#         -tsim data/new/${lang}/tgt_sim.npy \
#         -tr 8 -sr 0 -m $MODEL \
#         -p $PROMPT -e $N_EGS -y \
#         -r $OUTDIR/african/8t_2/${lang}
# done
 
# 8t_3
# for lang in hau ibo kin lug luo; do
#     python iterate.py \
#         -d data/new/${lang}/test.tsv \
#         -sd data/processed/train_en_conll-amh-swa-wol.tsv \
#         -td $OUTDIR/african/8t_2/${lang}/responses.json \
#         -ssim data/new/${lang}/src_sim.npy \
#         -tsim data/new/${lang}/tgt_sim.npy \
#         -tr 8 -sr 0 -m $MODEL \
#         -p $PROMPT -e $N_EGS -y \
#         -r $OUTDIR/african/8t_3/${lang}
# done

python new_results/aggregate.py $OUTDIR
