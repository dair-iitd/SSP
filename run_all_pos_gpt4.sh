set -e

MODEL="gpt-4"
PROMPT="pos_8s_tsv_adapt"
N_EGS=100
OUTDIR="new_results/gpt4_new"

# # 8s
# for lang in fo got gsw; do
#     python iterate.py \
#         -d data/new/${lang}/test.tsv \
#         -sd ../llama/data/processed/train_en-is-de-30k.tsv \
#         -ssim data/new/${lang}/src_sim.npy \
#         -tr 0 -sr 8 -m $MODEL \
#         -p $PROMPT -e $N_EGS -y --slow \
#         -r $OUTDIR/germanic_new/8s/${lang}
# done
 
# # 8t
# for lang in fo got gsw; do
#     python iterate.py \
#         -d data/new/${lang}/test.tsv \
#         -sd ../llama/data/processed/train_en-is-de-30k.tsv \
#         -td $OUTDIR/germanic_new/8s/${lang}/responses.json \
#         -ssim data/new/${lang}/src_sim.npy \
#         -tsim data/new/${lang}/tgt_sim.npy \
#         -tr 8 -sr 0 -m $MODEL \
#         -p $PROMPT -e $N_EGS -y --slow \
#         -r $OUTDIR/germanic_new/8t/${lang}
# done

# 8t_decr
# for lang in fo got gsw; do
#     python iterate.py \
#         -d data/new/${lang}/test.tsv \
#         -sd ../llama/data/processed/train_en-is-de-30k.tsv \
#         -td $OUTDIR/germanic_new/8s/${lang}/responses.json \
#         -ssim data/new/${lang}/src_sim.npy \
#         -tsim data/new/${lang}/tgt_sim.npy \
#         -tr 8 -sr 0 -m $MODEL \
#         -p $PROMPT -e $N_EGS -y --slow \
#         -r $OUTDIR/germanic_new/8t_decr/${lang}
# done

# 8t_sky
for lang in got gsw; do
    python iterate.py \
        -d data/new/${lang}/test.tsv \
        -sd ../llama/data/processed/train_en-is-de-30k.tsv \
        -td data/new/${lang}/test.tsv \
        -ssim data/new/${lang}/src_sim.npy \
        -tsim data/new/${lang}/tgt_sim.npy \
        -tr 8 -sr 0 -m $MODEL \
        -p $PROMPT -e $N_EGS -y --slow \
        -r $OUTDIR/germanic_new/8t_sky/${lang}
done

# 8s_random
# for lang in fo got gsw; do
#     python iterate.py \
#         -d data/new/${lang}/test.tsv \
#         -sd ../llama/data/processed/train_en-is-de-30k.tsv \
#         -ssim data/random_100_allsrc.npy \
#         -tr 0 -sr 8 -m $MODEL \
#         -p $PROMPT -e $N_EGS -y \
#         -r $OUTDIR/germanic_new/8s_random/${lang}
# done
 
# 8t_random
# for lang in fo got gsw; do
#     python iterate.py \
#         -d data/new/${lang}/test.tsv \
#         -sd ../llama/data/processed/train_en-is-de-30k.tsv \
#         -td $OUTDIR/germanic_new/8s/${lang}/responses.json \
#         -ssim data/new/${lang}/src_sim.npy \
#         -tsim data/random_100_alltgt.npy \
#         -tr 8 -sr 0 -m $MODEL \
#         -p $PROMPT -e $N_EGS -y --slow \
#         -r $OUTDIR/germanic_new/8t_random/${lang}
# done
 
# 8t_2
# for lang in fo got gsw; do
#     python iterate.py \
#         -d data/new/${lang}/test.tsv \
#         -sd ../llama/data/processed/-is-de-30k-amh-swa-wol.tsv \
#         -td $OUTDIR/germanic_new/8t/${lang}/responses.json \
#         -ssim data/new/${lang}/src_sim.npy \
#         -tsim data/new/${lang}/tgt_sim.npy \
#         -tr 8 -sr 0 -m $MODEL \
#         -p $PROMPT -e $N_EGS -y --slow \
#         -r $OUTDIR/germanic_new/8t_2/${lang}
# done
 
# 8t_3
# for lang in fo got gsw; do
#     python iterate.py \
#         -d data/new/${lang}/test.tsv \
#         -sd ../llama/data/processed/train_en-is-de-30k.tsv \
#         -td $OUTDIR/germanic_new/8t_2/${lang}/responses.json \
#         -ssim data/new/${lang}/src_sim.npy \
#         -tsim data/new/${lang}/tgt_sim.npy \
#         -tr 8 -sr 0 -m $MODEL \
#         -p $PROMPT -e $N_EGS -y \
#         -r $OUTDIR/germanic_new/8t_3/${lang}
# done

python new_results/aggregate.py $OUTDIR
