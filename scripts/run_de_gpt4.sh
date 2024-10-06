python iterate.py \
    -d ../llama/data/fo_100/test_fo_100.tsv \
    -sd ../llama/data/train_en-is-de-30k.tsv \
    -ssim ../llama/data/fo_100/test_fo_100_allsrc_sim.npy \
    -tr 0 -sr 8 -m gpt-4 \
    -r results/fo_100_8s_tsv_gpt4_allsrc_t05

python iterate.py \
    -d ../llama/data/got_100/test_got_100.tsv \
    -sd ../llama/data/train_en-is-de-30k.tsv \
    -ssim ../llama/data/got_100/test_got_100_allsrc_sim.npy \
    -tr 0 -sr 8 -m gpt-4 \
    -r results/got_100_8s_tsv_gpt4_allsrc_t05

python iterate.py \
    -d ../llama/data/gsw_100/test_gsw_100.tsv \
    -sd ../llama/data/train_en-is-de-30k.tsv \
    -ssim ../llama/data/gsw_100/test_gsw_100_allsrc_sim.npy \
    -tr 0 -sr 8 -m gpt-4 \
    -r results/gsw_100_8s_tsv_gpt4_allsrc_t05
