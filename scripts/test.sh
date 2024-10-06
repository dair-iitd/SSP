python iterate.py \
    -d ../llama/data/processed/test_luo.tsv \
    -td results/luo_100_8s_tsv_allsrc_t05/responses.json \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -tsim ../llama/data/luo_100/test_luo_sim.npy \
    -ssim ../llama/data/luo_100/test_luo_allsrc_sim.npy \
    -e 10 -i 2 -r results/luo_100_4s4t_tsv_allsrc_t05
