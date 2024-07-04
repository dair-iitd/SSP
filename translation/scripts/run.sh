# BUG!

python iterate.py \
    -d ../llama/data/processed/test_luo.tsv \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -ssim ../llama/data/luo_100/test_luo_sim.npy \
    -tsim ../llama/data/luo_100/test_luo_allsrc_sim.npy \
    -e 100 -r results/luo_100_8s_tsv_gpt4_allsrc_t05

python iterate.py \
    -d ../llama/data/processed/test_lug.tsv \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -ssim ../llama/data/lug_100/test_lug_sim.npy \
    -tsim ../llama/data/lug_100/test_lug_allsrc_sim.npy \
    -e 100 -r results/lug_100_8s_tsv_gpt4_allsrc_t05

python iterate.py \
    -d ../llama/data/processed/test_hau.tsv \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -ssim ../llama/data/hau_100/test_hau_sim.npy \
    -tsim ../llama/data/hau_100/test_hau_allsrc_sim.npy \
    -e 100 -r results/hau_100_8s_tsv_gpt4_allsrc_t05

python iterate.py \
    -d ../llama/data/processed/test_ibo.tsv \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -ssim ../llama/data/ibo_100/test_ibo_sim.npy \
    -tsim ../llama/data/ibo_100/test_ibo_allsrc_sim.npy \
    -e 100 -r results/ibo_100_8s_tsv_gpt4_allsrc_t05

python iterate.py \
    -d ../llama/data/processed/test_kin.tsv \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -ssim ../llama/data/kin_100/test_kin_sim.npy \
    -tsim ../llama/data/kin_100/test_kin_allsrc_sim.npy \
    -e 100 -r results/kin_100_8s_tsv_gpt4_allsrc_t05
