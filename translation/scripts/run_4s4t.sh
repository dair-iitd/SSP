python iterate.py \
    -d ../llama/data/processed/test_luo.tsv \
    -td results/luo_100_8s_tsv_gpt4_allsrc_t05/responses.json \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -tsim ../llama/data/luo_100/test_luo_sim.npy \
    -ssim ../llama/data/luo_100/test_luo_allsrc_sim.npy \
    -e 100 -r results/luo_100_4s4t_tsv_gpt4_allsrc_t05

python iterate.py \
    -d ../llama/data/processed/test_ibo.tsv \
    -td results/ibo_100_8s_tsv_gpt4_allsrc_t05/responses.json \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -tsim ../llama/data/ibo_100/test_ibo_sim.npy \
    -ssim ../llama/data/ibo_100/test_ibo_allsrc_sim.npy \
    -e 100 -r results/ibo_100_4s4t_tsv_gpt4_allsrc_t05

python iterate.py \
    -d ../llama/data/processed/test_lug.tsv \
    -td results/lug_100_8s_tsv_gpt4_allsrc_t05/responses.json \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -tsim ../llama/data/lug_100/test_lug_sim.npy \
    -ssim ../llama/data/lug_100/test_lug_allsrc_sim.npy \
    -e 100 -r results/lug_100_4s4t_tsv_gpt4_allsrc_t05

python iterate.py \
    -d ../llama/data/processed/test_hau.tsv \
    -td results/hau_100_8s_tsv_gpt4_allsrc_t05/responses.json \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -tsim ../llama/data/hau_100/test_hau_sim.npy \
    -ssim ../llama/data/hau_100/test_hau_allsrc_sim.npy \
    -e 100 -r results/hau_100_4s4t_tsv_gpt4_allsrc_t05

python iterate.py \
    -d ../llama/data/processed/test_kin.tsv \
    -td results/kin_100_8s_tsv_gpt4_allsrc_t05/responses.json \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -tsim ../llama/data/kin_100/test_kin_sim.npy \
    -ssim ../llama/data/kin_100/test_kin_allsrc_sim.npy \
    -e 100 -r results/kin_100_4s4t_tsv_gpt4_allsrc_t05
